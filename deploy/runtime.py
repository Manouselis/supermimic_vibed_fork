"""The exact policy inputs/outputs, independent of simulation and hardware transport."""

from pathlib import Path
import time
import threading

import torch

from supermimic.checkpoint import load_checkpoint
from supermimic import networks
from supermimic.motion import MotionLib
from supermimic.rotations import quat_to_mat, mat_to_6d, yaw_mat, rot_angle_sq
from .bundle import verify_bundle

_PRECISION_LOCK = threading.Lock()


def observation(q, qd, root_R, gyro_body, previous_action, reference_R, default_q, offsets):
    # Do not apply the generic Unitree locomotion example's scaling: our RunningNorm does it.
    gravity = -root_R[..., 2, :]
    relative_R = root_R.transpose(-1, -2) @ reference_R
    return torch.cat((q - default_q - offsets, qd, gyro_body, gravity,
                      previous_action, mat_to_6d(relative_R)), -1)


class PolicyRunner:
    """Reusable batched policy. Newest observation last, with repeat-first history initialization."""

    def __init__(self, checkpoint, default_q, action_scale, device='cpu', offsets=None):
        self.device = torch.device(device)
        ck = load_checkpoint(checkpoint, self.device)
        a = ck['args']
        if a['hist'] != 10 or a['goal_frames'] != 10:
            raise ValueError('Unsupported history/reference horizon')
        self.amp = a.get('amp', 'none') == 'bf16' and self.device.type == 'cuda'
        # Reuse the trained forward pass, including its fp32 tanh between bf16 MLP calls.
        self.pi = networks.Policy(99, 730, hist=10).to(self.device).eval()
        self.pi.load_state_dict(ck['pi'], strict=True)
        self.default_q = torch.as_tensor(default_q, dtype=torch.float32, device=self.device)
        self.scale = torch.as_tensor(action_scale, dtype=torch.float32, device=self.device)
        self.offsets = torch.zeros(29, device=self.device) if offsets is None else torch.as_tensor(offsets, dtype=torch.float32, device=self.device)
        if any(x.shape != (29,) or not bool(torch.isfinite(x).all()) for x in [self.default_q, self.scale, self.offsets]):
            raise ValueError('Invalid actuator/offset vectors')
        if not bool((self.scale > 0).all()):
            raise ValueError('Action scales must be positive')
        self.history = None
        self.previous_action = None

    def reset(self):
        self.history = self.previous_action = None

    @torch.inference_mode()
    def step(self, q, qd, root_R, gyro_body, reference_R, goal):
        expected = [(q, (29,)), (qd, (29,)), (root_R, (3, 3)), (gyro_body, (3,)),
                    (reference_R, (3, 3)), (goal, (730,))]
        batch = q.shape[0]
        if any(x.shape != (batch, *shape) or not bool(torch.isfinite(x).all()) for x, shape in expected):
            raise ValueError('Non-finite or incorrectly shaped policy input')
        if self.previous_action is None:
            self.previous_action = torch.zeros_like(q)
        if self.previous_action.shape != q.shape:
            raise ValueError('Reset history before changing batch size')
        obs = observation(q, qd, root_R, gyro_body, self.previous_action, reference_R,
                          self.default_q, self.offsets)
        self.history = (obs[:, None].repeat(1, 10, 1) if self.history is None
                        else torch.cat((self.history[:, 1:], obs[:, None]), 1))
        with _PRECISION_LOCK:
            previous_precision = networks.AMP_DTYPE
            try:
                networks.AMP_DTYPE = torch.bfloat16 if self.amp else None
                action = self.pi(self.history, goal).float()
            finally:
                networks.AMP_DTYPE = previous_precision
        target = self.default_q + self.offsets + self.scale * action
        if not bool(torch.isfinite(action).all() & torch.isfinite(target).all()):
            raise ValueError('Non-finite policy output')
        self.previous_action = action.clone()
        return action, target, obs


class Session:
    """One motion, one robot, no command interface. Hardware shadow feedback is NOT policy-controlled."""

    def __init__(self, bundle, clip, device='cpu', offsets=None):
        self.bundle = Path(bundle)
        self.config, self.spec, self.manifest = verify_bundle(bundle)
        if clip not in self.manifest['clips']:
            raise ValueError(f'Unknown clip: {clip}')
        self.motion = MotionLib([str(self.bundle / 'motions' / (clip + '.csv'))], None, device)
        self.runner = PolicyRunner(self.bundle / 'policy', self.spec['default_q'], self.spec['action_scale'], device, offsets)
        self.device = self.runner.device
        self.heading = None
        self.frame = 0

    def tensors(self, sample):
        indices = sample.get('motor_indices', list(range(29)))
        q = torch.tensor([[sample['q'][i] for i in indices]], dtype=torch.float32, device=self.device)
        qd = torch.tensor([[sample['qd'][i] for i in indices]], dtype=torch.float32, device=self.device)
        quat = torch.tensor([sample['quaternion_wxyz']], dtype=torch.float32, device=self.device)
        norm = float(torch.linalg.vector_norm(quat))
        if not 0.98 <= norm <= 1.02:
            raise ValueError('IMU quaternion norm outside 0.98..1.02')
        R = quat_to_mat(quat)
        gyro = torch.tensor([sample['gyro_body']], dtype=torch.float32, device=self.device)
        if not all(bool(torch.isfinite(x).all()) for x in [q, qd, R, gyro]):
            raise ValueError('Non-finite feedback')
        return q, qd, R, gyro

    def begin(self, sample, checks):
        q, qd, R, _ = self.tensors(sample)
        ref_R = self.motion.root_R[:1]
        if any(float(torch.linalg.vector_norm(x[0, :2, 0])) < .1 for x in [R, ref_R]):
            raise ValueError('Cannot align yaw when the pelvis forward axis is nearly vertical')
        # Match the unknown IMU-world heading once. Never fit away roll/pitch or joint errors.
        self.heading = yaw_mat(R) @ yaw_mat(ref_R).transpose(-1, -2)
        joint_error = float((q - self.motion.q[:1] - self.runner.offsets).abs().max())
        tilt_error = float(torch.sqrt(rot_angle_sq(R, self.heading @ ref_R)))
        speed = float(qd.abs().max())
        if (joint_error > checks['start_joint_error_rad'] or tilt_error > checks['start_tilt_error_rad']
                or speed > checks['start_joint_speed_rad_s']):
            raise ValueError(f'Start state does not match clip: joint={joint_error:.3f} rad, tilt={tilt_error:.3f} rad, speed={speed:.3f} rad/s. Do not move the robot automatically.')
        self.runner.reset()
        self.frame = 0
        return {'start_joint_error_rad': joint_error, 'start_tilt_error_rad': tilt_error,
                'heading_alignment': self.heading[0].cpu().tolist(),
                'reference_initial_joint_speed_max_rad_s': float(self.motion.qd[0].abs().max()),
                'reference_initial_root_speed_m_s': float(torch.linalg.vector_norm(self.motion.root_v[0])),
                'start_velocity_matching': 'NOT_ESTABLISHED_BY_STATIONARY_POSE_CHECK'}

    def step(self, sample, checks):
        if self.heading is None:
            raise ValueError('Call begin before stepping')
        if self.frame >= int(self.motion.clip_end[0]):
            raise StopIteration('Reference motion complete')
        q, qd, R, gyro = self.tensors(sample)
        lo = torch.tensor(self.spec['q_lo'], device=self.device)
        hi = torch.tensor(self.spec['q_hi'], device=self.device)
        if bool(((q - self.runner.offsets < lo) | (q - self.runner.offsets > hi)).any()):
            raise ValueError('Feedback exceeds training-model joint limits')
        f = torch.tensor([self.frame], device=self.device)
        start = time.perf_counter()
        action, target, _ = self.runner.step(q, qd, R, gyro, self.heading @ self.motion.root_R[f], self.motion.goal(f, 10))
        # .cpu() synchronizes CUDA, so timings include completion rather than just enqueue time.
        action_cpu, target_cpu = action[0].cpu(), target[0].cpu()
        elapsed = time.perf_counter() - start
        reasons = []
        if bool(((target[0] - self.runner.offsets < lo) | (target[0] - self.runner.offsets > hi)).any()):
            reasons.append('predicted_target_outside_model_limits')
        if float((target[0] - q[0]).abs().max()) > checks['max_target_step_rad']:
            reasons.append('predicted_target_too_far_from_feedback')
        self.frame += 1
        return {'frame': self.frame - 1, 'inference_s': elapsed, 'action': action_cpu.tolist(),
                'target_q': target_cpu.tolist(), 'checks': reasons}
