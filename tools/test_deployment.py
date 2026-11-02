"""Offline deployment checks. Optional artifact tests use DEPLOY_BUNDLE and ORIGINAL_CHECKPOINT."""

import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

import torch

from deploy.bundle import verify_bundle
from deploy.receiver import Receiver
from deploy.run import load_config, require_review, mapped_sample
from deploy.runtime import observation, PolicyRunner, Session
from supermimic.g1_spec import G1_JOINT_NAMES, default_joint_pos_list, action_scale
from supermimic import networks
from supermimic.motion import MotionLib
from supermimic.rotations import quat_to_mat, mat_to_6d, mat_to_quat, yaw_mat

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = Path(os.environ.get('DEPLOY_BUNDLE', ROOT / 'deploy/artifacts/g1-lafan1-040000'))


class ContractTest(unittest.TestCase):
    def reviewed_config(self, target):
        import yaml
        cfg = yaml.safe_load((ROOT/'deploy/config.yaml').read_text())
        for key in ['model_reviewed', 'joint_order_verified', 'imu_frame_verified', 'joint_offsets_verified']:
            cfg['robot'][key] = True  # synthetic receiver fixture only
        cfg['robot']['imu_frame'] = 'pelvis'
        target.write_text(yaml.safe_dump(cfg))
        return cfg

    def test_replay_cli_processes_timestamps_and_reports_unapplied_actions(self):
        from deploy.run import main
        from contextlib import redirect_stdout
        import io
        class FakeSession:
            def begin(self, *a): return {}
            def step(self, *a): return {'checks': [], 'inference_s': .001}
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = self.reviewed_config(tmp/'config.yaml')
            rows = [{'format':'g1-lowstate-capture-v1', 'robot_config':cfg['robot']}]
            rows.extend({'received_monotonic_s':100.+i*.02, 'tick':i+1, 'mode_pr':0} for i in range(3))
            (tmp/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with patch('deploy.run.get_session', return_value=FakeSession()), redirect_stdout(io.StringIO()):
                code = main(['--config',str(tmp/'config.yaml'),'--output',str(tmp/'run'),
                             'replay','--input',str(tmp/'samples.jsonl'),'--clip','fixture'])
            self.assertEqual(code, 0)
            report = json.loads((tmp/'run/report.json').read_text())
            self.assertEqual(report['samples'],3)
            self.assertEqual(report['previous_action_source'],'predicted_not_applied')

    def test_shadow_rejects_duplicate_tick_and_closes_receiver(self):
        from deploy.run import main
        from contextlib import redirect_stdout
        import io, time
        class FakeSession:
            def begin(self, *a): return {}
            def step(self, *a): return {'checks': [], 'inference_s': .001}
        class FakeReceiver:
            closed = False
            def __init__(self, *a): pass
            def snapshot(self, *a):
                return {'received_monotonic_s':time.monotonic(), 'tick':1, 'mode_pr':0, 'mode_machine':3}
            def close(self): FakeReceiver.closed = True
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            self.reviewed_config(tmp/'config.yaml')
            with patch('deploy.run.get_session',return_value=FakeSession()), \
                 patch('deploy.run.benchmark',return_value={'over_20ms':0}), \
                 patch('deploy.receiver.Receiver',FakeReceiver), redirect_stdout(io.StringIO()):
                code = main(['--config',str(tmp/'config.yaml'),'--output',str(tmp/'run'),
                             'shadow','--interface','fixture-only','--clip','fixture','--seconds','.08'])
            self.assertEqual(code,1)
            self.assertTrue(FakeReceiver.closed)
            self.assertIn('No new robot tick',json.loads((tmp/'run/report.json').read_text())['error'])

    def test_cli_failure_is_saved_and_does_not_overwrite(self):
        from deploy.run import main
        from contextlib import redirect_stdout
        import io
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'run'
            with redirect_stdout(io.StringIO()):
                code = main(['--output', str(out), '--bundle', str(Path(tmp)/'missing'), 'check'])
            self.assertEqual(code, 1)
            report = json.loads((out/'report.json').read_text())
            self.assertEqual(report['status'], 'FAILED')
            self.assertFalse(report['motor_publishing'])
            with self.assertRaises(FileExistsError):
                main(['--output', str(out), 'check'])

    def test_capture_cli_with_receive_only_fixture(self):
        from deploy.run import main
        from contextlib import redirect_stdout
        import io, time
        class FakeReceiver:
            closed = False
            tick = 0
            def __init__(self, *a): pass
            def snapshot(self, *a):
                self.tick += 1
                return {'received_monotonic_s': time.monotonic(), 'tick': self.tick, 'mode_pr':0,
                        'mode_machine':3, 'q':[0.]*35, 'qd':[0.]*35,
                        'quaternion_wxyz':[1.,0.,0.,0.], 'gyro_body':[0.,0.,0.]}
            def close(self): FakeReceiver.closed = True
        with tempfile.TemporaryDirectory() as tmp, patch('deploy.receiver.Receiver', FakeReceiver):
            out = Path(tmp)/'capture'
            with redirect_stdout(io.StringIO()):
                code = main(['--output',str(out),'capture','--interface','fixture-only','--seconds','.05'])
            self.assertEqual(code, 0)
            rows = [json.loads(x) for x in (out/'samples.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['format'], 'g1-lowstate-capture-v1')
            self.assertGreater(len(rows), 1)
            self.assertTrue(FakeReceiver.closed)

    def test_stale_capture_closes_receiver_and_saves_failure(self):
        from deploy.run import main
        from contextlib import redirect_stdout
        import io
        class BrokenReceiver:
            closed = False
            def __init__(self, *a): pass
            def snapshot(self, *a): raise ValueError('Stale robot feedback')
            def close(self): BrokenReceiver.closed = True
        with tempfile.TemporaryDirectory() as tmp, patch('deploy.receiver.Receiver', BrokenReceiver):
            out = Path(tmp)/'capture'
            with redirect_stdout(io.StringIO()):
                code = main(['--output',str(out),'capture','--interface','fixture-only'])
            self.assertEqual(code, 1)
            self.assertTrue(BrokenReceiver.closed)
            self.assertIn('Stale',json.loads((out/'report.json').read_text())['error'])

    def test_config_blocks_unreviewed_hardware(self):
        cfg = load_config(ROOT / 'deploy/config.yaml')
        with self.assertRaisesRegex(ValueError, 'review flags'):
            require_review(cfg)
        for key in ['model_reviewed', 'joint_order_verified', 'imu_frame_verified', 'joint_offsets_verified']:
            cfg['robot'][key] = True
        with self.assertRaisesRegex(ValueError, 'pelvis'):
            require_review(cfg)
        cfg['robot']['imu_frame'] = 'pelvis'
        require_review(cfg)

    def test_config_rejects_duplicate_mapping_and_nonfinite_offsets(self):
        import yaml
        original = yaml.safe_load((ROOT / 'deploy/config.yaml').read_text())
        for modify in [lambda c: c['robot']['motor_indices'].__setitem__(1, 0),
                       lambda c: c['robot']['joint_offsets_rad'].__setitem__(0, float('nan')),
                       lambda c: c['robot'].__setitem__('model_reviewed', 'false'),
                       lambda c: c['checks'].__setitem__('stale_s', 0)]:
            cfg = copy.deepcopy(original)
            modify(cfg)
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / 'config.yaml'
                p.write_text(yaml.safe_dump(cfg))
                with self.assertRaises(ValueError):
                    load_config(p)

    def test_no_hardware_writer_or_mode_switcher_in_deployment_code(self):
        forbidden = {'ChannelPublisher', 'LowCmd_', 'MotionSwitcherClient', 'LocoClient', 'ReleaseMode', 'CreateSendChannel'}
        for source in (ROOT / 'deploy').glob('*.py'):
            tree = ast.parse(source.read_text())
            used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
            self.assertFalse(forbidden & used, str(source))

    def test_no_simulator_import_for_inference(self):
        script = 'import deploy.runtime, sys; assert not any(n in sys.modules for n in ["isaaclab", "mujoco", "unitree_sdk2py"])'
        subprocess.run([sys.executable, '-c', script], cwd=ROOT, check=True)

    def test_observation_order_and_units(self):
        q = torch.arange(29).float()[None]
        qd, prev = q + 100, q - 100
        R = quat_to_mat(torch.tensor([[.7, .2, .3, .4]]))
        ref = quat_to_mat(torch.tensor([[.3, .1, .8, .2]]))
        gyro = torch.tensor([[1., 2., 3.]])
        default, offsets = torch.ones(29), torch.ones(29) * .2
        o = observation(q, qd, R, gyro, prev, ref, default, offsets)
        torch.testing.assert_close(o[:, :29], q - 1.2)
        torch.testing.assert_close(o[:, 29:58], qd)  # no generic locomotion velocity scaling
        torch.testing.assert_close(o[:, 58:61], gyro)
        torch.testing.assert_close(o[:, 61:64], -R[:, 2, :])
        torch.testing.assert_close(o[:, 64:93], prev)
        expected = (R.transpose(-1, -2) @ ref)[:, :, :2].reshape(1, 6)
        torch.testing.assert_close(o[:, 93:], expected)
        self.assertEqual(tuple(o.shape), (1, 99))

    def test_ab_feedback_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'A/B'):
            mapped_sample({'mode_pr': 1}, load_config(ROOT / 'deploy/config.yaml'))


class ReceiverTest(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.rx = Receiver.__new__(Receiver)
        self.rx.lock = threading.Lock()
        self.rx.clock = lambda: self.now
        self.rx.latest = self.rx.last_tick = self.rx.error = None
        self.rx.crc = types.SimpleNamespace(Crc=lambda msg: 123)

    def msg(self, tick):
        return types.SimpleNamespace(tick=tick, crc=123, mode_pr=0, mode_machine=3, version=[1, 2],
                motor_state=[types.SimpleNamespace(q=float(i), dq=0.) for i in range(35)],
                imu_state=types.SimpleNamespace(quaternion=[1., 0., 0., 0.], gyroscope=[0., 0., 0.]))

    def test_no_packet_then_fresh_then_stale(self):
        self.assertIsNone(self.rx.snapshot(.1))
        self.rx._callback(self.msg(10))
        self.assertEqual(self.rx.snapshot(.1)['q'][28], 28.)
        self.now += .2
        with self.assertRaisesRegex(ValueError, 'Stale'):
            self.rx.snapshot(.1)

    def test_duplicate_tick_does_not_refresh_age(self):
        self.rx._callback(self.msg(10))
        self.now += .2
        self.rx._callback(self.msg(10))
        with self.assertRaisesRegex(ValueError, 'Stale'):
            self.rx.snapshot(.1)

    def test_wrap_and_reboot(self):
        self.rx._callback(self.msg(2**32 - 2))
        self.rx._callback(self.msg(1))
        self.assertEqual(self.rx.snapshot(.1)['tick'], 1)
        self.rx._callback(self.msg(0))
        with self.assertRaisesRegex(ValueError, 'backwards'):
            self.rx.snapshot(.1)

    def test_crc_failure_latches(self):
        msg = self.msg(1)
        msg.crc = 999
        self.rx._callback(msg)
        self.rx._callback(self.msg(2))
        with self.assertRaisesRegex(ValueError, 'CRC'):
            self.rx.snapshot(.1)

    def test_actual_constructor_only_creates_subscriber(self):
        calls = []
        class Subscriber:
            def __init__(self, topic, kind): calls.append(('subscribe', topic))
            def Init(self, handler, queue): calls.append(('init', queue))
            def Close(self): calls.append(('close',))
        modules = {n: types.ModuleType(n) for n in ['unitree_sdk2py', 'unitree_sdk2py.core',
                   'unitree_sdk2py.core.channel', 'unitree_sdk2py.idl', 'unitree_sdk2py.idl.unitree_hg',
                   'unitree_sdk2py.idl.unitree_hg.msg', 'unitree_sdk2py.idl.unitree_hg.msg.dds_',
                   'unitree_sdk2py.utils', 'unitree_sdk2py.utils.crc']}
        channel = modules['unitree_sdk2py.core.channel']
        channel.ChannelFactoryInitialize = lambda domain, interface: calls.append(('factory', domain, interface))
        channel.ChannelSubscriber = Subscriber  # deliberately no publisher attribute
        modules['unitree_sdk2py.idl.unitree_hg.msg.dds_'].LowState_ = object
        modules['unitree_sdk2py.utils.crc'].CRC = lambda: self.rx.crc
        with patch.dict(sys.modules, modules):
            rx = Receiver('test-interface', 9)
            rx.close()
        self.assertEqual(calls, [('factory', 9, 'test-interface'), ('subscribe', 'rt/lowstate'), ('init', 0), ('close',)])


@unittest.skipUnless((BUNDLE / 'bundle.json').is_file(), 'Set DEPLOY_BUNDLE to test released weights and real motions')
class ReleasedArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config, cls.spec, cls.manifest = verify_bundle(BUNDLE)

    def runner(self):
        return PolicyRunner(BUNDLE / 'policy', self.spec['default_q'], self.spec['action_scale'])

    def test_original_checkpoint_and_release_match(self):
        original = os.environ.get('ORIGINAL_CHECKPOINT')
        if not original:
            self.skipTest('Set ORIGINAL_CHECKPOINT for source/export parity')
        old = torch.load(original, map_location='cpu', weights_only=True)['pi']
        new = self.runner().pi.state_dict()
        self.assertEqual(set(old), set(new))
        for name in old:
            torch.testing.assert_close(old[name], new[name], rtol=0, atol=0)

    def test_history_actions_and_targets_match_original_forward(self):
        runner = self.runner()
        torch.manual_seed(1)
        history, previous = None, torch.zeros(2, 29)
        frozen = {n: x.clone() for n, x in runner.pi.state_dict().items() if '_norm.' in n}
        for i in range(12):
            q, qd, gyro = torch.randn(2, 29), torch.randn(2, 29), torch.randn(2, 3)
            R, ref = quat_to_mat(torch.randn(2, 4)), quat_to_mat(torch.randn(2, 4))
            goal = torch.randn(2, 730)
            obs = observation(q, qd, R, gyro, previous, ref, runner.default_q, runner.offsets)
            history = obs[:, None].repeat(1, 10, 1) if history is None else torch.cat((history[:, 1:], obs[:, None]), 1)
            with torch.inference_mode():
                expected = runner.pi(history, goal)
            action, target, actual_obs = runner.step(q, qd, R, gyro, ref, goal)
            torch.testing.assert_close(action, expected, rtol=0, atol=0)
            torch.testing.assert_close(target, runner.default_q + runner.scale * expected, rtol=0, atol=0)
            torch.testing.assert_close(runner.history, history, rtol=0, atol=0)
            previous = action
        self.assertIsNone(networks.AMP_DTYPE)
        for name, value in frozen.items():
            torch.testing.assert_close(value, runner.pi.state_dict()[name], rtol=0, atol=0)

    def test_finite_and_shape_guards(self):
        runner = self.runner()
        values = [torch.zeros(1,29), torch.zeros(1,29), torch.eye(3)[None], torch.zeros(1,3), torch.eye(3)[None], torch.zeros(1,730)]
        for i in range(len(values)):
            broken = [v.clone() for v in values]
            broken[i].flatten()[0] = float('nan')
            with self.assertRaisesRegex(ValueError, 'policy input'):
                runner.step(*broken)
        values[-1] = torch.zeros(1,729)
        with self.assertRaises(ValueError): runner.step(*values)

    def test_all_40_clips_reference_features_and_padding(self):
        class FakeKin:
            def forward(self, p, R, v, w, q, qd): return p[:,None], R[:,None], v[:,None], w[:,None]
        for name in self.manifest['clips']:
            source = [str(BUNDLE / 'motions' / (name + '.csv'))]
            full = MotionLib(source, FakeKin(), 'cpu')
            light = MotionLib(source, None, 'cpu')
            frames = torch.tensor([0, 1, int(light.clip_end[0])//2, int(light.clip_end[0])-2, int(light.clip_end[0])])
            torch.testing.assert_close(full.goal(frames,10), light.goal(frames,10), rtol=0, atol=0)
            goal = light.goal(torch.tensor([int(light.clip_end[0])]), 10).reshape(1,10,73)
            torch.testing.assert_close(goal[:,:,9:15], torch.zeros(1,10,6), rtol=0, atol=0)
            torch.testing.assert_close(goal[:,:,44:], torch.zeros(1,10,29), rtol=0, atol=0)

    def test_heading_alignment_does_not_fit_tilt_or_joints(self):
        from deploy.run import reference_sample
        session = Session(BUNDLE, 'walk1_subject1')
        cfg = load_config(ROOT / 'deploy/config.yaml')['checks']
        sample = reference_sample(session, 0)
        sample['qd'] = [0.] * 29
        R = quat_to_mat(torch.tensor([sample['quaternion_wxyz']]))
        heading = quat_to_mat(torch.tensor([[.70710678,0.,0.,.70710678]]))
        sample['quaternion_wxyz'] = mat_to_quat(heading @ R)[0].tolist()
        session.begin(sample, cfg)
        torch.testing.assert_close(session.heading, heading, atol=1e-5, rtol=1e-5)
        bad = copy.deepcopy(sample)
        bad['q'][0] += .5
        with self.assertRaisesRegex(ValueError, 'Start state'): session.begin(bad, cfg)
        bad = copy.deepcopy(sample)
        bad['quaternion_wxyz'] = [0.,0.,0.,0.]
        with self.assertRaisesRegex(ValueError, 'quaternion norm'): session.begin(bad, cfg)

    def test_bundle_hash_rejects_changed_file(self):
        from deploy.bundle import verify_files
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'policy.safetensors'
            p.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                verify_files(tmp, {'policy.safetensors': '0' * 64})
            with self.assertRaises(ValueError): verify_files(tmp, {'../outside': '0'*64})


if __name__ == '__main__':
    unittest.main()
