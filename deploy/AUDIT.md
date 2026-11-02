# Deployment audit

## What the policy expects

| Input/output | Actual implementation |
|---|---|
| Robot model | G1 29-DoF rev 1.0, including waist roll/pitch and both wrist pitch/yaw pairs |
| Policy period | 0.02 s / 50 Hz, not the raw CSV rate of 30 Hz |
| History | 10 × 99 values, oldest first. Repeat the first observation to initialize |
| Each observation | Joint position minus default/offset (29), unscaled velocity (29), pelvis gyro in body axes (3), projected gravity (3), previous normalized action (29), reference-relative pelvis orientation (6) |
| Reference | 10 future frames × 73 values = 730, built by the existing `MotionLib` |
| Rotation conventions | Pelvis-to-world quaternion **wxyz**. CSV input is **xyzw**. 6D rotation is the first two matrix columns, flattened in row order |
| Action | 29 values. `q_target = default_q + joint_offset + scale * action` |
| Scale and PD | Training `g1_spec.py` values. Scale is `0.25 * effort_limit / stiffness`. These are not verified hardware gains |

The policy does **not** require measured root position, root linear velocity, OptiTrack or cameras.
It does require the correct pelvis orientation and gyro. A torso IMU is not interchangeable.
Initial yaw is aligned once to the selected reference. Joint error and tilt are not fitted away.
No generic Unitree locomotion observation/action scales are substituted for this model's normalization.

## Supported now

- Verified bundle creation, including weights, normalization buffers, all 40 reference clips and model limits.
- Standalone PyTorch inference. Existing training/evaluation imports remain compatible.
- Source/export tensor comparison, original/deployment feature/action parity and simulator adapter check.
- Runtime inventory and CPU/CUDA inference benchmarking on the machine where the command runs.
- Native Unitree **receive-only** LowState capture, offline replay and live shadow inference.
- CRC, progressing ticks, freshness, finite input/output, mode/shape and diagnostic target checks.
- Receive ages use the local monotonic clock. They do not measure sensor-to-host transport latency.
- Explicit hardware-review gates. No motor publisher or control-mode client exists in this package.

## What remains unknown or unsupported

- The new G1 revision, unlocked joints, hands, Thor mounting/mass and firmware are unverified.
- Its installed JetPack/CUDA/Torch versions and SDK/DDS availability are unverified.
- Official G1 examples use a pelvis IMU and 29 motor slots, but this must be confirmed on this robot.
- The SDK low-level example writes at 2 ms. This policy runs at 20 ms. A future motor runner needs
  independently scheduled command transport and a tested watchdog/stop/handover procedure.
- Simulation uses implicit PD drives. Matching scalar gains alone does not prove hardware equivalence.
- Simulator reset copies **position, orientation, linear/angular velocity and joint position/velocity**
  from the reference. A physical robot cannot teleport into that state. A stationary pose match alone
  does not solve the starting-velocity problem.
- The local reference scan found a nonzero starting joint velocity in every clip. The lowest maximum
  joint speed at frame 0 was about **1.87 rad/s** (`walk1_subject5`). None is a verified standstill start.
- No physical startup transition, command streaming, end-of-clip stop, emergency-stop integration or
  controller handover is implemented. Those require a reviewed first-motion plan and explicit approval.
- No physical clip is approved. Jumps, falls, get-ups and fighting motions are excluded from first tests.

All 40 clips are familiar training motions. The original evaluation had no state/action noise and
disabled self-collisions. Completing them in simulation is not physical-robot validation.
Shadow also does not test balance or tracking, because its actions are never applied.

## Sources inspected

- [Unitree G1 low-level SDK example, pinned revision](https://github.com/unitreerobotics/unitree_sdk2_python/blob/814556d15970dd2ecf1c9984e845ca02ab07e206/example/g1/low_level/g1_low_level_example.py): joint order, PR/AB, LowState, 2 ms example command loop. **Do not run it as a diagnostic**, because it releases controllers and moves joints.
- [Unitree G1 deployment config](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e46c5d433564f24658bac64f254b7d2d4b/deploy/deploy_real/configs/g1.yaml) and [runner](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e46c5d433564f24658bac64f254b7d2d4b/deploy/deploy_real/deploy_real.py): pelvis IMU, wxyz and SDK feedback usage. Its locomotion policy is different from ours.
- [NVIDIA JetPack](https://developer.nvidia.com/embedded/jetpack): Thor uses the Arm/SBSA software stack. Inspect actual versions before selecting packages.
- [Our released policy](../docs/RELEASE.md), `mdp.py`, `motion.py`, `networks.py`, `g1_spec.py` and `env.py`: exact policy/training contract.

Local verification results are recorded in `deploy/VERIFIED.md`. No Thor was connected during preparation.
