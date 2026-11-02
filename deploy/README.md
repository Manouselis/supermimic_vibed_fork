# SuperMimic on G1 / Thor

**Training happened on the laptop. Thor will run inference in our own Docker image.**
The current software supports offline checks and receive-only hardware tests. Motor execution is not implemented.

## How the learning works

- **Policy:** the neural network choosing joint targets from robot feedback and the selected motion.
- **Reference clip:** a recorded movement to imitate. One shared policy handles the 40 clips.
- **Reward:** a score for matching body poses/velocities, moving smoothly and respecting joint limits.
- **Episode:** one simulated tracking attempt, ending at failure, a time limit or the clip's end.

Our training loop, in plain words:

1. Run the policy in the simulator. Save robot state, action and resulting next state in a replay buffer.
2. Train a **world model** on that experience. It learns to predict how the robot responds to actions.
3. Let the policy try short imagined sequences through this model. Calculate tracking reward for them.
4. Adjust policy weights by differentiating through those predictions. A **critic** estimates reward
   beyond the short imagined sequence. This implementation uses model-based policy optimization, not PPO.
5. Collect more simulator experience and repeat. Evaluate saved checkpoints on complete clips separately.

Training often starts partway through a clip and uses short episodes. Evaluation checks complete motions.
The world model helps train the policy, but is not required to run it on the robot. Its predictions
can be wrong, so policy improvements must be checked back in the physics simulator.
See [train.py](../supermimic/train.py) and [mdp.py](../supermimic/mdp.py) for the implementation.

## What runs on the robot

Every **20 ms / 50 Hz**, the inference process:

1. Reads joint angles/velocities and the pelvis IMU through native Unitree DDS.
2. Builds the last 10 observations and the next 10 reference frames.
3. Runs the frozen policy. It returns 29 actions, converted into joint-position targets.
4. In the future motor runner, sends validated targets to the G1's low-level controllers.
5. Reads the resulting feedback and repeats. No weights are updated on the robot.

Currently, **shadow mode stops at calculating and logging targets**. The robot stays under its existing
controller. Its previous predicted action is hypothetical because it was never applied.
A faster command-transport loop, startup transition, controller handover and tested stop procedure
still need implementation and approval before physical movement.

## Docker isolation

Our image is **`supermimic-g1-thor:dev`**. It contains Python, PyTorch, pinned SDK2 and CycloneDDS.
No host pip/apt installs, host venv, CUDA replacement, daemon changes or robot-service changes are needed.
Docker and, for GPU use, NVIDIA container support must already work on Thor. If missing, stop and review
that prerequisite rather than modifying the default environment automatically.

- Code, config and weights are read-only. Only project results are persisted to the host.
  Temporary runtime/cache files stay in the container's ephemeral `/tmp`.
- Offline commands use `--network none`. Only explicit `--robot` capture/shadow shares host networking for DDS.
- Runs use your UID and drop Linux capabilities. They mount neither the host root filesystem nor
  the Docker socket, and do not use privileged mode.
- `--gpu` explicitly exposes the GPU. Docker still shares the host kernel/driver, so compatibility needs checking.
- No ROS, Isaac Lab, world model or training process is needed inside the Thor deployment image.

The pinned candidate base is NVIDIA PyTorch **26.03**, ARM64, CUDA **13.2**, Ubuntu **24.04**.
Its ARM64 registry manifest was verified, but it has **not** been tested on this unconnected Thor.
Confirm its actual driver/JetPack first. [NVIDIA release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-03.html)
and [Thor container guide](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_cuda.html).

## Build and run on Thor

First follow [TASKS_TO_RUN.md](TASKS_TO_RUN.md) to inspect Thor and transfer this branch and the bundle.
From the repository root on Thor, with Docker already available:

```bash
bash deploy/docker/build.sh
bash deploy/docker/run.sh inspect
bash deploy/docker/run.sh check
bash deploy/docker/run.sh benchmark --samples 300
# GPU access and CUDA operation are checked separately:
bash deploy/docker/run.sh --gpu inspect
bash deploy/docker/run.sh --gpu --device cuda:0 benchmark --samples 300
```

CPU is the default. `build.sh` refuses a non-ARM64 host. A reviewed alternative base can be supplied
with `THOR_BASE_IMAGE`; your image tag can be changed with `THOR_DEPLOY_IMAGE`. No `latest` base is selected silently.

For robot feedback, replace the interface with the one found in the inventory:

```bash
bash deploy/docker/run.sh --robot capture --interface ACTUAL_INTERFACE --seconds 10
```

After checking the physical-review flags in [config.yaml](config.yaml):

```bash
bash deploy/docker/run.sh replay --input /runs/TAKE/samples.jsonl --clip walk1_subject1
bash deploy/docker/run.sh --robot shadow --interface ACTUAL_INTERFACE --clip walk1_subject1 --seconds 5
```

`walk1_subject1` is a software example, not an approved first physical motion.
`--gpu` and `--robot` must precede application options. CPU shadow needs only `--robot`.
For a custom profile, use `run.sh --config /absolute/profile.yaml ...`.
Config and weights mount at `/config/config.yaml` and `/bundle`. Results persist under host
`deploy/runs/`, visible inside the container as `/runs/`. Override host paths with `THOR_BUNDLE` or `THOR_RUNS`.
Each invocation creates a new result folder. No background service is installed. Ctrl+C stops diagnostics.

Checks reject invalid mappings, unsupported IMU/AB frames, mismatched start poses, stale feedback,
non-finite values, sudden/out-of-model hypothetical targets and missed deadlines. No targets are clipped silently.
`COMPLETE` means that diagnostic completed, not that a physical motion is safe or accurate.

## Files and verification

- [AUDIT.md](AUDIT.md): exact inputs/actions, source references and remaining hardware gaps.
- [VERIFIED.md](VERIFIED.md): measured local results, separate from untested Thor behavior.
- `deploy/artifacts/g1-thor-pilot.tar.gz`: local 66.7 MiB transfer archive, weights plus all 40 references.
- `deploy/run.py`: offline/capture/shadow CLI. `deploy/receiver.py` creates only a LowState subscriber.
- `deploy/docker/`: separate Dockerfile and build/run wrappers. The existing laptop training Dockerfile is unchanged.

All 40 references are packaged, but none is approved for physical execution. Their initial velocities
are nonzero. Simulation can initialize those states directly; the physical robot needs a reviewed transition.
All 40 completed local simulation tests, with self-collisions disabled. That is not hardware validation.

To reproduce local verification in the laptop simulator environment:

```bash
python -m unittest -v tools.test_deployment tools.test_deployment_docker tools.test_policy_release
python supermimic/evaluate.py --run_dir deploy/runs/sim-check \
  --ckpt deploy/artifacts/g1-lafan1-040000/policy --deployment-check \
  --result-json deploy/runs/sim-check/evaluation.json
```

The second command is laptop simulation only. It compares the adapter with the original pipeline each step.
On a laptop, direct `python -m deploy.run ...` commands remain available in a suitable local environment.
For rebuilding bundles, run `python -m deploy.run prepare --help`. Source hashes and matching model/40 CSVs
are mandatory. Keep the [motion attribution and licence terms](../docs/RELEASE_LICENSES.md) with the bundle.
