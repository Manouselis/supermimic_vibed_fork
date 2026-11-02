# Next session: G1 with Thor

Goal for the first connection: **inspect the machine, receive feedback and test inference without moving the robot**.
All commands below are diagnostic. They do not take control of the G1.

## 1. Connect and identify the correct robot

1. Connect the laptop to the **new G1/Thor**, following your normal robot-network procedure.
   Use a network with only this G1 publishing `LowState`.
2. Send the new Thor SSH address and username in this chat. Do not reuse the old robot address.
3. Keep the robot stationary under its existing supported controller. Have its operator/remote available.
4. On the laptop, run the inventory, using the actual address:

```bash
cd /home/pantelis/work_ws/humanoid_ws/src/supermimic_thor_deploy
bash deploy/check_thor.sh USER@NEW_THOR_IP
```

This sends the inventory script over SSH stdin. It does not install files, subscribe to robot data,
release a controller, start a container or change services. Review architecture, OS/L4T, CUDA/Torch,
GPU operation, SDK availability and interface names before installing anything.

## 2. Transfer code and the prepared bundle

The branch is local until explicitly published. Once ready to transfer it through Git:

```bash
# Laptop
git push -u origin feature/thor-deployment-prep
```

On Thor, clone a separate checkout, or use an existing clean checkout after reviewing `git status`:

```bash
git clone --branch feature/thor-deployment-prep \
  https://github.com/Manouselis/supermimic_vibed_fork.git ~/supermimic_thor_deploy
```

The weights/reference archive is intentionally outside Git. Transfer the prepared archive:

```bash
# Laptop. Substitute USER and NEW_THOR_IP.
scp deploy/artifacts/g1-thor-pilot.tar.gz deploy/artifacts/g1-thor-pilot.tar.gz.sha256 \
  USER@NEW_THOR_IP:~/supermimic_thor_deploy/
```

On Thor:

```bash
cd ~/supermimic_thor_deploy
sha256sum -c g1-thor-pilot.tar.gz.sha256
mkdir -p deploy/artifacts
test ! -e deploy/artifacts/g1-lafan1-040000
tar -xzf g1-thor-pilot.tar.gz -C deploy/artifacts
python3 deploy/inspect_host.py
```

Stop if checksums fail or `deploy/artifacts/g1-lafan1-040000` already contains a different bundle.
Do not overwrite an existing deployment or reuse its configuration without review.

## 3. Select the supported Python runtime

1. Use the inventory to find a working Thor-compatible Torch installation. Do not copy the laptop's
   x86 Docker image or replace system CUDA. If Torch is missing/broken, let us choose its installation
   from the verified OS/JetPack versions first.
2. Create an isolated environment that can reuse that compatible installation:

```bash
python3 -m venv --system-site-packages .venv-thor
source .venv-thor/bin/activate
python -m pip install -r deploy/requirements.txt
python -m deploy.run inspect
python -m deploy.run check
python -m deploy.run benchmark --samples 300
```

`check` must complete with all 40 clips. Benchmark reports bundle/session loading separately from warm
inference after 30 warm-up calls. Its `over_20ms` must be zero before shadow.
CPU is the default. Test CUDA separately only if the inventory's CUDA operation succeeds:

```bash
python -m deploy.run --device cuda:0 benchmark --samples 300
```

3. Confirm the **existing** SDK2/CycloneDDS imports in this environment. If absent, stop and install
   the official SDK against the inspected system libraries. No SDK motion example needs to be run.

## 4. Capture feedback without inference

Identify the Thor interface connected to the robot from the inventory. Replace `ACTUAL_INTERFACE`:

```bash
python -m deploy.run capture --interface ACTUAL_INTERFACE --seconds 10
```

Review the new `deploy/runs/.../report.json` and `samples.jsonl`:

- Valid CRCs and progressing ticks, no stale feedback or long sample gaps.
- Correct motor count/order and PR feedback (`mode_pr=0`).
- Finite joint angles/velocities and sensible stationary values.
- Quaternion order **wxyz**, verified pelvis-frame orientation/gyro.

If packets do not arrive, diagnose networking/SDK configuration. Do not disable the existing controller.

## 5. Review the model, then try receive-only inference

1. Confirm the physical G1 is compatible with the 29-DoF training model. Check the waist, wrists,
   hands, extra Thor mass/mounting and firmware. Record mismatches rather than declaring them harmless.
2. Review `deploy/config.yaml`. Set each review flag true only after checking it. Set `imu_frame: pelvis`
   only when that is established. Zero offsets mean no added encoder correction, not measured accuracy.
3. Choose a clip after inspecting its initial pose and velocity. `walk1_subject1` is a convenient
   software example, **not an approved first physical motion**.
4. A captured stationary pose often will not match the selected clip. A start-state rejection is
   expected in that case. Do not move the robot merely to bypass this check.

```bash
python -m deploy.run replay --input deploy/runs/TAKE/samples.jsonl --clip walk1_subject1
python -m deploy.run shadow --interface ACTUAL_INTERFACE --clip walk1_subject1 --seconds 5
```

Shadow keeps the robot under its existing controller. Inspect hypothetical targets, timing and any
rejection reason. It cannot prove balance, tracking accuracy or physical motion success.

## 6. Stop before motor execution

Send the reports back in this chat. We must review model differences, initialization and the first
motion, then implement/review the command loop, controller handover, watchdog, safe stop and operator
emergency-stop procedure. Motor execution requires a separate explicit approval.
There is no command in this version that moves the robot, and no automatic sequence of 40 clips.
