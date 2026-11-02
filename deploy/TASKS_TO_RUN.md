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

## 3. Build our Docker image on Thor

Nothing is installed into the host Python environment. Do not run host pip/apt installs or create a
host venv. The image contains PyTorch, CycloneDDS, SDK2 and this application.

1. Review the read-only inventory. Check the actual ARM64 architecture, JetPack/L4T and driver against
   the candidate NVIDIA CUDA 13.2 base. If they differ, select a compatible image before building.
2. Check existing Docker access. NVIDIA container support is additionally needed for `--gpu`:

```bash
cd ~/supermimic_thor_deploy
docker info
bash deploy/docker/build.sh
bash deploy/docker/run.sh inspect
bash deploy/docker/run.sh check
bash deploy/docker/run.sh benchmark --samples 300
```

The script builds **our** `supermimic-g1-thor:dev` image on ARM64, with a pinned NVIDIA PyTorch base
and pinned SDK/DDS commits. It does not touch the laptop simulator image or default robot services.
The base is a multi-GB download. Confirm adequate storage and internet access first.
If Docker or NVIDIA container support is missing, stop and review the prerequisite rather than
changing the host setup automatically.

`check` must report all 40 clips. Benchmark reports bundle/session loading and warm inference after
30 warm-up calls. `over_20ms` must be zero before shadow. CPU is the default.

3. If GPU access is desired, test a real CUDA operation inside our image, then benchmark:

```bash
bash deploy/docker/run.sh --gpu inspect
bash deploy/docker/run.sh --gpu --device cuda:0 benchmark --samples 300
```

Do not upgrade the host driver to make a guessed image work. Choose the image to match the reviewed host.
A reviewed alternative can be supplied with `THOR_BASE_IMAGE=IMAGE bash deploy/docker/build.sh`.

## 4. Capture feedback without inference

Identify the Thor interface connected to the robot from the inventory. Replace `ACTUAL_INTERFACE`:

```bash
bash deploy/docker/run.sh --robot capture --interface ACTUAL_INTERFACE --seconds 10
```

Review the new host `deploy/runs/.../report.json` and `samples.jsonl` (the same directory is `/runs/` inside the container):

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
bash deploy/docker/run.sh replay --input /runs/TAKE/samples.jsonl --clip walk1_subject1
bash deploy/docker/run.sh --robot shadow --interface ACTUAL_INTERFACE --clip walk1_subject1 --seconds 5
```

Offline runs have no network access. `--robot` explicitly shares host networking for receive-only DDS.
The container runs without privileged mode, with read-only code/config/weights and only results persisted to the host.
Shadow keeps the robot under its existing controller. Inspect hypothetical targets, timing and any
rejection reason. It cannot prove balance, tracking accuracy or physical motion success.

## 6. Stop before motor execution

Send the reports back in this chat. We must review model differences, initialization and the first
motion, then implement/review the command loop, controller handover, watchdog, safe stop and operator
emergency-stop procedure. Motor execution requires a separate explicit approval.
There is no command in this version that moves the robot, and no automatic sequence of 40 clips.
