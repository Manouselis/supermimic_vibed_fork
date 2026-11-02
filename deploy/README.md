# G1 / Thor deployment preparation

**Ready for offline tests and receive-only hardware checks. Motor execution is not implemented.**

The runner uses the released checkpoint 40,000 and an explicitly selected reference clip.
It does not replay raw joint angles or need the learned world model at inference time.
All 40 references are packaged. None is cleared for physical execution.

## Start here

- [TASKS_TO_RUN.md](TASKS_TO_RUN.md): exact next-session steps.
- [AUDIT.md](AUDIT.md): observation contract, gaps and source references.
- [config.yaml](config.yaml): paths, runtime selection and physical-review flags.

The current local bundle is `deploy/artifacts/g1-lafan1-040000/`, containing verified weights,
40 motion CSVs, the model URDF and actuator metadata. Bundles and diagnostics are Git-ignored.
Code travels through Git. Transfer the bundle separately because weights/motions are not Git source.

## Commands

Run from the repository root. Global options go **before** the command:

```bash
python3 deploy/inspect_host.py                        # works before dependencies are installed
python -m deploy.run check                           # file hashes and inference contract
python -m deploy.run list-motions
python -m deploy.run benchmark --samples 300          # laptop/Thor inference timing, not tracking
python -m deploy.run --device cuda:0 benchmark        # only with a working compatible CUDA build
python -m deploy.run capture --interface eth0 --seconds 10
python -m deploy.run replay --input deploy/runs/TAKE/samples.jsonl --clip walk1_subject1
python -m deploy.run shadow --interface eth0 --clip walk1_subject1 --seconds 5
```

Replace `eth0` with the inspected robot-network interface. No automatic interface/robot discovery.
`capture` only records feedback. `replay` and `shadow` require the reviewed configuration flags.
They reject mismatched starting posture, invalid inputs, unsupported IMU/AB frames, stale feedback,
unsafe hypothetical targets and live deadline overruns. They do not correct or clip predictions.
Each command saves `report.json`, plus `samples.jsonl` for live capture/shadow, in a new run directory.
`COMPLETE` means that diagnostic finished. It does not mean the robot can safely execute a motion.

Shadow uses its previous **predicted, unapplied** action in the history. The real robot remains under
its existing controller. This checks data plumbing and timing, not closed-loop motion quality.
There is no `--execute`, command publisher, controller handover or automatic robot-mode change.

## Dependencies

Inference needs compatible PyTorch, NumPy, safetensors and PyYAML. Install the small dependencies
with `python -m pip install -r deploy/requirements.txt` inside an isolated environment.
The Torch build must match the actual Thor software stack. This requirements file deliberately
does not replace PyTorch/CUDA. SDK2/CycloneDDS are only needed for `capture` and `shadow`.
No ROS, Isaac Lab, MuJoCo, camera, TensorRT or ONNX is required on Thor for this version.

## Rebuild a portable bundle

Use the exact release and matching motion/model assets, not arbitrary replacement files:

```bash
python -m deploy.run prepare \
  --policy /absolute/published-policy-directory \
  --motion-dir /absolute/lafan1_g1 \
  --urdf /absolute/g1_29dof_rev_1_0.urdf \
  --destination deploy/artifacts/g1-lafan1-040000
```

Existing destinations are rejected. The original policy manifest and all 40 CSV hashes are checked.
The bundle includes no robot meshes or simulator assets because hardware inference does not use them.
Keep the source motion terms with any transferred bundle. See [release licensing](../docs/RELEASE_LICENSES.md).

## Verification

```bash
python -m unittest -v tools.test_deployment tools.test_policy_release
python supermimic/evaluate.py --run_dir deploy/runs/sim-check \
  --ckpt deploy/artifacts/g1-lafan1-040000/policy --deployment-check \
  --result-json deploy/runs/sim-check/evaluation.json
```

The second command runs on the existing **laptop simulator**, with the usual simulator assets installed.
It drives all 40 simulated robots through the deployment adapter and compares every step against
the original pipeline. Do not install the x86 simulator Docker image on Thor.
Tests using real artifacts are explicitly skipped if the bundle is absent. Set `ORIGINAL_CHECKPOINT`
to the original `.pt` file to also compare every released tensor with its source.
