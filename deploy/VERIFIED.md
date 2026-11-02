# Local verification, 2 November 2026

**No Thor or physical G1 was connected.** All tests used the laptop's existing x86 simulator image,
isolated from the network. Detailed results: [verification.json](verification.json).

- **23 automated tests passed, none skipped** with the real bundle and original checkpoint available.
  Includes all 40 motion feature/padding checks, policy history and action parity, mocked native SDK
  capture/replay/shadow, missing/invalid data, stale/duplicate ticks, CRC and review gates.
- Source `.pt` and released Safetensors: every tensor/normalization buffer identical. Action differences
  were zero on CPU and CUDA for zero, normalization-scaled and real-reference inputs.
- **40 complete simulator rollouts, 0 failures**, mean normalized return **0.86870**.
  Deployment observations, actions and targets matched the original pipeline exactly at every step.
  The adapter's actions drove the simulator. Self-collisions remain disabled.
- The earlier release recorded 0.8696. This fresh run is another nominal simulation trial, not a claim
  of identical repeatability, a held-out benchmark or a real-robot result.
- CPU FP32 warm inference on this laptop, one thread and 300 calls: median **1.19 ms**,
  P95 **1.28 ms**, maximum **1.43 ms**. No call exceeded 20 ms. Bundle/session loading took about 2.18 s.
  These are indicative inference-only measurements. They do not establish Thor or end-to-end timing.
- The portable archive includes hashes, attribution, checkpoint 40,000 and all 40 reference clips.
  It is about **66.7 MiB**, under `deploy/artifacts/g1-thor-pilot.tar.gz`, with a SHA-256 sidecar.

Hardware packet reception, IMU/joint conventions, model/payload agreement and Thor timing remain
unverified. Stationary pose checks do not establish the reference's initial velocities. Motor
execution, startup transition, controller handover and safe stopping are not implemented.

## Docker update

- The separate Dockerfile built successfully on the laptop with the existing x86 PyTorch base.
  The pinned production NVIDIA ARM64 base manifest exists. Its Thor build/GPU compatibility is untested.
- **33 tests passed**, including 10 wrapper/isolation tests and all 23 policy/deployment tests.
- Real bundle checking and CPU/CUDA inference passed through the read-only, non-root, network-isolated wrapper.
- Native SDK CRC and receive-only DDS startup passed on isolated loopback with native CycloneDDS 0.10.5
  and Python bindings 0.10.2. Native 0.10.2 crashed with the SDK XML configuration on this test system.
- Unitree is installed editable **inside the image** so its packaged native CRC libraries remain available.
- Test-only temporary storage permits mock executable scripts. Production runs retain noexec temporary storage.
- No host packages, drivers, Docker daemon configuration or robot services were modified.
