"""python -m deploy.run <command>. All commands are offline or receive-only."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deploy.bundle import prepare, verify_bundle, write_json


def load_config(path):
    import math
    import yaml
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    if cfg['device'] not in ['cpu', 'cuda:0']:
        raise ValueError('device must be cpu or cuda:0')
    cfg['bundle'] = str((path.parent / cfg['bundle']).resolve())
    robot, checks = cfg['robot'], cfg['checks']
    indices = robot['motor_indices']
    if len(indices) != 29 or len(set(indices)) != 29 or any(type(i) is not int or not 0 <= i < 35 for i in indices):
        raise ValueError('Expected 29 unique integer SDK indices in 0..34')
    offsets = robot['joint_offsets_rad']
    if len(offsets) != 29 or not all(type(x) in (int, float) and math.isfinite(x) for x in offsets):
        raise ValueError('Expected 29 finite joint offsets')
    for key in ['model_reviewed', 'joint_order_verified', 'imu_frame_verified', 'joint_offsets_verified']:
        if type(robot[key]) is not bool:
            raise ValueError(f'{key} must be true or false')
    if type(robot['dds_domain']) is not int or not 0 <= robot['dds_domain'] <= 232:
        raise ValueError('DDS domain must be an integer in 0..232')
    if type(cfg['threads']) is not int or not 1 <= cfg['threads'] <= 64:
        raise ValueError('threads must be in 1..64')
    for name, value in checks.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'Invalid positive diagnostic limit: {name}')
    return cfg


def require_review(cfg):
    robot = cfg['robot']
    if not all(robot[k] for k in ['model_reviewed', 'joint_order_verified', 'imu_frame_verified', 'joint_offsets_verified']):
        raise ValueError('Hardware review flags are incomplete. Capture/inspect are available without them.')
    if robot['imu_frame'] != 'pelvis':
        raise ValueError('Only a verified pelvis IMU is supported. Do not substitute torso IMU data.')


def get_session(cfg, clip):
    import torch
    from deploy.runtime import Session
    torch.set_num_threads(cfg['threads'])
    return Session(cfg['bundle'], clip, cfg['device'], cfg['robot']['joint_offsets_rad'])


def timing(values):
    import numpy as np
    return {'samples': len(values), 'mean_ms': float(np.mean(values) * 1000),
            'p50_ms': float(np.percentile(values, 50) * 1000),
            'p95_ms': float(np.percentile(values, 95) * 1000), 'max_ms': float(max(values) * 1000),
            'over_20ms': sum(v > .02 for v in values)}


def reference_sample(session, frame):
    from supermimic.rotations import mat_to_quat
    m = session.motion
    q = (m.q[frame] + session.runner.offsets).cpu().tolist()
    return {'q': q, 'qd': m.qd[frame].cpu().tolist(),
            'quaternion_wxyz': mat_to_quat(m.root_R[frame]).cpu().tolist(),
            'gyro_body': (m.root_R[frame].T @ m.root_w[frame]).cpu().tolist()}


def benchmark(session, samples=300):
    # Perfect reference-state inputs isolate inference cost. This is NOT a closed-loop robot test.
    import torch
    values = []
    for i in range(samples + 30):
        f = i % (int(session.motion.clip_end[0]))
        q, qd, R, gyro = session.tensors(reference_sample(session, f))
        idx = torch.tensor([f], device=session.device)
        goal = session.motion.goal(idx, 10)
        if session.device.type == 'cuda':
            torch.cuda.synchronize()
        t = time.perf_counter()
        session.runner.step(q, qd, R, gyro, session.motion.root_R[idx], goal)
        if session.device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t
        if i >= 30:
            values.append(elapsed)
    session.runner.reset()
    return timing(values)


def motion_summaries(bundle, manifest):
    import math
    import torch
    from supermimic.motion import MotionLib
    rows = []
    for clip in manifest['clips']:
        m = MotionLib([str(Path(bundle) / 'motions' / (clip + '.csv'))], None, 'cpu')
        rows.append({'clip': clip, 'duration_s': round(float(m.clip_end[0]) * .02, 2),
                     'initial_joint_speed_max_rad_s': round(float(m.qd[0].abs().max()), 4),
                     'initial_root_speed_m_s': round(float(torch.linalg.vector_norm(m.root_v[0])), 4),
                     'initial_root_height_m': round(float(m.root_p[0,2]), 4),
                     'initial_tilt_rad': round(math.acos(max(-1., min(1., float(m.root_R[0,2,2])))), 4),
                     'initial_q_rad': m.q[0].tolist(), 'physical_execution_approved': False})
    return rows


def first_sample(receiver, checks):
    deadline = time.monotonic() + checks['first_packet_timeout_s']
    while time.monotonic() < deadline:
        sample = receiver.snapshot(checks['stale_s'])
        if sample:
            return sample
        time.sleep(.005)
    raise ValueError('No valid LowState received. Check interface, DDS and SDK without changing robot controllers.')


def mapped_sample(sample, cfg):
    if sample['mode_pr'] != 0:
        raise ValueError('LowState uses A/B coordinates. Only PR (mode_pr=0) is implemented.')
    sample = dict(sample)
    sample['motor_indices'] = cfg['robot']['motor_indices']
    return sample


def run_live(args, cfg, out, report):
    from deploy.receiver import Receiver
    session = None
    if args.command == 'shadow':
        require_review(cfg)
        session = get_session(cfg, args.clip)
        report['warm_inference'] = benchmark(session, 30)
        if report['warm_inference']['over_20ms']:
            raise ValueError('Warm inference exceeds the 20ms policy budget. Inspect/benchmark before live shadow.')
    receiver = Receiver(cfg['robot']['interface'], cfg['robot']['dds_domain'])
    report.update(previous_action_source='predicted_not_applied' if session else None,
                  control_of_robot='NONE', motor_publishing=False)
    count, last_tick, times, ages, gaps = 0, None, [], [], []
    try:
        sample = first_sample(receiver, cfg['checks'])
        initial_machine_mode = sample['mode_machine']
        if session:
            report.update(session.begin(mapped_sample(sample, cfg), cfg['checks']))
        start = due = time.monotonic()
        last_received = None
        with (out / 'samples.jsonl').open('x') as stream:
            stream.write(json.dumps({'format': 'g1-lowstate-capture-v1', 'robot_config': cfg['robot'],
                                     'clip': args.clip if session else None, 'motor_publishing': False}) + '\n')
            while time.monotonic() - start < args.seconds:
                wait = due - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                sample = receiver.snapshot(cfg['checks']['stale_s'])
                if sample is None:
                    raise ValueError('Feedback vanished')
                if session and sample['mode_machine'] != initial_machine_mode:
                    raise ValueError('Robot machine mode changed during shadow inference')
                if sample['tick'] == last_tick:
                    # A progressing tick, not repeated callbacks, defines freshness.
                    if session:
                        raise ValueError('No new robot tick for this 20ms shadow period')
                    due += .02
                    continue
                last_tick = sample['tick']
                age = time.monotonic() - sample['received_monotonic_s']
                ages.append(age)
                if last_received is not None:
                    gap = sample['received_monotonic_s'] - last_received
                    gaps.append(gap)
                    if session and gap > .04:
                        raise ValueError(f'Feedback sampling gap {gap:.3f}s exceeds two policy periods')
                last_received = sample['received_monotonic_s']
                if session:
                    try:
                        result = session.step(mapped_sample(sample, cfg), cfg['checks'])
                    except StopIteration:
                        break
                    sample['prediction'] = result
                    times.append(result['inference_s'])
                stream.write(json.dumps(sample, allow_nan=False) + '\n')
                count += 1
                report['samples'] = count
                if session and result['checks']:
                    raise ValueError(', '.join(result['checks']))
                due += .02
                if session and time.monotonic() > due:
                    raise ValueError('Shadow processing missed its 20ms deadline. No commands were sent.')
                if not session and time.monotonic() > due:
                    due = time.monotonic()  # capture diagnostics never build a processing backlog
    finally:
        receiver.close()
        report['samples'] = count
        if ages:
            report['max_receive_age_s'] = max(ages)
        if gaps:
            report['max_sample_gap_s'] = max(gaps)
        if times:
            report['inference'] = timing(times)
    if not count:
        raise ValueError('No samples captured')


def replay(args, cfg, report):
    require_review(cfg)
    session = get_session(cfg, args.clip)
    values, previous_t, previous_tick, count = [], None, None, 0
    with args.input.open() as stream:
        header = json.loads(next(stream))
        if header.get('format') != 'g1-lowstate-capture-v1':
            raise ValueError('Unsupported recording')
        if header['robot_config']['motor_indices'] != cfg['robot']['motor_indices']:
            raise ValueError('Recording and config use different motor indices')
        for line in stream:
            sample = mapped_sample(json.loads(line), cfg)
            t = sample['received_monotonic_s']
            if previous_t is not None:
                delta = (sample['tick'] - previous_tick) % 2**32
                if not 0 < t - previous_t <= .04 or not 0 < delta < 2**31:
                    raise ValueError('Recorded timestamps/ticks stalled, reversed or have a >40ms gap')
            if count == 0:
                report.update(session.begin(sample, cfg['checks']))
            try:
                result = session.step(sample, cfg['checks'])
            except StopIteration:
                break
            if result['checks']:
                raise ValueError(', '.join(result['checks']))
            values.append(result['inference_s'])
            previous_t, previous_tick = t, sample['tick']
            count += 1
    if not count:
        raise ValueError('Recording contains no feedback')
    report.update(samples=count, inference=timing(values),
                  previous_action_source='predicted_not_applied', physical_tracking_accuracy='NOT_TESTED')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'deploy/config.yaml')
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda:0'])
    parser.add_argument('--output', type=Path, help='New results directory. Existing paths are never overwritten.')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('inspect')
    p = sub.add_parser('prepare')
    for name in ['policy', 'motion-dir', 'urdf', 'destination']:
        p.add_argument('--' + name, type=Path, required=True)
    sub.add_parser('check')
    sub.add_parser('list-motions')
    p = sub.add_parser('benchmark')
    p.add_argument('--clip', default='walk1_subject1')
    p.add_argument('--samples', type=int, default=300)
    p = sub.add_parser('replay')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--clip', required=True)
    for command in ['capture', 'shadow']:
        p = sub.add_parser(command)
        p.add_argument('--interface', required=True)
        p.add_argument('--seconds', type=float, default=10 if command == 'capture' else 5)
        if command == 'shadow':
            p.add_argument('--clip', required=True)
    args = parser.parse_args(argv)
    out = args.output or ROOT / 'deploy/runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    report = {'command': args.command, 'status': 'FAILED', 'motor_publishing': False,
              'physical_robot_execution': 'NOT_IMPLEMENTED', 'output': str(out.resolve())}
    code = 1
    try:
        if args.command == 'inspect':
            from deploy.inspect_host import inspect
            report.update(host=inspect())
        elif args.command == 'prepare':
            report.update(prepare(args.policy, args.motion_dir, args.urdf, args.destination))
        else:
            cfg = load_config(args.config)
            for key in ['bundle', 'device']:
                if getattr(args, key) is not None:
                    cfg[key] = str(getattr(args, key))
            if hasattr(args, 'interface'):
                cfg['robot']['interface'] = args.interface
            report['config'] = cfg
            if args.command in ['check', 'list-motions']:
                policy, spec, manifest = verify_bundle(cfg['bundle'])
                report.update(checkpoint=policy['checkpoint']['iteration'], clips=manifest['clips'],
                              robot_review_complete=all(cfg['robot'][k] for k in ['model_reviewed', 'joint_order_verified', 'imu_frame_verified', 'joint_offsets_verified']))
                if args.command == 'list-motions':
                    report['motion_summaries'] = motion_summaries(cfg['bundle'], manifest)
            elif args.command == 'benchmark':
                if not 1 <= args.samples <= 10000:
                    raise ValueError('samples must be in 1..10000')
                load_start = time.perf_counter()
                session = get_session(cfg, args.clip)
                report['bundle_and_session_load_s'] = time.perf_counter() - load_start
                report.update(inference=benchmark(session, args.samples), device=str(session.device),
                              precision='bf16_MLP_fp32_state' if session.runner.amp else 'fp32',
                              inputs='perfect reference-state feedback, not hardware')
            elif args.command == 'replay':
                replay(args, cfg, report)
            else:
                import math
                if not math.isfinite(args.seconds) or not 0 < args.seconds <= 120:
                    raise ValueError('seconds must be in (0, 120]')
                run_live(args, cfg, out, report)
        report['status'], code = 'COMPLETE', 0
    except KeyboardInterrupt:
        report.update(status='INTERRUPTED', error='Stopped by operator. No motor commands exist in this runner.')
        code = 130
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        write_json(out / 'report.json', report)
        print(json.dumps(report, indent=2, allow_nan=False))
    return code


if __name__ == '__main__':
    sys.exit(main())
