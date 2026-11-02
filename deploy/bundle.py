"""Create and verify a portable inference bundle. No Isaac Lab or hardware SDK imports."""

import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from supermimic.g1_spec import G1_JOINT_NAMES, default_joint_pos_list, action_scale, motor_of, _kp, _kd


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify_files(root, hashes):
    root = Path(root).resolve()
    for name, expected in hashes.items():
        p = (root / name).resolve()
        if root not in p.parents or not p.is_file() or digest(p) != expected:
            raise ValueError(f'File missing, outside bundle, or SHA-256 mismatch: {name}')


def verify_policy(root):
    manifest = read_json(Path(root) / 'manifest.json')
    if set(manifest) != {'config.json', 'policy.safetensors'}:
        raise ValueError('Expected the published policy/config hash manifest')
    verify_files(root, manifest)
    config = read_json(Path(root) / 'config.json')
    p, control = config['policy'], config['control']
    if (config['format'] != 'supermimic-policy-v1' or p['obs_dim'] != 99 or p['hist'] != 10
            or p['goal_frames'] != 10 or p['goal_dim'] != 730 or p['z_dim'] != 32
            or control['dt_seconds'] != 0.02 or control['joint_names'] != G1_JOINT_NAMES):
        raise ValueError('Unsupported policy contract. Do not guess observation or actuator layouts.')
    if p['amp'] not in ['none', 'bf16'] or config['dataset']['source_fps'] != 30 or config['dataset']['control_fps'] != 50:
        raise ValueError('Unsupported precision or reference sampling rate')
    expected_obs = ['joint_position_minus_default_minus_offset:29', 'joint_velocity:29',
                    'root_angular_velocity_body:3', 'projected_gravity:3', 'previous_action:29',
                    'reference_relative_root_orientation_6d:6']
    if control['observation_order'] != expected_obs:
        raise ValueError('Unexpected observation order')
    return config


def robot_spec(urdf):
    joints = {j.attrib['name']: j for j in ET.parse(urdf).getroot().findall('joint')}
    lo, hi = [], []
    for name in G1_JOINT_NAMES:
        limit = joints[name].find('limit')
        lo.append(float(limit.attrib['lower']))
        hi.append(float(limit.attrib['upper']))
    return {'joint_names': G1_JOINT_NAMES, 'default_q': default_joint_pos_list(),
            'action_scale': [action_scale(n) for n in G1_JOINT_NAMES],
            'kp': [_kp(motor_of(n)[2]) for n in G1_JOINT_NAMES],
            'kd': [_kd(motor_of(n)[2]) for n in G1_JOINT_NAMES],
            'effort_limit': [motor_of(n)[0] for n in G1_JOINT_NAMES],
            'velocity_limit': [motor_of(n)[1] for n in G1_JOINT_NAMES],
            'q_lo': lo, 'q_hi': hi, 'control_dt': 0.02,
            'note': 'Training-model parameters, not verified physical-robot calibration or control gains.'}


def prepare(policy, motion_dir, urdf, output):
    policy, motion_dir, urdf, output = map(Path, [policy, motion_dir, urdf, output])
    config = verify_policy(policy)
    expected_urdf = config['assets_sha256']['g1_description/g1_29dof_rev_1_0.urdf']
    if digest(urdf) != expected_urdf:
        raise ValueError('URDF does not match the released training-model hash')
    motions = config['dataset']['motion_sha256']
    if len(motions) != 40:
        raise ValueError('Expected the selected release with all 40 reference clips')
    verify_files(motion_dir, motions)
    spec = robot_spec(urdf)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'policy').mkdir()
    (output / 'motions').mkdir()
    for name in ['config.json', 'manifest.json', 'policy.safetensors']:
        shutil.copyfile(policy / name, output / 'policy' / name)
    for name in motions:
        shutil.copyfile(motion_dir / name, output / 'motions' / name)
    shutil.copyfile(urdf, output / 'model.urdf')
    shutil.copyfile(Path(__file__).resolve().parents[1] / 'docs/RELEASE_LICENSES.md', output / 'NOTICE.md')
    write_json(output / 'robot.json', spec)
    hashes = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output / 'bundle.json', {'format': 'supermimic-g1-receive-only-v1',
               'checkpoint': config['checkpoint']['iteration'], 'files': hashes,
               'clips': sorted(Path(n).stem for n in motions), 'motor_publishing': False})
    verify_bundle(output)
    return {'bundle': str(output.resolve()), 'clips': len(motions), 'checkpoint': config['checkpoint']['iteration']}


def verify_bundle(root):
    root = Path(root)
    manifest = read_json(root / 'bundle.json')
    if manifest['format'] != 'supermimic-g1-receive-only-v1' or manifest['motor_publishing'] is not False:
        raise ValueError('Unsupported bundle')
    verify_files(root, manifest['files'])
    config = verify_policy(root / 'policy')
    # Re-derive all actuator values instead of trusting an editable sidecar.
    spec = read_json(root / 'robot.json')
    if spec != robot_spec(root / 'model.urdf'):
        raise ValueError('robot.json differs from the training contract')
    if digest(root / 'model.urdf') != config['assets_sha256']['g1_description/g1_29dof_rev_1_0.urdf']:
        raise ValueError('Wrong model asset')
    verify_files(root / 'motions', config['dataset']['motion_sha256'])
    clips = sorted(Path(n).stem for n in config['dataset']['motion_sha256'])
    if manifest['clips'] != clips or len(clips) != 40 or manifest['checkpoint'] != config['checkpoint']['iteration']:
        raise ValueError('Bundle clip/checkpoint metadata mismatch')
    return config, spec, manifest
