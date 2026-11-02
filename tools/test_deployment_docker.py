"""Container wrapper tests. Docker is mocked so tests cannot contact a robot or pull images."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DockerWrapperTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        (self.dir/'bundle').mkdir()
        self.log = self.dir/'docker.jsonl'
        docker = self.dir/'docker'
        docker.write_text('''#!/usr/bin/env python3
import json,os,sys
with open(os.environ['MOCK_DOCKER_LOG'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')
if sys.argv[1:3]==['image','inspect']:
    print(os.environ.get('MOCK_ROLE','receive-only') if 'Labels' in sys.argv[4] else os.environ.get('MOCK_ARCH','amd64'))
''')
        docker.chmod(0o755)
        uname = self.dir/'uname'
        uname.write_text('#!/bin/sh\nprintf "%s\\n" "${MOCK_UNAME:-x86_64}"\n')
        uname.chmod(0o755)
        git = self.dir/'git'
        git.write_text('#!/bin/sh\ncase "$*" in *rev-parse*) echo 0123456789abcdef ;; esac\n')
        git.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.dir)+os.pathsep+os.environ['PATH'],
                        MOCK_DOCKER_LOG=str(self.log), THOR_BUNDLE=str(self.dir/'bundle'),
                        THOR_RUNS=str(self.dir/'runs'))

    def run_script(self, *args, script='run.sh'):
        p = subprocess.run(['bash',str(ROOT/'deploy/docker'/script),*args],env=self.env,capture_output=True,text=True)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []
        return p,calls

    def test_offline_is_network_isolated_with_read_only_inputs(self):
        p,calls = self.run_script('check')
        self.assertEqual(p.returncode,0,p.stderr)
        run = calls[-1]
        self.assertEqual(run[run.index('--network')+1],'none')
        for flag in ['--rm','--read-only','--cap-drop','--security-opt','--user']:
            self.assertIn(flag,run)
        mounts = [run[i+1] for i,x in enumerate(run) if x=='--mount']
        self.assertEqual(sum('readonly' in x for x in mounts),2)
        self.assertTrue(any('dst=/runs' in x and 'readonly' not in x for x in mounts))
        self.assertFalse(any('/var/run/docker.sock' in x for x in run))
        self.assertNotIn('--privileged',run)
        self.assertNotIn('--gpus',run)

    def test_gpu_access_is_explicit(self):
        p,calls = self.run_script('--gpu','--device','cuda:0','benchmark')
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertIn('--gpus',calls[-1])
        self.assertEqual(calls[-1][calls[-1].index('--runtime')+1],'nvidia')
        self.assertEqual(calls[-1][calls[-1].index('--network')+1],'none')

    def test_capture_needs_robot_network_flag(self):
        p,calls = self.run_script('capture','--interface','test0')
        self.assertEqual(p.returncode,2)
        self.assertFalse(calls)

    def test_capture_shares_only_networking(self):
        p,calls = self.run_script('--robot','capture','--interface','test0')
        self.assertEqual(p.returncode,0,p.stderr)
        run=calls[-1]
        self.assertEqual(run[run.index('--network')+1],'host')
        for flag in ['--privileged','--pid','--ipc','--device']:
            self.assertNotIn(flag,run)

    def test_robot_flag_rejected_for_offline_command(self):
        p,calls=self.run_script('--robot','check')
        self.assertEqual(p.returncode,2)
        self.assertFalse(calls)

    def test_wrong_image_role_is_rejected(self):
        self.env['MOCK_ROLE']='other'
        p,calls=self.run_script('check')
        self.assertEqual(p.returncode,2)
        self.assertFalse(any(c[0]=='run' for c in calls))

    def test_wrong_architecture_is_rejected(self):
        self.env['MOCK_ARCH']='arm64'
        p,calls=self.run_script('check')
        self.assertEqual(p.returncode,2)
        self.assertFalse(any(c[0]=='run' for c in calls))

    def test_option_value_is_not_mistaken_for_command(self):
        p,calls=self.run_script('--output','capture','check')
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(calls[-1][calls[-1].index('--network')+1],'none')

    def test_native_arm_build_has_pinned_base_and_platform(self):
        self.env.update(MOCK_UNAME='aarch64',MOCK_ARCH='arm64')
        p,calls=self.run_script(script='build.sh')
        self.assertEqual(p.returncode,0,p.stderr)
        build=calls[-1]
        self.assertEqual(build[build.index('--platform')+1],'linux/arm64')
        self.assertTrue(any('BASE_IMAGE=nvcr.io/nvidia/pytorch:26.03-py3@sha256:' in x for x in build))

    def test_thor_build_refuses_laptop_architecture(self):
        p,calls=self.run_script(script='build.sh')
        self.assertEqual(p.returncode,2)
        self.assertFalse(calls)


if __name__=='__main__':
    unittest.main()
