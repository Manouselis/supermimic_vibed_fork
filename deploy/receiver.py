"""Receive-only Unitree LowState capture. No command message, publisher or mode-switch client."""

import threading
import time


class Receiver:
    def __init__(self, interface, domain=0, clock=time.monotonic):
        if not interface or not isinstance(interface, str):
            raise ValueError('Set an explicit robot network interface')
        # These imports are deliberately local: offline tooling needs neither DDS nor the SDK.
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        from unitree_sdk2py.utils.crc import CRC

        self.clock, self.lock = clock, threading.Lock()
        self.latest, self.last_tick, self.error = None, None, None
        self.crc = CRC()
        ChannelFactoryInitialize(domain, interface)
        self.subscriber = ChannelSubscriber('rt/lowstate', LowState_)
        # SDK callback without a Python FIFO: retain only the newest feedback, never a backlog.
        self.subscriber.Init(self._callback, 0)

    def _callback(self, msg):
        try:
            if self.crc.Crc(msg) != msg.crc:
                raise ValueError('LowState CRC mismatch')
            tick = int(msg.tick)
            with self.lock:
                if self.last_tick is not None:
                    advance = (tick - self.last_tick) % (2**32)
                    if advance == 0:  # duplicate data must not reset the freshness timer
                        return
                    if advance >= 2**31:
                        raise ValueError('Robot tick moved backwards or robot rebooted')
                sample = {'received_monotonic_s': self.clock(), 'tick': tick,
                          'mode_pr': int(msg.mode_pr), 'mode_machine': int(msg.mode_machine),
                          'version': list(msg.version), 'q': [float(m.q) for m in msg.motor_state],
                          'qd': [float(m.dq) for m in msg.motor_state],
                          'quaternion_wxyz': list(msg.imu_state.quaternion),
                          'gyro_body': list(msg.imu_state.gyroscope)}
                self.latest, self.last_tick = sample, tick
        except Exception as exc:
            with self.lock:
                self.error = str(exc)

    def snapshot(self, stale_s):
        with self.lock:
            if self.error:
                raise ValueError(self.error)
            if self.latest is None:
                return None
            sample = dict(self.latest)
        age = self.clock() - sample['received_monotonic_s']
        if age < 0 or age > stale_s:
            raise ValueError(f'Stale robot feedback: {age:.3f}s')
        return sample

    def close(self):
        self.subscriber.Close()
