"""底层关节伺服(2 kHz)。

不带真硬件时给 mock 实现,接 Oli SDK 时在 step 里调用真接口。
"""
from __future__ import annotations
import time
from common.types import JointCommand
from common.logging import setup_logger

_log = setup_logger("low_level_servo")


class LowLevelServo:
    """关节底层伺服。

    模式:real(调 Oli SDK)/ mock(只打印)。
    """

    def __init__(self, sdk_client=None, mode: str = "mock", config: dict | None = None):
        self.sdk = sdk_client
        self.mode = mode
        self.config = config or {}
        self._estop_active = False
        # TODO: 启动 2kHz 实时线程(linux RT,Windows 较难)

    def step(self, cmd: JointCommand) -> None:
        """单步下发。"""
        if self._estop_active:
            return
        if self.mode == "mock":
            return
        # TODO: 真实模式 — self.sdk.send_joint_cmd(cmd)
        # 注意:必须在 < 0.5ms 内完成,否则 2kHz 控制会卡顿

    def emergency_stop(self) -> None:
        """L4:急停。"""
        self._estop_active = True
        if self.mode == "real" and self.sdk is not None:
            # TODO: self.sdk.estop()
            pass
        _log.critical("L4: servo emergency stop activated")

    def release_estop(self) -> None:
        self._estop_active = False

    def close(self) -> None:
        pass


__all__ = ["LowLevelServo"]
