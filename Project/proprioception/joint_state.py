"""31 维关节状态读取 + 下发(编码器 + 力矩)。"""
from __future__ import annotations
import time
from common.types import JointState, JointCommand


class JointStateReader:
    """关节状态读取器。

    模式: real(调 Oli SDK) / mock(虚拟数据)。
    """

    def __init__(
        self,
        num_dof: int = 31,
        mode: str = "mock",
        config: dict | None = None,
    ):
        self.num_dof = num_dof
        self.mode = mode
        self.config = config or {}
        # 关节名(由 Oli URDF 决定,这里从 config 读)
        self.joint_names = self.config.get("robot", {}).get(
            "joint_names", [f"j{i}" for i in range(num_dof)]
        )
        # 力矩限位(Readme §6.2:腿 150 N·m)
        self.max_torque_nm = float(self.config.get("robot", {}).get("limits", {}).get("max_torque_nm", 150.0))
        if mode == "real":
            # TODO: 打开 Oli 关节驱动
            self._dev = None

    def read(self) -> JointState:
        """读一帧关节状态。"""
        if self.mode == "mock":
            # mock:全零
            zeros = [0.0] * self.num_dof
            return JointState(
                name=list(self.joint_names),
                position=zeros[:],
                velocity=zeros[:],
                effort=zeros[:],
                timestamp_ns=time.time_ns(),
            )
        # 真实模式 TODO
        # if self._dev is not None:
        #     raw = self._dev.read_joints()
        #     return JointState(name=self.joint_names, position=raw['q'], ...)
        raise NotImplementedError("real mode not yet wired to Oli SDK")

    def send_command(self, cmd: JointCommand) -> bool:
        """下发关节指令(部署/标定用)。

        Returns: True 成功,False 失败。
        """
        # L1 关节力矩限位:任何模式都强制(安全网)
        if len(cmd.effort) != self.num_dof:
            return False
        for t in cmd.effort:
            if abs(t) > self.max_torque_nm:
                return False

        if self.mode == "mock":
            return True

        # TODO: 真实模式 — self._dev.send_joint_cmd(cmd)
        return True

    def close(self) -> None:
        if self.mode == "real" and self._dev is not None:
            pass


__all__ = ["JointStateReader"]
