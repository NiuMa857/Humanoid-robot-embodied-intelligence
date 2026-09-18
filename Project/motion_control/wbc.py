"""全身控制 WBC(1 kHz)。

完整实现需要:
  - Oli URDF(pinocchio / yourdfpy)
  - 任务链定义
  - 零空间优化

此处给零阶 fallback。
"""
from __future__ import annotations
from common.types import Trajectory, RobotState, JointCommand
from common.logging import setup_logger

_log = setup_logger("wbc")


class WBCController:
    """WBC 抽象(零阶 fallback)。"""

    def __init__(self, urdf_path: str = "", config: dict | None = None):
        self.urdf_path = urdf_path
        self.config = config or {}
        self.num_dof = int(self.config.get("robot", {}).get("dof", 31))
        self.kp = float(self.config.get("kp", 80.0))
        self.kd = float(self.config.get("kd", 2.0))
        # 默认关节位置
        self.default_joint_pos = self.config.get(
            "default_joint_pos", [0.0] * self.num_dof
        )

    def control(self, traj: Trajectory, state: RobotState) -> JointCommand:
        """零阶 WBC:保持默认关节位置 + 阻抗控制。

        实装:任务空间加速度 → 关节力矩(逆动力学)→ 输出位置 + kp/kd。
        """
        target_pos = self.default_joint_pos[:]
        if state.joint is not None and state.joint.position:
            n = min(len(state.joint.position), self.num_dof)
            # 简化:用当前关节位置作 reference,加 kp/kd
            target_pos = list(state.joint.position[:n])
            if n < self.num_dof:
                target_pos += self.default_joint_pos[n:]

        kp = [self.kp] * self.num_dof
        kd = [self.kd] * self.num_dof
        zero = [0.0] * self.num_dof
        return JointCommand(
            position=target_pos,
            velocity=zero,
            effort=zero,
            kp=kp,
            kd=kd,
        )

    def reset(self) -> None:
        pass


__all__ = ["WBCController"]
