"""MPC 规划器抽象 + 通用零阶 fallback。

完整动力学模型(单刚体 + 摆动腿动力学)需要 Oli URDF 接入后实现。
此处提供零阶 fallback,作为接口骨架。
"""
from __future__ import annotations
from common.types import Skill, RobotState, Trajectory
from common.logging import setup_logger

_log = setup_logger("mpc")


class MPCPlanner:
    """MPC 规划器。

    当前实现:零阶保持(把当前 base_pose 当作 waypoint)。
    实装可注入:casadi / acados / 自实现 QP,配合 Oli URDF。
    """

    def __init__(self, urdf_path: str = "", config: dict | None = None):
        self.urdf_path = urdf_path
        self.config = config or {}
        self.horizon_s = float(self.config.get("horizon_s", 1.0))
        self.dt_s = float(self.config.get("dt_s", 0.01))
        # TODO: 加载 Oli 动力学(SRB + 摆动腿)
        # try:
        #     import pinocchio as pin
        #     self._model = pin.buildModelFromUrdf(urdf_path)
        # except ImportError:
        #     pass

    def plan(self, skill: Skill, state: RobotState, horizon_s: float | None = None, dt_s: float | None = None) -> Trajectory:
        """零阶规划:返回当前 base_pose 作为唯一 waypoint。

        实装版本:求解 QP,预测 horizon 步状态。
        """
        h = horizon_s or self.horizon_s
        d = dt_s or self.dt_s
        n_steps = max(1, int(h / d))

        waypoints = [state.base_pose] * n_steps
        return Trajectory(waypoints=waypoints, horizon_s=h, dt_s=d)

    def reset(self) -> None:
        """warm-start 清零。"""
        pass


__all__ = ["MPCPlanner"]
