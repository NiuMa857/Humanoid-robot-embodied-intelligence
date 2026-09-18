"""终止条件(Readme §7.2)。"""
from __future__ import annotations
import math
from common.types import RobotState


class Termination:
    """终止条件基类。返回 True = 该 episode 应终止。"""
    name: str = "base"


class FallTermination(Termination):
    """摔倒 = base 高度 < 阈值 或 倾角 > 60°。"""

    name = "fall"

    def __init__(self, min_height: float = 0.3, max_tilt_deg: float = 60.0):
        self.min_height = min_height
        self.max_tilt_deg = max_tilt_deg

    def __call__(self, state: RobotState) -> bool:
        if state.base_pose[2] < self.min_height:
            return True
        # 倾角
        qx, qy, qz, qw = state.base_pose[3:7]
        pitch = math.asin(max(-1, min(1, 2 * (qw * qy - qz * qx))))
        if abs(math.degrees(pitch)) >= self.max_tilt_deg:
            return True
        return False


class CollisionTermination(Termination):
    """碰撞 = 力矩突变(简化版:任何关节力矩 > 100N·m 视为异常)。"""

    name = "collision"

    def __init__(self, max_torque_nm: float = 200.0):
        self.max_torque_nm = max_torque_nm

    def __call__(self, state: RobotState) -> bool:
        if state.joint is None or not state.joint.effort:
            return False
        return any(abs(t) > self.max_torque_nm for t in state.joint.effort)


class TimeoutTermination(Termination):
    """单 episode 超时。"""

    name = "timeout"

    def __init__(self, max_steps: int = 1000):
        self.max_steps = max_steps

    def __call__(self, step_count: int) -> bool:
        return step_count >= self.max_steps


def check_all(terminations: list, state: RobotState, step_count: int) -> tuple[bool, str]:
    """跑所有终止条件,返回 (done, reason)。

    Args:
        terminations: 终止条件列表
        state: 当前状态
        step_count: 当前步数
    """
    for t in terminations:
        if isinstance(t, TimeoutTermination):
            if t(step_count):
                return True, t.name
        else:
            if t(state):
                return True, t.name
    return False, ""


__all__ = [
    "Termination",
    "FallTermination", "CollisionTermination", "TimeoutTermination",
    "check_all",
]
