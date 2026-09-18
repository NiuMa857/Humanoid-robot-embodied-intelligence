"""奖励函数(Oli 专属,通用实现,所有奖励可调权重)。"""
from __future__ import annotations
import math
from common.types import RobotState, JointCommand


class Reward:
    """奖励基类。子类实现 `__call__`。"""
    weight: float = 1.0
    name: str = "base"


class StairClimbReward(Reward):
    """楼梯攀爬奖励 = 前进速度 + 高度增益 - 能耗 - 跌倒惩罚。"""

    name = "stair_climb"
    weight = 1.0

    def __init__(self, w_forward: float = 1.0, w_height: float = 2.0,
                 w_energy: float = -0.01, w_fall: float = -10.0,
                 w_action_rate: float = -0.005):
        self.w_forward = w_forward
        self.w_height = w_height
        self.w_energy = w_energy
        self.w_fall = w_fall
        self.w_action_rate = w_action_rate

    def __call__(self, state: RobotState, prev_state: RobotState, action: JointCommand) -> float:
        r = 0.0
        # 前进奖励(用 base x 增量)
        if prev_state.base_pose[0] != 0 or state.base_pose[0] != 0:
            r += self.w_forward * (state.base_pose[0] - prev_state.base_pose[0])
        # 高度奖励(只对 height > 0.5 有效,鼓励爬升)
        height = state.base_pose[2]
        r += self.w_height * max(0.0, height - 0.5)
        # 能耗
        if action.effort:
            r += self.w_energy * sum(e * e for e in action.effort)
        # 跌倒惩罚(高度过低)
        if height < 0.3:
            r += self.w_fall
        # action rate(鼓励平滑)
        if action.position and prev_state.joint is not None and prev_state.joint.position:
            n = min(len(action.position), len(prev_state.joint.position))
            if n > 0:
                delta = sum((action.position[i] - prev_state.joint.position[i]) ** 2 for i in range(n))
                r += self.w_action_rate * delta
        return r


class GraspReward(Reward):
    """抓取奖励 = 抓取成功 +1,脱落 -1,精准度 bonus。"""

    name = "grasp"
    weight = 1.0

    def __call__(self, state: RobotState, grasp_force: float = 0.0,
                 target_force: float = 10.0) -> float:
        # 简化:在 grasp_force 接近 target 时给奖励
        err = abs(grasp_force - target_force)
        if err < 1.0:
            return 1.0
        return max(-1.0, 1.0 - err / 10.0)


class TrackingReward(Reward):
    """轨迹跟踪奖励(供 WBC 训练用)。"""

    name = "tracking"
    weight = 1.0

    def __call__(self, state: RobotState, target: RobotState) -> float:
        # base 位置误差
        pos_err = sum((state.base_pose[i] - target.base_pose[i]) ** 2 for i in range(3))
        # 关节位置误差
        joint_err = 0.0
        if state.joint is not None and target.joint is not None:
            n = min(len(state.joint.position), len(target.joint.position))
            if n > 0:
                joint_err = sum((state.joint.position[i] - target.joint.position[i]) ** 2 for i in range(n)) / n
        return -(pos_err + 0.1 * joint_err)


class StabilityReward(Reward):
    """姿态稳定奖励(高度 + 倾角)。"""

    name = "stability"
    weight = 0.5

    def __call__(self, state: RobotState) -> float:
        # base 高度
        h = state.base_pose[2]
        height_bonus = max(0.0, h - 0.5)
        # 倾角(简化:从 base_pose 第 4-7 维取 quat)
        qx, qy, qz, qw = state.base_pose[3], state.base_pose[4], state.base_pose[5], state.base_pose[6]
        # roll/pitch 简化(只用 qx, qy)
        roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        pitch = math.asin(max(-1, min(1, 2 * (qw * qy - qz * qx))))
        tilt = abs(roll) + abs(pitch)
        tilt_penalty = max(0.0, tilt - 0.3) * 2
        return height_bonus - tilt_penalty


__all__ = [
    "Reward",
    "StairClimbReward", "GraspReward", "TrackingReward", "StabilityReward",
]
