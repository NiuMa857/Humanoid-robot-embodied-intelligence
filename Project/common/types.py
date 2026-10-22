"""通用数据类型定义(对齐 Readme 各节接口)。

具体字段先留 TODO,等 Oli 官方 URDF/SDK 拿到后填实际 joint 名、限位等。
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Any, Optional


def to_dict(obj: Any) -> Any:
    """dataclass(含嵌套)→ 可 JSON 序列化的 dict;dict / None 原样返回。"""
    if obj is None:
        return {}
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    return obj


# ============================================================
# 机器人状态(参考 Readme §6.2)
# ============================================================

@dataclass
class JointState:
    """31 维关节状态(对齐 Oli 主动自由度:腿 6x2 + 臂 7x2 + 腰 3 + 颈 2)。"""
    name: list[str] = field(default_factory=list)        # TODO: Oli joint 名
    position: list[float] = field(default_factory=list)  # rad
    velocity: list[float] = field(default_factory=list)  # rad/s
    effort: list[float] = field(default_factory=list)    # N*m(读自力矩传感器)
    timestamp_ns: int = 0


@dataclass
class IMUState:
    """6 轴 IMU(自研,Readme §3.1)。"""
    orientation_xyzw: tuple = (0, 0, 0, 1)
    angular_velocity: tuple = (0, 0, 0)         # rad/s
    linear_acceleration: tuple = (0, 0, 0)      # m/s^2


@dataclass
class RobotState:
    """全身状态(Readme §3.2 本体感知输出,>33 维向量在外面拼接)。"""
    joint: Optional[JointState] = None
    imu: Optional[IMUState] = None
    base_pose: tuple = (0, 0, 0, 0, 0, 0, 1)   # x,y,z + xyzw 四元数
    timestamp_ns: int = 0


# ============================================================
# 感知输出(Readme §3.3)
# ============================================================

@dataclass
class DetectedObject:
    name: str = ""
    bbox_xyxy: tuple = (0, 0, 0, 0)
    pose_xyz_rpy: tuple = (0, 0, 0, 0, 0, 0)     # x,y,z + roll,pitch,yaw
    confidence: float = 0.0


@dataclass
class SceneGraph:
    objects: list = field(default_factory=list)    # list[DetectedObject]
    obstacles: list = field(default_factory=list)
    timestamp_ns: int = 0


# ============================================================
# 规划(Readme §4-5)
# ============================================================

@dataclass
class TaskGoal:
    type: str = ""         # e.g. "assembly", "transport"
    target: str = ""       # e.g. "螺丝"
    location: str = ""     # e.g. "工作台"
    params: dict = field(default_factory=dict)


@dataclass
class Skill:
    id: str = ""           # SK_WALK / SK_CLIMB / SK_REACH / SK_GRASP / ...
    params: dict = field(default_factory=dict)


@dataclass
class SkillSequence:
    skills: list = field(default_factory=list)     # list[Skill]


# ============================================================
# 控制(Readme §6.3)
# ============================================================

@dataclass
class Trajectory:
    waypoints: list = field(default_factory=list)   # TODO: 定义状态维度(可能嵌套 dataclass)
    horizon_s: float = 0.0
    dt_s: float = 0.0


@dataclass
class JointCommand:
    position: list = field(default_factory=list)    # rad
    velocity: list = field(default_factory=list)    # rad/s
    effort: list = field(default_factory=list)      # N*m
    kp: list = field(default_factory=list)          # 位置刚度
    kd: list = field(default_factory=list)          # 阻尼


# ============================================================
# 反馈(Readme §7.2)
# ============================================================

class FailureType:
    F_STABILITY = "F_STABILITY"
    F_GRASP = "F_GRASP"
    F_COLLISION = "F_COLLISION"
    F_TRAJECTORY = "F_TRAJECTORY"
    F_TIMEOUT = "F_TIMEOUT"
    F_PERCEPTION = "F_PERCEPTION"


@dataclass
class ExecutionStatus:
    current_skill_id: str = ""
    progress: float = 0.0
    failure: str = ""        # FailureType.* 之一,空 = OK
    error_msg: str = ""
