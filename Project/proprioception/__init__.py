"""proprioception — 本体感知层(Readme §3,1 kHz)。

支持 mock / sim / real 三种模式:
  - mock: 虚拟数据,跑测试用
  - sim:  从仿真器(Isaac Sim)读
  - real: 从 Oli SDK 读
"""
from .imu_reader import IMUReader
from .joint_state import JointStateReader
from .state_estimator import (
    StateEstimator, KinematicsBackend,
    quat_normalize, quat_rotate, quat_to_euler_zyx,
)
from .proprioception_node import ProprioceptionNode

__all__ = [
    "IMUReader", "JointStateReader",
    "StateEstimator", "KinematicsBackend",
    "quat_normalize", "quat_rotate", "quat_to_euler_zyx",
    "ProprioceptionNode",
]
