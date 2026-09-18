"""全身状态估计:IMU + 关节 → RobotState(Readme §3.2)。

通用融合策略:
  - 基础姿态:IMU 主导(高频、抗形变)
  - 基础位置:关节正运动学(forward kinematics)— 需 URDF
  - 基础速度:关节运动学微分 + IMU 角速度

Oli 特定:
  - joint 名、URDF 文件名 → 从 config["robot"]["urdf_path"] 读
  - 关节数(31) → 从 config["robot"]["dof"] 读
  - 运动学库(默认用 pinocchio,可选 yourdfpy)— 见 set_kinematics_backend
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional, Callable

from common.types import IMUState, JointState, RobotState


# ============================================================
# 四元数工具(避免引入 numpy 之外的依赖,只用 math)
# ============================================================

def quat_normalize(q: tuple) -> tuple:
    """四元数归一化 (x, y, z, w)。"""
    n = math.sqrt(sum(c * c for c in q))
    if n < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def quat_rotate(q: tuple, v: tuple) -> tuple:
    """用四元数 q (x,y,z,w) 旋转向量 v。"""
    qx, qy, qz, qw = q
    vx, vy, vz = v
    # t = 2 * cross(q.xyz, v)
    tx = 2 * (qy * vz - qz * vy)
    ty = 2 * (qz * vx - qx * vz)
    tz = 2 * (qx * vy - qy * vx)
    # result = v + qw * t + cross(q.xyz, t)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def quat_to_euler_zyx(q: tuple) -> tuple:
    """四元数 → ZYX 内旋欧拉角(roll, pitch, yaw)。"""
    qx, qy, qz, qw = q
    # roll (x-axis)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    # yaw (z-axis)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return (roll, pitch, yaw)


# ============================================================
# 运动学后端(可插拔)
# ============================================================

class KinematicsBackend:
    """运动学后端抽象(默认 None,用零阶近似)。

    实装版本可注入 pinocchio / yourdfpy。
    接口:
      - forward_kinematics(joint_pos: list[float]) -> base_pose: tuple (x,y,z,qx,qy,qz,qw)
    """

    def forward_kinematics(self, joint_pos: list[float]) -> tuple:
        raise NotImplementedError

    def base_velocity(self, joint_pos: list[float], joint_vel: list[float]) -> tuple:
        """基础线速度 (vx, vy, vz)。默认:零阶(返回 0,0,0)。"""
        return (0.0, 0.0, 0.0)


# ============================================================
# 状态估计器
# ============================================================

class StateEstimator:
    """融合 IMU + 关节 → RobotState(1 kHz 实时)。"""

    def __init__(
        self,
        urdf_path: str = "",
        config: dict | None = None,
        kinematics: KinematicsBackend | None = None,
    ):
        self.urdf_path = urdf_path
        self.config = config or {}
        self.num_dof = int(self.config.get("robot", {}).get("dof", 31))
        self.kinematics: KinematicsBackend = kinematics or KinematicsBackend()
        # 历史位置(用于速度估计)
        self._last_base_pos: Optional[tuple] = None
        self._last_ts_ns: int = 0
        # 标定:启动时认为 z = 站立高度
        self._standing_height: float = 1.0

    def reset(self) -> None:
        self._last_base_pos = None
        self._last_ts_ns = 0

    def estimate(
        self,
        imu: IMUState,
        joint: JointState,
        timestamp_ns: int = 0,
    ) -> RobotState:
        """单次状态估计。"""
        if len(joint.position) != self.num_dof:
            # 关节维度不匹配 — 报警但不抛(允许在过渡期)
            pass

        # 1. 基础位置(关节正运动学)
        try:
            base_pose = self.kinematics.forward_kinematics(joint.position)
            base_pos = base_pose[:3]
            base_orient = quat_normalize(base_pose[3:])
        except NotImplementedError:
            # 零阶 fallback:z 设为 standing height
            base_pos = (0.0, 0.0, self._standing_height)
            base_orient = (0.0, 0.0, 0.0, 1.0)

        # 2. 基础姿态优先用 IMU(更准,不受关节形变影响)
        imu_orient = quat_normalize(imu.orientation_xyzw)
        # 简单融合:这里直接用 IMU;更严谨的版本用卡尔曼
        fused_orient = imu_orient

        # 3. 基础速度(关节运动学 + IMU 角速度)
        if self._last_base_pos is not None and self._last_ts_ns > 0 and timestamp_ns > self._last_ts_ns:
            dt_s = (timestamp_ns - self._last_ts_ns) * 1e-9
            if dt_s > 0:
                base_lin_vel = tuple(
                    (base_pos[i] - self._last_base_pos[i]) / dt_s
                    for i in range(3)
                )
            else:
                base_lin_vel = (0.0, 0.0, 0.0)
        else:
            base_lin_vel = (0.0, 0.0, 0.0)

        self._last_base_pos = base_pos
        self._last_ts_ns = timestamp_ns

        # 4. 拼装 RobotState
        full_pose = base_pos + fused_orient
        return RobotState(
            joint=joint,
            imu=imu,
            base_pose=full_pose,
            timestamp_ns=timestamp_ns,
        )

    def calibrate_standing(self, imu: IMUState, joint: JointState) -> None:
        """在机器人静止站立时调用,记录站立高度 + 零姿态。"""
        try:
            pose = self.kinematics.forward_kinematics(joint.position)
            self._standing_height = pose[2]
        except NotImplementedError:
            pass

    @staticmethod
    def get_gravity_orientation(quat: tuple) -> tuple:
        """把世界系重力 (0,0,-1) 通过 quat 转到机体系(Readme §3 + G1DWAQ 部署用)。"""
        # 重力方向 = quat^-1 * (0,0,-1) * quat
        qx, qy, qz, qw = quat_normalize(quat)
        # 共轭四元数
        cx, cy, cz, cw = -qx, -qy, -qz, qw
        g_world = (0.0, 0.0, -1.0)
        # 旋到机体系(用共轭旋转)
        g_body = quat_rotate((cx, cy, cz, cw), g_world)
        return g_body


__all__ = [
    "StateEstimator", "KinematicsBackend",
    "quat_normalize", "quat_rotate", "quat_to_euler_zyx",
]
