"""本体感知 ROS 2 节点入口。

订阅:imu / joint_states / ft_sensor
发布:robot_state(1 kHz,Readme §3.3)

留 ROS 2 集成点(无 rclpy 时也能 import 测试)。
"""
from __future__ import annotations
import time
from common.types import RobotState
from .imu_reader import IMUReader
from .joint_state import JointStateReader
from .state_estimator import StateEstimator


class ProprioceptionNode:
    """本体感知节点(支持 mock 模式跑测试 / sim 模式跑仿真 / real 模式接 Oli SDK)。"""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.mode = self.config.get("mode", "mock")   # mock | sim | real
        self.freq_hz = int(self.config.get("freq_hz", 1000))
        self.dt_s = 1.0 / self.freq_hz

        self.imu = IMUReader(mode=self.mode, config=self.config.get("imu"))
        self.joints = JointStateReader(
            num_dof=int(self.config.get("robot", {}).get("dof", 31)),
            mode=self.mode,
            config=self.config,
        )
        self.estimator = StateEstimator(
            urdf_path=self.config.get("urdf_path", ""),
            config=self.config,
        )

    def calibrate(self) -> None:
        """静止状态下标定(开机时调用一次)。"""
        self.imu.calibrate()
        # 标定站立姿态
        joint = self.joints.read()
        imu = self.imu.read()
        self.estimator.calibrate_standing(imu, joint)

    def step(self) -> RobotState:
        """单步:读 IMU+关节 → 融合 → 输出 RobotState。"""
        imu = self.imu.read()
        joint = self.joints.read()
        return self.estimator.estimate(imu, joint, timestamp_ns=time.time_ns())

    def run(self, max_steps: int | None = None) -> None:
        """主循环:1 kHz step,直到 KeyboardInterrupt 或 max_steps。"""
        try:
            i = 0
            while max_steps is None or i < max_steps:
                state = self.step()
                # TODO: 发布到 ROS 2 topic
                # self.publisher.publish(state)
                time.sleep(self.dt_s)
                i += 1
        except KeyboardInterrupt:
            pass

    def shutdown(self) -> None:
        self.imu.close()
        self.joints.close()


__all__ = ["ProprioceptionNode"]
