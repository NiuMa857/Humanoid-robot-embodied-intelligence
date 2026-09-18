"""Isaac ↔ ROS 2 桥接。

让 ROS 节点在 Isaac 仿真里跑同一份代码(Readme §9.3 Sim2Real)。
"""
from __future__ import annotations
from common.logging import setup_logger

_log = setup_logger("isaac_bridge")


class IsaacBridge:
    """把 Isaac 仿真数据转 ROS 2 消息(或反之)。

    可选两种实装:
      1. 用 NVIDIA Isaac ROS Bridge(需要装 isaac_ros)
      2. 自实现:从 Isaac 读 sensor 数据 → 写 ROS topic
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._publishers = {}
        self._subscribers = {}

    def publish_camera(self, name: str, rgb, depth) -> None:
        """把 Isaac 相机图像发到 ROS 2 topic。"""
        # TODO: 用 rclpy / sensor_msgs/Image
        pass

    def publish_joint_states(self, joint_state) -> None:
        """把 Isaac 的关节状态发到 /oli/sensors/joint_states。"""
        # TODO
        pass

    def publish_imu(self, imu_state) -> None:
        """把 Isaac IMU 状态发到 /oli/sensors/imu。"""
        # TODO
        pass

    def subscribe_joint_command(self, callback) -> None:
        """订阅 /oli/control/joint_cmd,把 ROS 消息转给控制层。"""
        # TODO
        pass

    def start(self) -> None:
        _log.info("IsaacBridge started (placeholder)")

    def stop(self) -> None:
        _log.info("IsaacBridge stopped")


__all__ = ["IsaacBridge"]
