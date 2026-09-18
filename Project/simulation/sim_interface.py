"""仿真统一接口(Readme §9.4)。"""
from __future__ import annotations
from abc import ABC, abstractmethod
from common.types import RobotState, JointCommand, SceneGraph


class SimInterface(ABC):
    """仿真器统一抽象。Real 模式用空实现或 Oli SDK 替代。"""

    @abstractmethod
    def load_robot(self, urdf_path: str) -> None: ...

    @abstractmethod
    def load_scene(self, scene_config: str) -> None: ...

    @abstractmethod
    def reset(self) -> RobotState:
        """重置到初始位姿,返回初始状态。"""
        ...

    @abstractmethod
    def step(self, cmd: JointCommand) -> RobotState:
        """单步仿真,返回新状态。"""
        ...

    def get_ground_truth(self) -> dict:
        """获取 ground truth(用于验证 / 评估,Readme §9.4)。默认返回空。"""
        return {}

    def get_scene_graph(self) -> SceneGraph:
        """获取仿真器"看到"的场景(给 perception 模块,用于 sim2real 验证)。"""
        return SceneGraph()

    def close(self) -> None:
        """释放资源。"""
        pass


class MockSimulator(SimInterface):
    """Mock 仿真器:无重力、无接触,纯运动学。

    用途:
      - 单元测试(不需要真仿真器)
      - 验证接口流程
      - CI 跑通完整 pipeline
    """

    def __init__(self, num_dof: int = 31):
        self.num_dof = num_dof
        self._state: RobotState | None = None
        self._t: float = 0.0
        self._dt: float = 0.01

    def load_robot(self, urdf_path: str) -> None:
        pass

    def load_scene(self, scene_config: str) -> None:
        pass

    def reset(self) -> RobotState:
        self._t = 0.0
        self._state = RobotState(
            base_pose=(0, 0, 1.0, 0, 0, 0, 1),
            timestamp_ns=0,
        )
        return self._state

    def step(self, cmd: JointCommand) -> RobotState:
        self._t += self._dt
        # 简化的关节位置累积
        if self._state is not None and self._state.joint is not None and cmd.position:
            n = min(len(cmd.position), len(self._state.joint.position))
            new_pos = [self._state.joint.position[i] + cmd.position[i] * self._dt for i in range(n)]
            self._state.joint.position = new_pos
        return self._state

    def get_ground_truth(self) -> dict:
        return {"time": self._t, "joint_count": self.num_dof}


__all__ = ["SimInterface", "MockSimulator"]
