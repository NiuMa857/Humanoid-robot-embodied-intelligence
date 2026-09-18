"""训练环境基类(Readme §11,通用实现,无 Isaac 依赖)。

所有具体任务环境(StairEnv / IndustrialEnv)继承此类。
子类只需实现:`reset` / `step` / `observation_space` / `action_space`。
"""
from __future__ import annotations
from abc import abstractmethod
from typing import Any
from common.types import RobotState, JointState, IMUState, JointCommand
from common.logging import setup_logger

_log = setup_logger("base_env")


class BaseEnv:
    """训练环境基类(零依赖,可独立测试)。"""

    def __init__(self, task_config: dict):
        self.cfg = task_config
        self.num_envs: int = 1
        self.num_obs: int = 0
        self.num_actions: int = 0
        self._t: float = 0.0
        self._step_count: int = 0

    @abstractmethod
    def reset(self, env_ids: list[int] | None = None) -> tuple[Any, dict]:
        """重置(默认实现单 env;子类可重写为多 env 批量)。"""
        ...

    @abstractmethod
    def step(self, actions: Any) -> tuple[Any, float, bool, dict]:
        """单步。

        Returns:
            (obs, reward, done, info)
        """
        ...

    @property
    @abstractmethod
    def observation_space(self):
        ...

    @property
    @abstractmethod
    def action_space(self):
        ...

    def close(self) -> None:
        pass

    def seed(self, seed: int) -> None:
        """设置随机种子(默认用 Python random;子类可换 numpy/torch)。"""
        import random
        random.seed(seed)
        _log.info(f"seed set: {seed}")

    # ---------- 通用辅助 ----------

    def make_zero_state(self) -> RobotState:
        """构造一个"零"状态(子类 reset 时默认用)。"""
        return RobotState(
            joint=JointState(position=[0.0] * self.num_actions, velocity=[0.0] * self.num_actions),
            imu=IMUState(),
            base_pose=(0, 0, 1.0, 0, 0, 0, 1),
            timestamp_ns=0,
        )

    def elapsed_s(self) -> float:
        return self._t


__all__ = ["BaseEnv"]
