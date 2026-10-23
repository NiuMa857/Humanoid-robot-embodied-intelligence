"""技能级执行环境接口。

与 `SimInterface`(关节级,1 kHz)互补:认知决策层只关心"执行一个技能 → 成功/失败 + 新观测",
不关心关节指令。符号仿真、Isaac Sim、Oli 真机分别实现本接口,规划器代码不变。
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from common.types import Skill


@dataclass
class SkillResult:
    success: bool
    failure: str = ""        # FailureType.* 之一,成功时为空
    detail: str = ""         # 给重规划器的失败原因
    duration_s: float = 0.0  # 仿真/实际耗时


class SkillEnv(ABC):
    """技能级环境:reset → (execute → observe)* → goal_satisfied。"""

    @abstractmethod
    def reset(self, task: Any, seed: int | None = None) -> dict:
        """按任务初始化环境,返回初始观测(给规划器的场景 dict)。"""

    @abstractmethod
    def execute(self, skill: Skill) -> SkillResult:
        """执行一个技能并推进环境状态。"""

    @abstractmethod
    def observe(self) -> dict:
        """当前观测(场景 dict,可 JSON 序列化)。"""

    @abstractmethod
    def goal_satisfied(self) -> bool:
        """任务目标是否达成(评测用 ground truth,规划器不可见)。"""


__all__ = ["SkillEnv", "SkillResult"]
