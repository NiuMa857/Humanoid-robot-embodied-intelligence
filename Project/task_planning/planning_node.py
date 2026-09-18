"""任务规划 ROS 2 节点入口(完整实装)。"""
from __future__ import annotations
import time
from common.types import SceneGraph, TaskGoal, SkillSequence, ExecutionStatus
from common.logging import setup_logger
from .llm_client import make_llm_client, make_vlm_client
from .task_decomposer import TaskDecomposer
from .skill_library import SkillLibrary
from .replanner import Replanner

_log = setup_logger("planning_node")


class PlanningNode:
    """任务规划节点(slow system,1-10 Hz)。"""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.llm = make_llm_client(self.config)
        self.vlm = make_vlm_client(self.config)
        self.skills = SkillLibrary()
        self.decomposer = TaskDecomposer(llm=self.llm)
        self.replanner = Replanner(llm=self.llm)
        self._current_seq: SkillSequence | None = None

    def on_instruction(self, text: str, scene: SceneGraph) -> SkillSequence:
        """收到自然语言指令时:理解 → 分解 → 缓存 → 返回。"""
        # 1. 理解
        try:
            goal_dict = self.llm.understand_instruction(text)
        except Exception as e:
            _log.warning(f"LLM understand_instruction failed: {e}; using empty goal")
            goal_dict = {"type": "_default", "target": "", "location": ""}

        goal = TaskGoal(
            type=goal_dict.get("type", "_default"),
            target=goal_dict.get("target", ""),
            location=goal_dict.get("location", ""),
            params=goal_dict.get("params", {}),
        )

        # 2. 分解
        seq = self.decomposer.decompose(goal, scene)
        self._current_seq = seq
        return seq

    def on_failure(self, failure: str, failed_skill_id: str, current_state) -> SkillSequence:
        """失败时重规划,替换当前序列。"""
        new_seq = self.replanner.replan(failure, failed_skill_id, current_state)
        self._current_seq = new_seq
        return new_seq

    def get_current_sequence(self) -> SkillSequence | None:
        return self._current_seq

    def shutdown(self) -> None:
        self.llm.close()
        self.vlm.close()


__all__ = ["PlanningNode"]
