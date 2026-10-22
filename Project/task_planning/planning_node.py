"""任务规划 ROS 2 节点入口(完整实装)。"""
from __future__ import annotations
import time
from common.types import SceneGraph, TaskGoal, Skill, SkillSequence, ExecutionStatus
from common.logging import setup_logger
from .llm_client import make_llm_client, make_vlm_client
from .task_decomposer import TaskDecomposer
from .skill_library import SkillLibrary
from .replanner import Replanner

_log = setup_logger("planning_node")


class PlanningNode:
    """任务规划节点(slow system,1-10 Hz)。

    除了生成技能序列,还跟踪执行进度(当前目标 / 已完成技能 / 序列游标),
    失败时把这些上下文交给 Replanner,使重规划能接着剩余计划继续。
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        # 支持传入完整 default.yaml(llm/vlm 在 planning: 下)或直接传 planning 段
        planning_cfg = self.config.get("planning", self.config)
        self.llm = make_llm_client(planning_cfg)
        self.vlm = make_vlm_client(planning_cfg)
        self.skills = SkillLibrary()
        self.decomposer = TaskDecomposer(llm=self.llm, skill_library=self.skills)
        self.replanner = Replanner(llm=self.llm, skill_library=self.skills)
        self._current_seq: SkillSequence | None = None
        self._current_goal: TaskGoal | None = None
        self._cursor: int = 0
        self._completed: list[Skill] = []
        self._last_scene: SceneGraph | None = None

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
        self._current_goal = goal
        self._last_scene = scene
        self._completed = []
        self._set_sequence(seq)
        return seq

    def on_skill_completed(self, skill_id: str) -> Skill | None:
        """控制层报告当前技能成功,游标前进。返回下一个待执行技能(没有则 None)。"""
        current = self._current_skill()
        if current is None:
            _log.warning(f"on_skill_completed({skill_id}) but no skill pending")
            return None
        if current.id != skill_id:
            _log.warning(f"on_skill_completed({skill_id}) but expected {current.id}")
            return current
        self._completed.append(current)
        self._cursor += 1
        return self._current_skill()

    def on_failure(
        self,
        failure: str,
        failed_skill_id: str,
        current_state,
        scene: SceneGraph | None = None,
    ) -> SkillSequence:
        """失败时带上目标 / 进度上下文重规划,替换当前序列。"""
        if scene is not None:
            self._last_scene = scene
        remaining: list[Skill] = []
        if self._current_seq is not None:
            current = self._current_skill()
            if current is not None and current.id == failed_skill_id:
                remaining = self._current_seq.skills[self._cursor + 1:]
            else:
                _log.warning(f"on_failure({failed_skill_id}) does not match current skill "
                             f"{current.id if current else None}; replanning without remaining plan")
        new_seq = self.replanner.replan(
            failure, failed_skill_id, current_state,
            scene=self._last_scene,
            goal=self._current_goal,
            completed=self._completed,
            remaining=remaining,
        )
        self._set_sequence(new_seq)
        return new_seq

    def get_current_sequence(self) -> SkillSequence | None:
        return self._current_seq

    @property
    def completed_skills(self) -> list[Skill]:
        """本次任务(跨重规划)已完成的技能。"""
        return list(self._completed)

    def _set_sequence(self, seq: SkillSequence) -> None:
        self._current_seq = seq
        self._cursor = 0

    def _current_skill(self) -> Skill | None:
        if self._current_seq is None or self._cursor >= len(self._current_seq.skills):
            return None
        return self._current_seq.skills[self._cursor]

    def shutdown(self) -> None:
        self.llm.close()
        self.vlm.close()


__all__ = ["PlanningNode"]
