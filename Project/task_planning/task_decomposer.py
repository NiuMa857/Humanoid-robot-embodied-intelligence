"""任务分解器:TaskGoal + SceneGraph → SkillSequence(完整实装)。

支持 LLM 调用 + 降级模板。
"""
from __future__ import annotations
from common.types import TaskGoal, SceneGraph, Skill, SkillSequence, to_dict
from common.logging import setup_logger
from .llm_client import BaseLLMClient, LLMError

_log = setup_logger("task_decomposer")


class TaskDecomposer:
    """调 LLM 做任务分解;失败时降级到本地预设模板(Readme §10.2)。"""

    def __init__(self, llm: BaseLLMClient, fallback_templates: dict | None = None):
        self.llm = llm
        self.fallback_templates = fallback_templates or _DEFAULT_TEMPLATES

    def decompose(self, goal: TaskGoal, scene: SceneGraph) -> SkillSequence:
        """主入口。LLM 失败自动降级。"""
        goal_dict, scene_dict = to_dict(goal), to_dict(scene)
        try:
            raw_skills = self.llm.decompose(goal_dict, scene_dict)
            return self._parse(raw_skills)
        except (LLMError, Exception) as e:
            _log.warning(f"LLM decompose failed ({type(e).__name__}: {e}), falling back to template")
            return self._fallback(goal, scene)

    def _parse(self, raw: list[dict]) -> SkillSequence:
        return SkillSequence(skills=[Skill(id=s.get("id", ""), params=s.get("params", {})) for s in raw])

    def _fallback(self, goal: TaskGoal, scene: SceneGraph) -> SkillSequence:
        """按 goal.type 选预设模板。"""
        template = self.fallback_templates.get(goal.type, self.fallback_templates.get("_default", []))
        skills = [Skill(id=s["id"], params=s.get("params", {})) for s in template]
        # 用 goal 替换参数占位符
        for s in skills:
            if "{target}" in s.params:
                s.params["target"] = goal.target
            if "{location}" in s.params:
                s.params["location"] = goal.location
        return SkillSequence(skills=skills)


# 缺省降级模板(简单示例,真用时按业务扩展)
_DEFAULT_TEMPLATES: dict[str, list[dict]] = {
    "transport": [
        {"id": "SK_STAND", "params": {}},
        {"id": "SK_WALK", "params": {"target": "{location}"}},
        {"id": "SK_REACH", "params": {"target": "{target}"}},
        {"id": "SK_GRASP", "params": {"force": 10}},
        {"id": "SK_WALK", "params": {"target": "home"}},
        {"id": "SK_PLACE", "params": {}},
    ],
    "assembly": [
        {"id": "SK_STAND", "params": {}},
        {"id": "SK_WALK", "params": {"target": "{location}"}},
        {"id": "SK_REACH", "params": {"target": "{target}"}},
        {"id": "SK_GRASP", "params": {"force": 8}},
        {"id": "SK_ASSEMBLE", "params": {"precision": "high"}},
    ],
    "climb": [
        {"id": "SK_STAND", "params": {}},
        {"id": "SK_CLIMB", "params": {"target": "stairs_30deg"}},
    ],
    "_default": [
        {"id": "SK_STAND", "params": {}},
    ],
}


__all__ = ["TaskDecomposer"]
