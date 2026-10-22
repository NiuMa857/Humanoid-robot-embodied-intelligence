"""重规划:失败检测 + 触发(Readme §5.1 + §7.2)。

实现:
  - 失败分类(L1-L5 通用规则,基于执行状态)
  - LLM 调重规划
  - 降级:每种失败类型一个兜底技能序列
"""
from __future__ import annotations
from common.types import Skill, SkillSequence, RobotState, FailureType, ExecutionStatus, SceneGraph, to_dict
from common.logging import setup_logger
from .llm_client import BaseLLMClient, LLMError

_log = setup_logger("replanner")


class Replanner:
    """失败检测 + 触发重规划。"""

    def __init__(self, llm: BaseLLMClient):
        self.llm = llm

    def classify_failure(
        self,
        expected: RobotState,
        actual: RobotState,
        execution: ExecutionStatus,
    ) -> str:
        """失败分类(Readme §7.2)。

        规则:
          - 实际 base 高度 < 0.3m 或 倾角 > 60° → F_STABILITY
          - 关节力矩突变(> 2x 平均)→ F_COLLISION
          - 进度长时间不动 → F_TIMEOUT
          - 检测到但 grasp 失败 → F_GRASP
          - 跟踪误差 > 阈值 → F_TRAJECTORY
          - 感知中断 → F_PERCEPTION
        """
        # 简化的稳定性检测
        if actual.base_pose[2] < 0.3:
            return FailureType.F_STABILITY

        # TODO: 倾角检测、力矩突变检测等需要更复杂逻辑,留给实装
        if execution.failure:
            return execution.failure
        return FailureType.F_TIMEOUT  # 默认

    def replan(
        self,
        failure: str,
        failed_skill_id: str,
        current_state: RobotState,
        scene: SceneGraph | dict | None = None,
    ) -> SkillSequence:
        """主入口。"""
        state = {
            "failed_skill": failed_skill_id,
            "base_pose": list(current_state.base_pose),
            "scene": to_dict(scene),
        }
        # 先尝试 LLM
        try:
            raw = self.llm.replan(failure=failure, state=state)
            return SkillSequence(
                skills=[Skill(id=s.get("id", ""), params=s.get("params", {})) for s in raw]
            )
        except (LLMError, Exception) as e:
            _log.warning(f"LLM replan failed ({type(e).__name__}), falling back")
            return self._fallback(failure, failed_skill_id)

    def _fallback(self, failure: str, failed_skill_id: str) -> SkillSequence:
        """LLM 不可用时的硬编码兜底(Readme §10.2)。"""
        if failure == FailureType.F_STABILITY:
            # 失稳:先恢复站立
            return SkillSequence(skills=[Skill(id="SK_STAND", params={})])
        if failure == FailureType.F_GRASP:
            # 抓取失败:重试 1 次,然后放弃
            return SkillSequence(skills=[Skill(id="SK_REACH", params={}), Skill(id="SK_GRASP", params={"retry": True})])
        if failure == FailureType.F_COLLISION:
            # 碰撞:反向运动 + 重新规划
            return SkillSequence(skills=[Skill(id="SK_TURN", params={"angle_deg": 30}), Skill(id="SK_WALK", params={"retry": True})])
        if failure == FailureType.F_TRAJECTORY:
            # 轨迹偏差:重做当前技能
            return SkillSequence(skills=[Skill(id=failed_skill_id, params={"retry": True})])
        if failure == FailureType.F_TIMEOUT:
            # 超时:放弃
            return SkillSequence(skills=[Skill(id="SK_STAND", params={})])
        if failure == FailureType.F_PERCEPTION:
            # 感知失败:请求重感知
            return SkillSequence(skills=[Skill(id="SK_STAND", params={"wait_for_perception": True})])
        return SkillSequence(skills=[Skill(id="SK_STAND", params={})])


__all__ = ["Replanner"]
