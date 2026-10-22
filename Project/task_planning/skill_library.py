"""技能库(Readme §5.2):8 个基元技能完整实装。

完整版(上一版是注册表),加:
  - check_feasibility:检查机器人状态是否能执行
  - estimate_duration:估算执行时长
  - get_required_capabilities:技能依赖的能力
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from common.types import Skill, SkillSequence, RobotState, JointState
from common.logging import setup_logger

_log = setup_logger("skill_library")


@dataclass
class SkillSpec:
    id: str
    name: str
    description: str
    required_dofs: list[str] = field(default_factory=list)
    requires_ee: bool = False
    timeout_s: float = 30.0
    estimated_duration_s: float = 5.0
    failure_tolerance: dict = field(default_factory=dict)
    """失败容忍度:每个子阶段允许失败次数,超过触发 F_xxx"""


class SkillLibrary:
    """技能注册表。"""

    SKILLS: dict[str, SkillSpec] = {
        "SK_WALK": SkillSpec(
            id="SK_WALK", name="平地行走",
            description="双足行走,速度<=5km/h",
            required_dofs=["left_leg", "right_leg"],
            timeout_s=60.0, estimated_duration_s=20.0,
        ),
        "SK_CLIMB": SkillSpec(
            id="SK_CLIMB", name="斜梯攀爬",
            description="斜坡/阶梯攀爬",
            required_dofs=["left_leg", "right_leg", "left_arm", "right_arm", "waist"],
            timeout_s=120.0, estimated_duration_s=60.0,
        ),
        "SK_REACH": SkillSpec(
            id="SK_REACH", name="手臂伸展",
            description="末端到达目标位置",
            required_dofs=["left_arm", "right_arm", "waist"],
            timeout_s=10.0, estimated_duration_s=3.0,
        ),
        "SK_GRASP": SkillSpec(
            id="SK_GRASP", name="抓取",
            description="夹爪闭合抓取物体",
            required_dofs=["left_arm", "right_arm"],
            requires_ee=True,
            timeout_s=15.0, estimated_duration_s=5.0,
            failure_tolerance={"grasp_miss": 2},
        ),
        "SK_PLACE": SkillSpec(
            id="SK_PLACE", name="放置",
            description="将物体放置到目标位置",
            required_dofs=["left_arm", "right_arm"],
            requires_ee=True,
            timeout_s=15.0, estimated_duration_s=5.0,
        ),
        "SK_ASSEMBLE": SkillSpec(
            id="SK_ASSEMBLE", name="装配",
            description="精准插入/旋拧操作",
            required_dofs=["left_arm", "right_arm", "waist"],
            requires_ee=True,
            timeout_s=30.0, estimated_duration_s=10.0,
            failure_tolerance={"precision_miss": 3},
        ),
        "SK_STAND": SkillSpec(
            id="SK_STAND", name="站立平衡",
            description="静态站立保持",
            required_dofs=["left_leg", "right_leg", "waist"],
            timeout_s=600.0, estimated_duration_s=5.0,
        ),
        "SK_TURN": SkillSpec(
            id="SK_TURN", name="转向",
            description="原地转向",
            required_dofs=["left_leg", "right_leg", "waist"],
            timeout_s=15.0, estimated_duration_s=5.0,
        ),
    }

    def get(self, skill_id: str) -> SkillSpec:
        if skill_id not in self.SKILLS:
            raise KeyError(f"unknown skill: {skill_id}")
        return self.SKILLS[skill_id]

    def list_ids(self) -> list[str]:
        return list(self.SKILLS.keys())

    MAX_SEQUENCE_LEN = 30

    def validate_sequence(self, raw: Any) -> tuple[SkillSequence | None, list[str]]:
        """校验 LLM 返回的原始技能列表。

        规则:非空 list、长度 <= MAX_SEQUENCE_LEN、每项是含 id 的 dict、
        id 在技能库中(大小写/空白归一化)、params 为 dict(缺省 {})。

        Returns:
            (SkillSequence, []) 校验通过;(None, errors) 任一项不合法。
        """
        if not isinstance(raw, list) or not raw:
            return None, [f"expected non-empty list, got {type(raw).__name__}: {raw!r:.80}"]
        if len(raw) > self.MAX_SEQUENCE_LEN:
            return None, [f"sequence too long: {len(raw)} > {self.MAX_SEQUENCE_LEN}"]
        skills, errors = [], []
        for i, item in enumerate(raw):
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                errors.append(f"[{i}] missing skill id: {item!r:.80}")
                continue
            skill_id = item["id"].strip().upper()
            if skill_id not in self.SKILLS:
                errors.append(f"[{i}] unknown skill: {item['id']}")
            params = item.get("params") or {}
            if not isinstance(params, dict):
                errors.append(f"[{i}] params must be dict, got {type(params).__name__}")
            skills.append(Skill(id=skill_id, params=params))
        if errors:
            return None, errors
        return SkillSequence(skills=skills), []

    def check_feasibility(self, skill: Skill, robot_state: RobotState) -> tuple[bool, str]:
        """检查机器人状态是否能执行该 skill。

        Returns:
            (feasible, reason):feasible=False 时 reason 解释为什么不能
        """
        spec = self.get(skill.id)
        # 1. 关节健康(简化:所有关节 velocity < 阈值视为健康)
        if robot_state.joint is not None:
            max_vel = max(abs(v) for v in robot_state.joint.velocity) if robot_state.joint.velocity else 0
            if max_vel > 10.0:  # rad/s
                return False, f"joint velocity too high: {max_vel:.1f} rad/s"
        # 2. 末端执行器在线(对需要 EE 的 skill)
        if spec.requires_ee:
            # TODO: 检查 EE 状态(Oli SDK 反馈)
            pass
        # 3. base 高度合理(没摔倒)
        if robot_state.base_pose[2] < 0.3:
            return False, f"base too low: {robot_state.base_pose[2]:.2f}m, robot may have fallen"
        return True, "ok"

    def estimate_duration(self, skill: Skill) -> float:
        return self.get(skill.id).estimated_duration_s

    def get_required_capabilities(self, skill: Skill) -> list[str]:
        """返回技能依赖的能力(用于上层调度检查)。"""
        spec = self.get(skill.id)
        caps = list(spec.required_dofs)
        if spec.requires_ee:
            caps.append("end_effector")
        return caps


__all__ = ["SkillLibrary", "SkillSpec"]
