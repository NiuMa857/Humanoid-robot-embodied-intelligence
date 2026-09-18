"""task_planning.skill_library 单元测试。"""
import sys
sys.path.insert(0, ".")

from common.types import RobotState, JointState, Skill
from task_planning.skill_library import SkillLibrary, SkillSpec


def test_skill_library_has_all_skills():
    lib = SkillLibrary()
    expected = {"SK_WALK", "SK_CLIMB", "SK_REACH", "SK_GRASP",
                "SK_PLACE", "SK_ASSEMBLE", "SK_STAND", "SK_TURN"}
    assert set(lib.list_ids()) == expected


def test_skill_get_returns_spec():
    lib = SkillLibrary()
    spec = lib.get("SK_CLIMB")
    assert isinstance(spec, SkillSpec)
    assert spec.requires_ee is False
    assert "left_leg" in spec.required_dofs


def test_skill_get_ee_required():
    lib = SkillLibrary()
    assert lib.get("SK_GRASP").requires_ee is True
    assert lib.get("SK_ASSEMBLE").requires_ee is True
    assert lib.get("SK_WALK").requires_ee is False


def test_skill_unknown_raises():
    lib = SkillLibrary()
    try:
        lib.get("SK_NOT_EXIST")
        assert False, "should have raised"
    except KeyError:
        pass


def test_skill_feasibility_fall_detection():
    lib = SkillLibrary()
    state_ok = RobotState(base_pose=(0, 0, 1.0, 0, 0, 0, 1))
    state_fall = RobotState(base_pose=(0, 0, 0.1, 0, 0, 0, 1))
    ok, _ = lib.check_feasibility(Skill(id="SK_WALK"), state_ok)
    assert ok
    ok, reason = lib.check_feasibility(Skill(id="SK_WALK"), state_fall)
    assert not ok
    assert "low" in reason.lower()


def test_skill_estimate_duration():
    lib = SkillLibrary()
    assert lib.estimate_duration(Skill(id="SK_WALK")) > 0
    assert lib.estimate_duration(Skill(id="SK_CLIMB")) > lib.estimate_duration(Skill(id="SK_REACH"))


def test_skill_required_capabilities():
    lib = SkillLibrary()
    caps = lib.get_required_capabilities(Skill(id="SK_GRASP"))
    assert "end_effector" in caps
    assert "left_arm" in caps
