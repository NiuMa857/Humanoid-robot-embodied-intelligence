"""common/ 单元测试:config + logging + types。"""
import sys
sys.path.insert(0, ".")
import os
import logging

from common.config import load_config, env_override, _coerce, _set_nested, clear_cache
from common.logging import setup_logger
from common.types import (
    JointState, IMUState, RobotState,
    DetectedObject, SceneGraph,
    TaskGoal, Skill, SkillSequence,
    Trajectory, JointCommand, FailureType, ExecutionStatus,
)


def test_types_defaults():
    js = JointState()
    assert js.name == [] and js.position == []
    assert js.timestamp_ns == 0
    rs = RobotState()
    assert rs.joint is None
    assert rs.base_pose == (0, 0, 0, 0, 0, 0, 1)
    sg = SkillSequence(skills=[Skill(id="SK_WALK")])
    assert sg.skills[0].id == "SK_WALK"
    assert FailureType.F_STABILITY == "F_STABILITY"


def test_config_load():
    cfg = load_config("default")
    assert isinstance(cfg, dict)
    assert "robot" in cfg
    assert cfg["robot"]["dof"] == 31


def test_config_coerce():
    assert _coerce("true") is True
    assert _coerce("false") is False
    assert _coerce("42") == 42
    assert _coerce("3.14") == 3.14
    assert _coerce("a,b,c") == ["a", "b", "c"]
    assert _coerce("hello") == "hello"


def test_set_nested():
    out = _set_nested({}, ["test", "val"], 123)
    assert out == {"test": {"val": 123}}


def test_env_override_via_env_var():
    """通过真实环境变量测试 env_override。"""
    os.environ["OLI_TESTVAL__X"] = "99"
    try:
        out = env_override({})
        assert out.get("testval", {}).get("x") == 99
    finally:
        del os.environ["OLI_TESTVAL__X"]


def test_logger_json():
    log = setup_logger("test_json_" + str(os.getpid()), level="INFO", json=True)
    log.info("hello", extra={"skill_id": "SK_WALK"})
    assert log.name.startswith("test_json_")


def test_logger_plain():
    log = setup_logger("test_plain_" + str(os.getpid()), level="DEBUG", json=False)
    assert log.level == logging.DEBUG
