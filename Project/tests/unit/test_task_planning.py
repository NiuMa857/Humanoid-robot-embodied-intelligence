"""task_planning/ 单元测试:LLM 工厂 + fallback 路径。"""
import sys
sys.path.insert(0, ".")

from common.types import (
    TaskGoal, SceneGraph, Skill, RobotState, JointState, FailureType,
)
from task_planning import (
    TaskDecomposer, Replanner, SkillLibrary,
    RemoteLLMClient, LocalLLMClient,
    make_llm_client, make_vlm_client, LLMError,
    PlanningNode,
)


class _FailingLLM:
    """总是抛 LLMError,用于测试 fallback 路径。"""
    def understand_instruction(self, text):
        raise LLMError("mock")
    def decompose(self, goal, scene):
        raise LLMError("mock")
    def replan(self, failure, state):
        raise LLMError("mock")
    def close(self):
        pass


def test_skill_library_basic():
    lib = SkillLibrary()
    assert "SK_WALK" in lib.list_ids()
    assert lib.get("SK_GRASP").requires_ee is True
    assert lib.get("SK_CLIMB").estimated_duration_s > 0


def test_skill_feasibility_fall_detection():
    lib = SkillLibrary()
    state_ok = RobotState(base_pose=(0, 0, 1.0, 0, 0, 0, 1))
    state_fall = RobotState(base_pose=(0, 0, 0.1, 0, 0, 0, 1))
    ok, _ = lib.check_feasibility(Skill(id="SK_WALK"), state_ok)
    assert ok
    ok, reason = lib.check_feasibility(Skill(id="SK_WALK"), state_fall)
    assert not ok and "low" in reason.lower()


def test_task_decomposer_fallback_transport():
    """LLM 失败时退化到 transport 模板。"""
    d = TaskDecomposer(llm=_FailingLLM())
    seq = d.decompose(TaskGoal(type="transport", target="box", location="warehouse"), SceneGraph())
    ids = [s.id for s in seq.skills]
    assert ids[0] == "SK_STAND"
    assert "SK_GRASP" in ids
    assert "SK_PLACE" in ids


def test_replanner_fallback_per_failure_type():
    r = Replanner(llm=_FailingLLM())
    state = RobotState(base_pose=(0, 0, 1.0, 0, 0, 0, 1))
    for ft in [FailureType.F_STABILITY, FailureType.F_GRASP, FailureType.F_COLLISION,
               FailureType.F_TIMEOUT, FailureType.F_TRAJECTORY, FailureType.F_PERCEPTION]:
        seq = r.replan(ft, "SK_WALK", state)
        assert len(seq.skills) > 0


def test_llm_factory():
    c1 = make_llm_client({"llm": {"mode": "remote", "endpoint": "https://x", "api_key": "k"}})
    assert isinstance(c1, RemoteLLMClient)
    c2 = make_llm_client({"llm": {"mode": "local", "endpoint": "http://localhost:8000"}})
    assert isinstance(c2, LocalLLMClient)
    v1 = make_vlm_client({"vlm": {"mode": "local"}})
    assert v1.endpoint == "http://127.0.0.1:8001"


def test_planning_node_instruction_to_sequence():
    pn = PlanningNode(config={"llm": {"mode": "remote", "endpoint": "", "api_key": ""}})
    # 无 LLM endpoint 时,会 fallback
    seq = pn.on_instruction("把箱子搬到工作台", SceneGraph())
    assert seq is not None
    assert len(seq.skills) > 0
    pn.shutdown()
