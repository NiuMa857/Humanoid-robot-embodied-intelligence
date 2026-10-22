"""task_planning/ 单元测试:LLM 工厂 + fallback 路径。"""
import sys
sys.path.insert(0, ".")
import json
import os

from common.types import (
    TaskGoal, SceneGraph, Skill, RobotState, JointState, FailureType, DetectedObject,
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


class _RecordingLLM:
    """像真实客户端一样 json.dumps 入参,并记录调用。"""
    def __init__(self):
        self.decompose_calls = []
        self.replan_calls = []
    def understand_instruction(self, text):
        return {"type": "transport", "target": "box", "location": "workbench"}
    def decompose(self, goal, scene):
        json.dumps(goal, ensure_ascii=False)
        json.dumps(scene, ensure_ascii=False)
        self.decompose_calls.append((goal, scene))
        return [{"id": "SK_WALK", "params": {"target": "workbench"}}]
    def replan(self, failure, state):
        json.dumps(state, ensure_ascii=False)
        self.replan_calls.append((failure, state))
        return [{"id": "SK_STAND", "params": {}}]
    def close(self):
        pass


def _scene_with_objects():
    return SceneGraph(
        objects=[DetectedObject(name="box", bbox_xyxy=(10, 20, 30, 40),
                                pose_xyz_rpy=(0.5, 0.1, 0.8, 0, 0, 0), confidence=0.9)],
        obstacles=[DetectedObject(name="pallet", pose_xyz_rpy=(1.0, 0, 0.1, 0, 0, 0))],
        timestamp_ns=123,
    )


def test_task_decomposer_sends_serializable_scene_to_llm():
    """含 DetectedObject 的场景应被序列化后传给 LLM,而不是悄悄降级到模板。"""
    llm = _RecordingLLM()
    seq = TaskDecomposer(llm=llm).decompose(TaskGoal(type="transport", target="box"), _scene_with_objects())
    assert len(llm.decompose_calls) == 1
    _, scene = llm.decompose_calls[0]
    assert scene["objects"][0]["name"] == "box"
    assert scene["obstacles"][0]["name"] == "pallet"
    assert [s.id for s in seq.skills] == ["SK_WALK"]


def test_replanner_sends_serializable_scene_to_llm():
    llm = _RecordingLLM()
    state = RobotState(base_pose=(0, 0, 1.0, 0, 0, 0, 1))
    seq = Replanner(llm=llm).replan(FailureType.F_GRASP, "SK_GRASP", state, scene=_scene_with_objects())
    assert len(llm.replan_calls) == 1
    assert llm.replan_calls[0][1]["scene"]["objects"][0]["name"] == "box"
    assert [s.id for s in seq.skills] == ["SK_STAND"]


def test_planning_node_reads_nested_planning_config():
    """default.yaml 把 llm/vlm 配置放在 planning: 下。"""
    pn = PlanningNode(config={
        "planning": {
            "llm": {"mode": "local", "endpoint": "http://127.0.0.1:9000", "model": "qwen"},
            "vlm": {"mode": "local", "endpoint": "http://127.0.0.1:9001"},
        }
    })
    assert isinstance(pn.llm, LocalLLMClient)
    assert pn.llm.endpoint == "http://127.0.0.1:9000"
    assert pn.llm.model == "qwen"
    assert pn.vlm.endpoint == "http://127.0.0.1:9001"
    pn.shutdown()


def test_planning_node_with_default_yaml():
    from common.config import load_config
    cfg = load_config("default")
    pn = PlanningNode(config=cfg)
    assert pn.llm.model == cfg["planning"]["llm"]["model"]
    assert pn.vlm.model == cfg["planning"]["vlm"]["model"]
    pn.shutdown()


def test_llm_factory_reads_api_key_from_env():
    os.environ["_TEST_OLI_LLM_KEY"] = "secret-from-env"
    try:
        c = make_llm_client({"llm": {"mode": "remote", "endpoint": "https://x", "api_key_env": "_TEST_OLI_LLM_KEY"}})
        assert c.api_key == "secret-from-env"
        c2 = make_llm_client({"llm": {"mode": "remote", "api_key": "explicit", "api_key_env": "_TEST_OLI_LLM_KEY"}})
        assert c2.api_key == "explicit"
    finally:
        del os.environ["_TEST_OLI_LLM_KEY"]


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
