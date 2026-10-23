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


def test_llm_factory_passes_temperature_and_extra_body():
    c = make_llm_client({"llm": {
        "mode": "remote", "endpoint": "https://x", "temperature": 0.0,
        "extra_body": {"thinking": {"type": "disabled"}},
    }})
    assert c.temperature == 0.0
    assert c.extra_body == {"thinking": {"type": "disabled"}}
    assert make_llm_client({"llm": {"mode": "local"}}).extra_body == {}


def test_llm_factory_reads_api_key_from_env():
    os.environ["_TEST_OLI_LLM_KEY"] = "secret-from-env"
    try:
        c = make_llm_client({"llm": {"mode": "remote", "endpoint": "https://x", "api_key_env": "_TEST_OLI_LLM_KEY"}})
        assert c.api_key == "secret-from-env"
        c2 = make_llm_client({"llm": {"mode": "remote", "api_key": "explicit", "api_key_env": "_TEST_OLI_LLM_KEY"}})
        assert c2.api_key == "explicit"
    finally:
        del os.environ["_TEST_OLI_LLM_KEY"]


class _ScriptedLLM:
    """按预设返回 decompose / replan 结果,并记录 replan 的入参。"""
    def __init__(self, decompose_result=None, replan_result=None):
        self.decompose_result = decompose_result
        self.replan_result = replan_result
        self.replan_states = []
    def understand_instruction(self, text):
        return {"type": "transport", "target": "box", "location": "workbench"}
    def decompose(self, goal, scene):
        return self.decompose_result
    def replan(self, failure, state):
        json.dumps(state, ensure_ascii=False)
        self.replan_states.append(state)
        if isinstance(self.replan_result, Exception):
            raise self.replan_result
        return self.replan_result
    def close(self):
        pass


_TRANSPORT_PLAN = [
    {"id": "SK_WALK", "params": {"target": "shelf"}},
    {"id": "SK_REACH", "params": {"target": "box"}},
    {"id": "SK_GRASP", "params": {"force": 10}},
    {"id": "SK_WALK", "params": {"target": "workbench"}},
    {"id": "SK_PLACE", "params": {}},
]


def test_task_decomposer_rejects_unknown_skill_and_falls_back():
    llm = _ScriptedLLM(decompose_result=[{"id": "SK_WALK"}, {"id": "SK_FLY", "params": {}}])
    seq = TaskDecomposer(llm=llm).decompose(TaskGoal(type="transport", target="box"), SceneGraph())
    assert "SK_FLY" not in [s.id for s in seq.skills]
    assert [s.id for s in seq.skills][0] == "SK_STAND"   # transport 模板


def test_task_decomposer_rejects_malformed_output():
    for bad in [[], None, {"skills": "x"}, [{"params": {}}], [{"id": "SK_WALK", "params": "fast"}]]:
        llm = _ScriptedLLM(decompose_result=bad)
        seq = TaskDecomposer(llm=llm).decompose(TaskGoal(type="climb"), SceneGraph())
        assert [s.id for s in seq.skills] == ["SK_STAND", "SK_CLIMB"], bad


def test_task_decomposer_fallback_fills_placeholders():
    """模板里的 {target}/{location} 应被替换成 goal 的值,且不污染共享模板。"""
    d = TaskDecomposer(llm=_FailingLLM())
    seq = d.decompose(TaskGoal(type="transport", target="box", location="warehouse"), SceneGraph())
    assert seq.skills[1].params["target"] == "warehouse"
    assert seq.skills[2].params["target"] == "box"
    seq2 = d.decompose(TaskGoal(type="transport", target="bolt", location="bench"), SceneGraph())
    assert seq2.skills[1].params["target"] == "bench"
    assert seq2.skills[2].params["target"] == "bolt"


def test_task_decomposer_normalizes_skill_id():
    llm = _ScriptedLLM(decompose_result=[{"id": " sk_walk ", "params": {"target": "a"}}])
    seq = TaskDecomposer(llm=llm).decompose(TaskGoal(type="transport"), SceneGraph())
    assert [s.id for s in seq.skills] == ["SK_WALK"]
    assert seq.skills[0].params == {"target": "a"}


def test_skill_library_validate_sequence():
    lib = SkillLibrary()
    seq, errors = lib.validate_sequence(_TRANSPORT_PLAN)
    assert errors == [] and len(seq.skills) == 5
    seq, errors = lib.validate_sequence([{"id": "SK_FLY"}])
    assert seq is None and any("SK_FLY" in e for e in errors)
    seq, errors = lib.validate_sequence([{"id": "SK_WALK"}] * 100)
    assert seq is None and errors


def test_replanner_sends_goal_and_progress_to_llm():
    llm = _ScriptedLLM(replan_result=[{"id": "SK_GRASP", "params": {"retry": True}}])
    state = RobotState(base_pose=(0, 0, 1.0, 0, 0, 0, 1))
    Replanner(llm=llm).replan(
        FailureType.F_GRASP, "SK_GRASP", state,
        goal=TaskGoal(type="transport", target="box", location="workbench"),
        completed=[Skill(id="SK_WALK"), Skill(id="SK_REACH")],
        remaining=[Skill(id="SK_WALK", params={"target": "workbench"}), Skill(id="SK_PLACE")],
    )
    sent = llm.replan_states[0]
    assert sent["goal"]["target"] == "box"
    assert [s["id"] for s in sent["completed_skills"]] == ["SK_WALK", "SK_REACH"]
    assert [s["id"] for s in sent["remaining_skills"]] == ["SK_WALK", "SK_PLACE"]


def test_replanner_fallback_resumes_remaining_plan():
    """LLM 不可用时:恢复动作 + 重试失败技能 + 继续剩余计划(超时则放弃)。"""
    r = Replanner(llm=_FailingLLM())
    state = RobotState(base_pose=(0, 0, 1.0, 0, 0, 0, 1))
    remaining = [Skill(id="SK_WALK", params={"target": "workbench"}), Skill(id="SK_PLACE")]
    seq = r.replan(FailureType.F_GRASP, "SK_GRASP", state, remaining=remaining)
    assert [s.id for s in seq.skills] == ["SK_REACH", "SK_GRASP", "SK_WALK", "SK_PLACE"]
    assert seq.skills[1].params.get("retry") is True
    seq = r.replan(FailureType.F_TIMEOUT, "SK_GRASP", state, remaining=remaining)
    assert [s.id for s in seq.skills] == ["SK_STAND"]


def test_replanner_rejects_invalid_llm_output():
    llm = _ScriptedLLM(replan_result=[{"id": "SK_TELEPORT"}])
    state = RobotState(base_pose=(0, 0, 1.0, 0, 0, 0, 1))
    seq = Replanner(llm=llm).replan(FailureType.F_TRAJECTORY, "SK_WALK", state)
    assert [s.id for s in seq.skills] == ["SK_WALK"]


def test_planning_node_tracks_progress_for_replan():
    llm = _ScriptedLLM(decompose_result=_TRANSPORT_PLAN, replan_result=LLMError("down"))
    pn = PlanningNode(config={})
    pn.llm = llm
    pn.decomposer.llm = llm
    pn.replanner.llm = llm
    state = RobotState(base_pose=(0, 0, 1.0, 0, 0, 0, 1))

    pn.on_instruction("把箱子搬到工作台", SceneGraph())
    pn.on_skill_completed("SK_WALK")
    pn.on_skill_completed("SK_REACH")
    new_seq = pn.on_failure(FailureType.F_GRASP, "SK_GRASP", state)

    sent = llm.replan_states[0]
    assert sent["goal"]["target"] == "box"
    assert [s["id"] for s in sent["completed_skills"]] == ["SK_WALK", "SK_REACH"]
    assert [s["id"] for s in sent["remaining_skills"]] == ["SK_WALK", "SK_PLACE"]
    assert [s.id for s in new_seq.skills] == ["SK_REACH", "SK_GRASP", "SK_WALK", "SK_PLACE"]
    assert pn.get_current_sequence() is new_seq
    assert [s.id for s in pn.completed_skills] == ["SK_WALK", "SK_REACH"]


def test_remote_llm_chat_url_handles_v1_suffix():
    assert RemoteLLMClient(endpoint="https://api.deepseek.com/v1")._chat_url() == \
        "https://api.deepseek.com/v1/chat/completions"
    assert RemoteLLMClient(endpoint="http://127.0.0.1:8000/")._chat_url() == \
        "http://127.0.0.1:8000/v1/chat/completions"


def test_remote_llm_without_endpoint_fails_fast():
    c = RemoteLLMClient(endpoint="", max_retries=3)
    try:
        c.decompose({}, {})
        assert False, "should raise"
    except LLMError:
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
