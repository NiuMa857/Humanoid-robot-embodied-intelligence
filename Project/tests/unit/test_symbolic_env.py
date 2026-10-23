"""simulation.symbolic + evaluation 单元测试。"""
import sys
sys.path.insert(0, ".")
import dataclasses
import json
import logging

from common.types import Skill, FailureType
from simulation.symbolic import SymbolicEnv, FailureModel, TASKS, TASKS_BY_ID
from task_planning import PlanningNode, SkillLibrary
from evaluation import run_episode, summarize

logging.disable(logging.WARNING)

_PERFECT = FailureModel(walk=1, walk_slope_per_kg=0, climb=1, climb_per_kg=0, reach=1,
                        grasp=1, grasp_per_kg=0, place=1, assemble=1, turn=1)


def S(id, **params):
    return Skill(id=id, params=params)


def _carry(obj, src, dst_steps, dst):
    return [S("SK_WALK", target=src), S("SK_REACH", target=obj), S("SK_GRASP", target=obj),
            *dst_steps, S("SK_PLACE", target=dst)]


_UP = [S("SK_WALK", target="stairs_bottom"), S("SK_CLIMB", target="platform")]

ORACLE_PLANS = {
    "T01_transport_light": _carry("box_light", "shelf", [S("SK_WALK", target="workbench")], "workbench"),
    "T02_transport_heavy": _carry("box_heavy", "shelf", [S("SK_WALK", target="workbench")], "workbench"),
    "T03_go_upstairs": list(_UP),
    "T04_light_upstairs": _carry("box_light", "shelf", _UP, "platform"),
    "T05_heavy_upstairs": _carry("box_heavy", "shelf", _UP, "platform"),
    "T06_heavy_upstairs_ramp": _carry("box_heavy", "shelf", [S("SK_WALK", target="platform")], "platform"),
    "T07_two_boxes": _carry("box_light", "shelf", [S("SK_WALK", target="workbench")], "workbench")
                     + _carry("box_heavy", "shelf", [S("SK_WALK", target="workbench")], "workbench"),
    "T08_assemble_bolt": [S("SK_WALK", target="parts_bin"), S("SK_REACH", target="bolt"), S("SK_GRASP", target="bolt"),
                          S("SK_WALK", target="assembly_station"), S("SK_ASSEMBLE", target="bolt", into="base_plate")],
    "T09_assemble_two": [S("SK_WALK", target="parts_bin"), S("SK_REACH", target="bracket"), S("SK_GRASP", target="bracket"),
                         S("SK_WALK", target="assembly_station"), S("SK_ASSEMBLE", target="bracket", into="base_plate"),
                         S("SK_WALK", target="parts_bin"), S("SK_REACH", target="bolt"), S("SK_GRASP", target="bolt"),
                         S("SK_WALK", target="assembly_station"), S("SK_ASSEMBLE", target="bolt", into="base_plate")],
    "T11_tool_then_box": _carry("tool", "workbench", _UP, "platform")
                         + [S("SK_CLIMB", target="stairs_bottom")]
                         + _carry("box_light", "shelf", [S("SK_WALK", target="workbench")], "workbench"),
    "T12_pick_lighter_upstairs": _carry("box_light", "shelf", [S("SK_WALK", target="platform")], "platform"),
}


def test_every_feasible_task_has_valid_optimal_plan():
    lib = SkillLibrary()
    for task in TASKS:
        if not task.feasible:
            continue
        plan = ORACLE_PLANS[task.id]
        _, errors = lib.validate_sequence([{"id": s.id, "params": s.params} for s in plan])
        assert errors == [], (task.id, errors)
        env = SymbolicEnv(failure_model=_PERFECT)
        env.reset(task, seed=0)
        assert not env.goal_satisfied(), task.id
        for s in plan:
            r = env.execute(s)
            assert r.success, (task.id, s, r.detail)
        assert env.goal_satisfied(), task.id
        assert len(plan) == task.optimal_len, (task.id, len(plan), task.optimal_len)


def test_observation_is_json_serializable_and_hides_goal():
    env = SymbolicEnv()
    obs = env.reset(TASKS_BY_ID["T06_heavy_upstairs_ramp"], seed=0)
    text = json.dumps(obs, ensure_ascii=False)
    assert "goal" not in obs and "ramp_bottom" in text
    assert ["ramp_bottom", "platform", "slope"] in obs["walk_connections"]
    obs2 = env.reset(TASKS_BY_ID["T05_heavy_upstairs"], seed=0)
    assert "ramp_bottom" not in json.dumps(obs2)


def test_precondition_failures_are_deterministic():
    env = SymbolicEnv(failure_model=_PERFECT)
    env.reset(TASKS_BY_ID["T01_transport_light"], seed=0)
    r = env.execute(S("SK_GRASP", target="box_light"))
    assert not r.success and r.failure == FailureType.F_GRASP and "SK_REACH" in r.detail
    r = env.execute(S("SK_REACH", target="box_light"))            # 不在货架
    assert not r.success and r.failure == FailureType.F_PERCEPTION
    r = env.execute(S("SK_CLIMB", target="platform"))             # 不在楼梯底部
    assert not r.success and r.failure == FailureType.F_TRAJECTORY
    r = env.execute(S("SK_WALK", target="platform"))              # 无坡道,步行不可达
    assert not r.success and "SK_CLIMB" in r.detail
    assert env.execute(S("SK_WALK", target="货架")).success          # 中文地点名
    assert env.execute(S("SK_REACH", target="大货箱")).success       # 中文物体名
    r = env.execute(S("SK_GRASP", target="crate_big"))
    assert not r.success and "payload" in r.detail


def test_fall_on_stairs_drops_object_and_requires_stand():
    env = SymbolicEnv(failure_model=dataclasses.replace(_PERFECT, climb=0.0))
    env.reset(TASKS_BY_ID["T05_heavy_upstairs"], seed=0)
    env.rng.random = lambda: 0.99
    for s in [S("SK_WALK", target="shelf"), S("SK_REACH", target="box_heavy"),
              S("SK_GRASP", target="box_heavy"), S("SK_WALK", target="stairs_bottom")]:
        assert env.execute(s).success
    r = env.execute(S("SK_CLIMB", target="platform"))
    assert not r.success and r.failure == FailureType.F_STABILITY and "dropped box_heavy" in r.detail
    obs = env.observe()
    assert obs["robot"]["standing"] is False and obs["robot"]["holding"] is None
    assert next(o for o in obs["objects"] if o["name"] == "box_heavy")["location"] == "stairs_bottom"
    r = env.execute(S("SK_WALK", target="shelf"))
    assert not r.success and "SK_STAND" in r.detail
    assert env.execute(S("SK_STAND")).success


def test_climb_success_rate_depends_on_payload():
    def rate(obj, n=400):
        ok = 0
        for seed in range(n):
            env = SymbolicEnv()
            env.reset(TASKS_BY_ID["T05_heavy_upstairs"], seed=seed)
            env.state.robot_location = "stairs_bottom"
            if obj:
                env.state.holding = obj
                env.state.objects[obj]["location"] = "robot_hand"
            ok += env.execute(S("SK_CLIMB", target="platform")).success
        return ok / n
    empty, heavy = rate(None), rate("box_heavy")
    assert empty > 0.9 and heavy < 0.75, (empty, heavy)


def test_same_seed_is_reproducible():
    def run(seed):
        env = SymbolicEnv()
        env.reset(TASKS_BY_ID["T05_heavy_upstairs"], seed=seed)
        return [env.execute(s).success for s in ORACLE_PLANS["T05_heavy_upstairs"]]
    assert run(7) == run(7)


def test_benchmark_template_planner_closed_loop():
    env = SymbolicEnv(failure_model=_PERFECT)
    make = lambda: PlanningNode(config={})
    r = run_episode(make, env, TASKS_BY_ID["T01_transport_light"], seed=0, use_goal_hint=True)
    assert r.success and r.replans == 0 and r.termination == "plan_done"
    assert [s["id"] for s in r.initial_plan][0] == "SK_STAND"
    assert r.llm["calls"] == 0

    r = run_episode(make, env, TASKS_BY_ID["T10_infeasible_crate"], seed=0, use_goal_hint=True, max_replans=3)
    assert not r.success and not r.feasible
    assert r.termination == "max_replans" and r.replans == 3

    r = run_episode(make, env, TASKS_BY_ID["T04_light_upstairs"], seed=0, use_goal_hint=True, max_replans=2)
    assert not r.success     # 模板不知道要先走到楼梯底部再攀爬

    s = summarize([r], price_in=2, price_out=8)
    assert s["episodes"] == 1 and "T04_light_upstairs" in s["per_task"]


def test_benchmark_replans_after_stochastic_failure():
    """随机失败 → 兜底重规划沿用参数继续剩余计划 → 最终成功。"""
    env = SymbolicEnv(failure_model=_PERFECT)
    calls = {"n": 0}
    real_execute = env.execute

    def flaky(skill):
        if skill.id == "SK_GRASP" and calls["n"] == 0:
            calls["n"] += 1
            return type(real_execute(Skill(id="SK_STAND")))(False, FailureType.F_GRASP, "grasp slipped", 5.0)
        return real_execute(skill)
    env.execute = flaky
    r = run_episode(lambda: PlanningNode(config={}), env, TASKS_BY_ID["T01_transport_light"], seed=0, use_goal_hint=True)
    assert r.success and r.replans == 1, r.trace
