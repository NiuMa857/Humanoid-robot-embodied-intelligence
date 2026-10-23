"""认知决策层闭环评测:PlanningNode × SkillEnv。

一个 episode:
  reset → on_instruction → 循环 { 执行下一个技能;成功 → on_skill_completed;
                                 失败 → on_failure(重规划) } → 计划执行完 / 超出步数或重规划上限
  成功 = 计划执行完后 env.goal_satisfied()(ground truth,规划器不可见)。
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field, asdict
from typing import Callable

from common.types import RobotState, to_dict
from simulation.skill_env import SkillEnv
from task_planning import PlanningNode

_USAGE_KEYS = ("calls", "errors", "prompt_tokens", "completion_tokens", "latency_s")


@dataclass
class EpisodeResult:
    task_id: str
    seed: int
    feasible: bool
    success: bool
    termination: str                 # plan_done | max_steps | max_replans
    steps: int
    replans: int
    sim_time_s: float
    wall_time_s: float
    initial_plan: list[dict]
    failures: list[dict] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    llm: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


def _usage(pn: PlanningNode) -> dict:
    u = getattr(pn.llm, "usage", None) or {}
    return {k: u.get(k, 0) for k in _USAGE_KEYS}


def run_episode(
    make_planner: Callable[[], PlanningNode],
    env: SkillEnv,
    task,
    seed: int,
    use_goal_hint: bool = False,
    max_steps: int = 40,
    max_replans: int = 6,
) -> EpisodeResult:
    t0 = time.time()
    pn = make_planner()
    obs = env.reset(task, seed=seed)
    usage0 = _usage(pn)
    seq = pn.on_instruction(task.instruction, obs, goal=task.goal_hint if use_goal_hint else None)
    initial_plan = [to_dict(s) for s in seq.skills]

    steps = replans = 0
    sim_time = 0.0
    termination = "plan_done"
    trace, failures = [], []
    while (skill := pn.next_skill) is not None:
        if steps >= max_steps:
            termination = "max_steps"
            break
        res = env.execute(skill)
        steps += 1
        sim_time += res.duration_s
        trace.append({"skill": to_dict(skill), "success": res.success, "failure": res.failure, "detail": res.detail})
        if res.success:
            pn.on_skill_completed(skill.id)
            continue
        failures.append({"step": steps, "skill": skill.id, "failure": res.failure, "detail": res.detail})
        if replans >= max_replans:
            termination = "max_replans"
            break
        replans += 1
        fallen = not env.observe()["robot"]["standing"]
        state = RobotState(base_pose=(0, 0, 0.2 if fallen else 1.0, 0, 0, 0, 1))
        pn.on_failure(res.failure, skill.id, state, scene=env.observe(), failure_detail=res.detail)

    usage1 = _usage(pn)
    pn.shutdown()
    return EpisodeResult(
        task_id=task.id, seed=seed, feasible=task.feasible,
        success=env.goal_satisfied(), termination=termination,
        steps=steps, replans=replans, sim_time_s=sim_time, wall_time_s=time.time() - t0,
        initial_plan=initial_plan, failures=failures, trace=trace,
        llm={k: usage1[k] - usage0[k] for k in _USAGE_KEYS}, tags=list(task.tags),
    )


def summarize(results: list[EpisodeResult], price_in: float = 0.0, price_out: float = 0.0) -> dict:
    """汇总指标。price_* 为 元 / 百万 tokens(用于估算费用)。"""
    feas = [r for r in results if r.feasible]
    infeas = [r for r in results if not r.feasible]

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    per_task: dict[str, dict] = {}
    for r in results:
        d = per_task.setdefault(r.task_id, {"n": 0, "success": 0, "steps": [], "replans": []})
        d["n"] += 1
        d["success"] += int(r.success)
        d["steps"].append(r.steps)
        d["replans"].append(r.replans)
    tin = sum(r.llm.get("prompt_tokens", 0) for r in results)
    tout = sum(r.llm.get("completion_tokens", 0) for r in results)
    return {
        "episodes": len(results),
        "success_rate_feasible": mean([r.success for r in feas]),
        "first_plan_success_rate": mean([r.success and r.replans == 0 for r in feas]),
        "avg_steps_feasible": mean([r.steps for r in feas]),
        "avg_replans_feasible": mean([r.replans for r in feas]),
        "infeasible_avg_steps": mean([r.steps for r in infeas]),
        "llm_calls": sum(r.llm.get("calls", 0) for r in results),
        "llm_errors": sum(r.llm.get("errors", 0) for r in results),
        "prompt_tokens": tin,
        "completion_tokens": tout,
        "est_cost_yuan": (tin * price_in + tout * price_out) / 1e6,
        "per_task": {
            k: {"success_rate": v["success"] / v["n"], "avg_steps": mean(v["steps"]),
                "avg_replans": mean(v["replans"]), "n": v["n"]}
            for k, v in per_task.items()
        },
    }


__all__ = ["EpisodeResult", "run_episode", "summarize", "asdict"]
