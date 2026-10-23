"""符号仿真闭环评测入口。

用法(在 Project/ 目录下):
  python -m evaluation.run_benchmark --planner template --seeds 20
  python -m evaluation.run_benchmark --planner llm --seeds 3                 # 读 configs/default.yaml
  python -m evaluation.run_benchmark --planner llm --model deepseek-v4-pro --tasks T05_heavy_upstairs,T06_heavy_upstairs_ramp

planner:
  template  不调 LLM;使用任务的结构化 goal_hint + 预设模板 + 规则兜底重规划(基线)
  llm       自然语言指令 → LLM 理解 / 分解 / 重规划(LLM 失败时自动降级到模板)

结果:每个 episode 一行 JSON 写入 evaluation/results/,终端打印汇总。
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import load_config
from simulation.symbolic import SymbolicEnv, TASKS, TASKS_BY_ID
from task_planning import PlanningNode, SkillLibrary
from evaluation.benchmark import run_episode, summarize


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--planner", choices=["template", "llm"], default="template")
    ap.add_argument("--model", default=None, help="覆盖 planning.llm.model")
    ap.add_argument("--tasks", default="", help="逗号分隔的任务 id,默认全部")
    ap.add_argument("--seeds", type=int, default=5, help="每个任务跑多少个随机种子")
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--max-replans", type=int, default=6)
    ap.add_argument("--price-in", type=float, default=2.0, help="元/百万输入 token(默认 deepseek-flash 高峰价)")
    ap.add_argument("--price-out", type=float, default=8.0, help="元/百万输出 token")
    ap.add_argument("--out", default="", help="结果 JSONL 路径,默认 evaluation/results/<时间>_<planner>.jsonl")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    logging.disable(logging.WARNING)
    tasks = [TASKS_BY_ID[t] for t in args.tasks.split(",") if t] if args.tasks else TASKS

    if args.planner == "llm":
        cfg = load_config("default")
        if args.model:
            cfg["planning"]["llm"]["model"] = args.model
        make_planner = lambda: PlanningNode(config=cfg)
        label = f"llm-{cfg['planning']['llm']['model']}"
    else:
        make_planner = lambda: PlanningNode(config={})
        label = "template"

    durations = {k: v.estimated_duration_s for k, v in SkillLibrary.SKILLS.items()}
    env = SymbolicEnv(durations=durations)

    out = Path(args.out) if args.out else PROJECT_ROOT / "evaluation" / "results" / f"{time.strftime('%Y%m%d_%H%M%S')}_{label}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    results = []
    with out.open("w", encoding="utf-8") as f:
        for task in tasks:
            for seed in range(args.seeds):
                r = run_episode(make_planner, env, task, seed, use_goal_hint=(args.planner == "template"),
                                max_steps=args.max_steps, max_replans=args.max_replans)
                results.append(r)
                f.write(json.dumps({"planner": label, **asdict(r)}, ensure_ascii=False) + "\n")
                if not args.quiet:
                    mark = "OK  " if r.success else ("N/A " if not r.feasible else "FAIL")
                    print(f"{mark} {task.id:28s} seed={seed:<3d} steps={r.steps:<3d} replans={r.replans:<2d} "
                          f"llm_calls={r.llm.get('calls', 0):<3d} {r.termination}", flush=True)

    s = summarize(results, price_in=args.price_in, price_out=args.price_out)
    print(f"\n===== {label} | {len(tasks)} tasks x {args.seeds} seeds =====")
    print(f"成功率(可行任务)      {s['success_rate_feasible']:.1%}")
    print(f"首次计划即成功率      {s['first_plan_success_rate']:.1%}")
    print(f"平均步数 / 重规划次数  {s['avg_steps_feasible']:.1f} / {s['avg_replans_feasible']:.2f}")
    print(f"不可行任务平均步数    {s['infeasible_avg_steps']:.1f}")
    print(f"LLM 调用 / 失败       {s['llm_calls']} / {s['llm_errors']}")
    print(f"tokens 输入 / 输出    {s['prompt_tokens']} / {s['completion_tokens']}  (估算 {s['est_cost_yuan']:.4f} 元)")
    print("\n任务                          成功率   平均步数  平均重规划")
    for tid, d in s["per_task"].items():
        print(f"{tid:28s} {d['success_rate']:6.0%}   {d['avg_steps']:6.1f}   {d['avg_replans']:6.2f}")
    print(f"\n逐 episode 结果: {out}")
    return s


if __name__ == "__main__":
    main()
