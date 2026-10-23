"""符号仿真环境:离散世界状态 + 技能前置条件 / 效果 + 与本体状态相关的随机失败。

用途:在没有 Oli URDF / 训练好的底层技能之前,对认知决策层(任务分解 / 重规划)做大批量、
可复现的闭环评测。一轮 episode 毫秒级。

建模要点:
  - 世界:命名地点 + 步行连接(可含坡道)+ 攀爬连接(楼梯);物体有所在地点和重量。
  - 前置条件不满足 → 确定性失败(例如未 SK_REACH 就 SK_GRASP、手上有物体还去抓、超出单臂负载)。
  - 满足前置条件时按 FailureModel 随机失败,成功率依赖本体状态(例如手持重物攀爬更容易失稳)。
  - 失败有副作用:攀爬失稳 → 摔倒(standing=False)且手中物体掉落在原地。

FailureModel 中的数值是占位假设,不是 Oli 实测数据;拿到底层技能的实测成功率后应替换。
"""
from __future__ import annotations
import random
from collections import deque
from dataclasses import dataclass, field

from common.types import Skill, FailureType
from simulation.skill_env import SkillEnv, SkillResult
from .tasks import SymbolicTask

PAYLOAD_LIMIT_KG = 3.0   # Readme 附录 A:单臂最大负载 3 kg


@dataclass
class FailureModel:
    """各技能成功率 = base - 惩罚项(按手中物体重量 w_kg),再裁剪到 [0.05, 1]。"""
    walk: float = 0.98
    walk_slope_per_kg: float = 0.02        # 经过坡道时,每 kg 手持重量的成功率下降
    climb: float = 0.95
    climb_per_kg: float = 0.12             # 攀爬时每 kg 手持重量的成功率下降(3 kg → 0.59)
    reach: float = 0.97
    grasp: float = 0.92
    grasp_per_kg: float = 0.04
    place: float = 0.97
    assemble: float = 0.80
    turn: float = 0.99

    @staticmethod
    def clip(p: float) -> float:
        return max(0.05, min(1.0, p))


@dataclass
class WorldState:
    robot_location: str
    standing: bool = True
    holding: str | None = None
    reached: str | None = None
    objects: dict[str, dict] = field(default_factory=dict)   # name -> {label, location, weight_kg}
    assembled: set[tuple[str, str]] = field(default_factory=set)


class SymbolicEnv(SkillEnv):
    """符号仿真环境(实现 SkillEnv)。"""

    def __init__(self, failure_model: FailureModel | None = None, durations: dict[str, float] | None = None):
        self.fm = failure_model or FailureModel()
        self.durations = durations or {}
        self.task: SymbolicTask | None = None
        self.state: WorldState | None = None
        self.rng = random.Random(0)

    # ---------------- SkillEnv ----------------

    def reset(self, task: SymbolicTask, seed: int | None = None) -> dict:
        self.task = task
        self.rng = random.Random(seed)
        self.state = WorldState(
            robot_location=task.robot_location,
            objects={o["name"]: dict(o) for o in task.objects},
        )
        return self.observe()

    def observe(self) -> dict:
        t, s = self.task, self.state
        return {
            "robot": {"location": s.robot_location, "holding": s.holding, "standing": s.standing,
                      "payload_limit_kg": PAYLOAD_LIMIT_KG},
            "objects": [
                {"name": n, "label": o["label"], "location": "robot_hand" if s.holding == n else o["location"],
                 "weight_kg": o["weight_kg"],
                 **({"assembled_into": next(b for p, b in s.assembled if p == n)}
                    if any(p == n for p, _ in s.assembled) else {})}
                for n, o in s.objects.items()
            ],
            "locations": [{"name": n, "label": l} for n, l in t.locations.items()],
            "walk_connections": [list(e[:2]) + (["slope"] if len(e) > 2 and e[2] == "slope" else []) for e in t.walk_edges],
            "climb_connections": [list(e) for e in t.climb_edges],
        }

    def goal_satisfied(self) -> bool:
        s, g = self.state, self.task.goal
        if "robot_location" in g and s.robot_location != g["robot_location"]:
            return False
        for name, loc in g.get("objects", {}).items():
            if s.holding == name or s.objects[name]["location"] != loc:
                return False
        return all(tuple(pair) in s.assembled for pair in g.get("assembled", []))

    def execute(self, skill: Skill) -> SkillResult:
        handler = getattr(self, f"_do_{skill.id.lower().removeprefix('sk_')}", None)
        if handler is None:
            return self._fail(skill, FailureType.F_TRAJECTORY, f"unknown skill {skill.id}")
        if skill.id != "SK_STAND" and not self.state.standing:
            return self._fail(skill, FailureType.F_STABILITY, "robot is not standing (fallen); SK_STAND first")
        return handler(skill)

    # ---------------- helpers ----------------

    def _ok(self, skill: Skill, extra_s: float = 0.0) -> SkillResult:
        return SkillResult(True, duration_s=self._duration(skill) + extra_s)

    def _fail(self, skill: Skill, failure: str, detail: str) -> SkillResult:
        return SkillResult(False, failure=failure, detail=detail, duration_s=self._duration(skill))

    def _duration(self, skill: Skill) -> float:
        return self.durations.get(skill.id, 5.0)

    def _roll(self, p: float) -> bool:
        return self.rng.random() < FailureModel.clip(p)

    def _held_weight(self) -> float:
        s = self.state
        return s.objects[s.holding]["weight_kg"] if s.holding else 0.0

    def _resolve_location(self, name: str) -> str | None:
        """地点名 / 地点中文名 / 物体名(→ 物体所在地点)。"""
        t, s = self.task, self.state
        if name in t.locations:
            return name
        for n, label in t.locations.items():
            if name == label:
                return n
        obj = self._resolve_object(name)
        if obj is not None and s.holding != obj:
            return s.objects[obj]["location"]
        return None

    def _resolve_object(self, name: str) -> str | None:
        s = self.state
        if name in s.objects:
            return name
        return next((n for n, o in s.objects.items() if o["label"] == name), None)

    def _walk_path(self, src: str, dst: str) -> list[tuple[str, str, str]] | None:
        """步行图 BFS,返回边列表 (a, b, kind);不可达返回 None。"""
        adj: dict[str, list[tuple[str, str]]] = {}
        for e in self.task.walk_edges:
            kind = e[2] if len(e) > 2 else "flat"
            adj.setdefault(e[0], []).append((e[1], kind))
            adj.setdefault(e[1], []).append((e[0], kind))
        prev: dict[str, tuple[str, str] | None] = {src: None}
        q = deque([src])
        while q:
            u = q.popleft()
            if u == dst:
                break
            for v, kind in adj.get(u, []):
                if v not in prev:
                    prev[v] = (u, kind)
                    q.append(v)
        if dst not in prev:
            return None
        path, v = [], dst
        while prev[v] is not None:
            u, kind = prev[v]
            path.append((u, v, kind))
            v = u
        return path[::-1]

    # ---------------- skills ----------------

    def _do_stand(self, skill: Skill) -> SkillResult:
        self.state.standing = True
        return self._ok(skill)

    def _do_turn(self, skill: Skill) -> SkillResult:
        self.state.reached = None
        return self._ok(skill) if self._roll(self.fm.turn) else self._fail(skill, FailureType.F_STABILITY, "lost balance while turning")

    def _do_walk(self, skill: Skill) -> SkillResult:
        s = self.state
        dst = self._resolve_location(str(skill.params.get("target", "")))
        if dst is None:
            return self._fail(skill, FailureType.F_TRAJECTORY, f"unknown destination {skill.params.get('target')!r}")
        path = self._walk_path(s.robot_location, dst)
        if path is None:
            return self._fail(skill, FailureType.F_TRAJECTORY,
                              f"{dst} is not reachable by walking from {s.robot_location} (stairs require SK_CLIMB)")
        s.reached = None
        for a, b, kind in path:
            p = self.fm.walk - (self.fm.walk_slope_per_kg * self._held_weight() if kind == "slope" else 0.0)
            if not self._roll(p):
                s.robot_location = a
                return self._fail(skill, FailureType.F_COLLISION, f"blocked/stumbled between {a} and {b}")
            s.robot_location = b
        return self._ok(skill, extra_s=10.0 * sum(k == "slope" for _, _, k in path))

    def _do_climb(self, skill: Skill) -> SkillResult:
        s = self.state
        dst = self._resolve_location(str(skill.params.get("target", "")))
        edge_ok = any({s.robot_location, dst} == set(e) for e in self.task.climb_edges)
        if dst is None or not edge_ok:
            return self._fail(skill, FailureType.F_TRAJECTORY,
                              f"no stairs between {s.robot_location} and {skill.params.get('target')!r}")
        s.reached = None
        w = self._held_weight()
        if not self._roll(self.fm.climb - self.fm.climb_per_kg * w):
            s.standing = False
            detail = "lost balance on stairs and fell"
            if s.holding:
                s.objects[s.holding]["location"] = s.robot_location
                detail += f"; dropped {s.holding} at {s.robot_location}"
                s.holding = None
            return self._fail(skill, FailureType.F_STABILITY, detail)
        s.robot_location = dst
        return self._ok(skill)

    def _do_reach(self, skill: Skill) -> SkillResult:
        s = self.state
        obj = self._resolve_object(str(skill.params.get("target", "")))
        if obj is None or s.objects[obj]["location"] != s.robot_location or s.holding == obj:
            return self._fail(skill, FailureType.F_PERCEPTION,
                              f"target {skill.params.get('target')!r} not found at {s.robot_location}")
        if not self._roll(self.fm.reach):
            return self._fail(skill, FailureType.F_TRAJECTORY, "arm failed to reach target")
        s.reached = obj
        return self._ok(skill)

    def _do_grasp(self, skill: Skill) -> SkillResult:
        s = self.state
        obj = self._resolve_object(str(skill.params.get("target", "")))
        if s.holding is not None:
            return self._fail(skill, FailureType.F_GRASP, f"hand is not empty (holding {s.holding})")
        if obj is None or s.reached != obj:
            return self._fail(skill, FailureType.F_GRASP, f"must SK_REACH {skill.params.get('target')!r} before grasping")
        w = s.objects[obj]["weight_kg"]
        if w > PAYLOAD_LIMIT_KG:
            return self._fail(skill, FailureType.F_GRASP, f"{obj} weighs {w} kg > payload limit {PAYLOAD_LIMIT_KG} kg")
        if not self._roll(self.fm.grasp - self.fm.grasp_per_kg * w):
            return self._fail(skill, FailureType.F_GRASP, f"grasp slipped on {obj}")
        s.holding, s.reached = obj, None
        s.objects[obj]["location"] = "robot_hand"
        return self._ok(skill)

    def _do_place(self, skill: Skill) -> SkillResult:
        s = self.state
        if s.holding is None:
            return self._fail(skill, FailureType.F_GRASP, "nothing in hand to place")
        target = skill.params.get("target")
        loc = self._resolve_location(str(target)) if target else s.robot_location
        if loc != s.robot_location:
            return self._fail(skill, FailureType.F_TRAJECTORY, f"robot is at {s.robot_location}, not at {target!r}")
        if not self._roll(self.fm.place):
            return self._fail(skill, FailureType.F_GRASP, "object slipped while placing")
        s.objects[s.holding]["location"] = s.robot_location
        s.holding = None
        return self._ok(skill)

    def _do_assemble(self, skill: Skill) -> SkillResult:
        s = self.state
        part = self._resolve_object(str(skill.params.get("target", "")))
        base = self._resolve_object(str(skill.params.get("into", "")))
        if part is None or s.holding != part:
            return self._fail(skill, FailureType.F_GRASP, f"must be holding part {skill.params.get('target')!r}")
        if base is None or s.objects[base]["location"] != s.robot_location:
            return self._fail(skill, FailureType.F_PERCEPTION,
                              f"assembly base {skill.params.get('into')!r} not at {s.robot_location}")
        if not self._roll(self.fm.assemble):
            return self._fail(skill, FailureType.F_TRAJECTORY, "insertion misaligned; part still in hand")
        s.objects[part]["location"] = s.robot_location
        s.holding = None
        s.assembled.add((part, base))
        return self._ok(skill)


__all__ = ["SymbolicEnv", "WorldState", "FailureModel", "PAYLOAD_LIMIT_KG"]
