"""符号仿真的长时程任务集(工业场景:搬运 / 上楼 / 装配)。

所有任务共用一张工厂地图(可选坡道),按难度覆盖:
  - 短任务(5 步左右)→ 长任务(10+ 步,多物体 / 多阶段)
  - 本体相关风险:手持重物上楼梯更易失稳;有坡道时可绕行
  - 属性推理:按重量 / 描述选物体
  - 不可行任务:超出负载,考察规划器能否及时放弃

`goal` 是评测用 ground truth(规划器不可见);`goal_hint` 是给"结构化指令"基线用的 TaskGoal
(单目标任务才有,多目标任务为 None)。
"""
from __future__ import annotations
from dataclasses import dataclass, field

from common.types import TaskGoal

LOCATIONS = {
    "start": "待命区",
    "shelf": "货架",
    "workbench": "工作台",
    "stairs_bottom": "楼梯底部",
    "platform": "二层平台",
    "ramp_bottom": "坡道入口",
    "parts_bin": "零件箱",
    "assembly_station": "装配工位",
}

WALK_EDGES = [
    ("start", "shelf"), ("start", "workbench"), ("shelf", "workbench"),
    ("start", "stairs_bottom"), ("workbench", "parts_bin"),
    ("workbench", "assembly_station"), ("parts_bin", "assembly_station"),
]
RAMP_EDGES = [("start", "ramp_bottom"), ("ramp_bottom", "platform", "slope")]
CLIMB_EDGES = [("stairs_bottom", "platform")]

OBJECTS = {
    "box_light": {"label": "轻箱子", "location": "shelf", "weight_kg": 1.0},
    "box_heavy": {"label": "重箱子", "location": "shelf", "weight_kg": 2.5},
    "crate_big": {"label": "大货箱", "location": "shelf", "weight_kg": 8.0},
    "tool": {"label": "扳手", "location": "workbench", "weight_kg": 0.5},
    "bolt": {"label": "螺栓", "location": "parts_bin", "weight_kg": 0.1},
    "bracket": {"label": "支架", "location": "parts_bin", "weight_kg": 0.3},
    "base_plate": {"label": "底座", "location": "assembly_station", "weight_kg": 5.0},
}


@dataclass
class SymbolicTask:
    id: str
    instruction: str
    goal: dict                       # {"objects": {name: loc}, "assembled": [[part, base]], "robot_location": loc}
    goal_hint: TaskGoal | None = None
    ramp: bool = False
    feasible: bool = True
    optimal_len: int = 0             # 参考最短技能数(无失败时)
    tags: list[str] = field(default_factory=list)
    robot_location: str = "start"
    locations: dict[str, str] = field(default_factory=lambda: dict(LOCATIONS))
    walk_edges: list[tuple] = field(default_factory=list)
    climb_edges: list[tuple] = field(default_factory=lambda: list(CLIMB_EDGES))
    objects: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if not self.walk_edges:
            self.walk_edges = list(WALK_EDGES) + (list(RAMP_EDGES) if self.ramp else [])
        if not self.ramp:
            self.locations.pop("ramp_bottom", None)
        if not self.objects:
            self.objects = [{"name": n, **o} for n, o in OBJECTS.items()]


def _t(id, instruction, goal, hint=None, **kw) -> SymbolicTask:
    return SymbolicTask(id=id, instruction=instruction, goal=goal, goal_hint=hint, **kw)


TASKS: list[SymbolicTask] = [
    _t("T01_transport_light", "把货架上的轻箱子搬到工作台上",
       {"objects": {"box_light": "workbench"}},
       TaskGoal("transport", "box_light", "workbench"), optimal_len=5, tags=["short"]),
    _t("T02_transport_heavy", "把货架上的重箱子搬到工作台上",
       {"objects": {"box_heavy": "workbench"}},
       TaskGoal("transport", "box_heavy", "workbench"), optimal_len=5, tags=["short"]),
    _t("T03_go_upstairs", "走到二层平台上去",
       {"robot_location": "platform"},
       TaskGoal("climb", "", "platform"), optimal_len=2, tags=["short", "stairs"]),
    _t("T04_light_upstairs", "把货架上的轻箱子送到二层平台",
       {"objects": {"box_light": "platform"}},
       TaskGoal("transport", "box_light", "platform"), optimal_len=6, tags=["stairs", "body"]),
    _t("T05_heavy_upstairs", "把货架上的重箱子送到二层平台",
       {"objects": {"box_heavy": "platform"}},
       TaskGoal("transport", "box_heavy", "platform"), optimal_len=6, tags=["stairs", "body"]),
    _t("T06_heavy_upstairs_ramp", "把货架上的重箱子送到二层平台",
       {"objects": {"box_heavy": "platform"}},
       TaskGoal("transport", "box_heavy", "platform"), ramp=True, optimal_len=5, tags=["stairs", "body", "ramp"]),
    _t("T07_two_boxes", "把货架上的两个箱子(轻箱子和重箱子)都搬到工作台上",
       {"objects": {"box_light": "workbench", "box_heavy": "workbench"}},
       optimal_len=10, tags=["long", "multi"]),
    _t("T08_assemble_bolt", "从零件箱取一个螺栓,装到装配工位的底座上",
       {"assembled": [["bolt", "base_plate"]]},
       TaskGoal("assembly", "bolt", "base_plate"), optimal_len=5, tags=["assembly"]),
    _t("T09_assemble_two", "把支架和螺栓都装到装配工位的底座上",
       {"assembled": [["bracket", "base_plate"], ["bolt", "base_plate"]]},
       optimal_len=10, tags=["long", "multi", "assembly"]),
    _t("T10_infeasible_crate", "把货架上的大货箱搬到工作台上",
       {"objects": {"crate_big": "workbench"}},
       TaskGoal("transport", "crate_big", "workbench"), feasible=False, tags=["infeasible"]),
    _t("T11_tool_then_box", "先把工作台上的扳手送到二层平台,再把货架上的轻箱子搬到工作台上",
       {"objects": {"tool": "platform", "box_light": "workbench"}},
       optimal_len=12, tags=["long", "multi", "stairs"]),
    _t("T12_pick_lighter_upstairs", "货架上有两个箱子,把其中比较轻的那个送到二层平台",
       {"objects": {"box_light": "platform"}},
       ramp=True, optimal_len=5, tags=["reasoning", "ramp"]),
]

TASKS_BY_ID = {t.id: t for t in TASKS}

__all__ = ["SymbolicTask", "TASKS", "TASKS_BY_ID", "LOCATIONS", "OBJECTS"]
