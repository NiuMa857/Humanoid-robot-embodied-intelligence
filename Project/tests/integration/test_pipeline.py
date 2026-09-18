"""端到端集成测试(无 ROS 2 / 无 Isaac)。"""
import sys
sys.path.insert(0, ".")
import time

from common.types import (
    RobotState, JointState, IMUState, JointCommand, SceneGraph,
    Skill, SkillSequence,
)
from proprioception import ProprioceptionNode
from environment_perception import PerceptionNode
from motion_control import ControlNode
from task_planning import PlanningNode


def test_full_pipeline_no_real_world():
    """全链路 mock:感知 → 规划 → 控制,验证流程通畅。"""
    # 1. 感知
    perception = PerceptionNode(config={"mode": "mock"})
    sg = perception.step()
    assert sg is not None
    perception.shutdown()

    # 2. 规划(无 LLM endpoint,触发 fallback)
    planner = PlanningNode(config={"llm": {"mode": "remote", "endpoint": "", "api_key": ""}})
    seq = planner.on_instruction("把箱子搬到工作台", sg)
    assert len(seq.skills) > 0
    planner.shutdown()

    # 3. 控制
    ctrl = ControlNode(config={"robot": {"dof": 31}})
    ctrl.on_skill_sequence(seq)
    # 跑 5 步
    for _ in range(5):
        state = RobotState(base_pose=(0, 0, 1.0, 0, 0, 0, 1),
                          joint=JointState(position=[0]*31, velocity=[0]*31))
        status = ctrl.step(state)
        assert status is not None
    ctrl.shutdown()


def test_perception_proprio_sync():
    """感知 + 本体感知同时跑(都 mock)。"""
    pn = ProprioceptionNode(config={"mode": "mock", "robot": {"dof": 31}})
    en = PerceptionNode(config={"mode": "mock"})
    for _ in range(3):
        rs = pn.step()
        sg = en.step()
        assert rs is not None and sg is not None
    pn.shutdown()
    en.shutdown()
