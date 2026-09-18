"""Batch 3 smoke test - task_planning + motion_control."""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')
import logging
logging.disable(logging.CRITICAL)

import math
from common.types import (
    TaskGoal, SceneGraph, Skill, SkillSequence, RobotState, JointState, IMUState,
    JointCommand, Trajectory, FailureType,
)
from task_planning import (
    TaskDecomposer, Replanner, SkillLibrary,
    RemoteLLMClient, LocalLLMClient, make_llm_client, make_vlm_client,
    LLMError, PlanningNode,
)
from motion_control import (
    MPCPlanner, WBCController, LowLevelServo,
    SkillExecutor, FeedbackMonitor, ControlNode,
)


# ===== helper: FailingLLM for fallback testing =====
class _FailingLLM:
    """总是抛 LLMError,用于测试 fallback 路径"""
    def understand_instruction(self, text):
        raise LLMError("mock fail")
    def decompose(self, goal, scene):
        raise LLMError("mock fail")
    def replan(self, failure, state):
        raise LLMError("mock fail")
    def close(self):
        pass


# 1. SkillLibrary
lib = SkillLibrary()
assert "SK_WALK" in lib.list_ids()
spec = lib.get("SK_GRASP")
assert spec.requires_ee
assert spec.estimated_duration_s > 0
print('SkillLibrary OK')

# 2. SkillLibrary.check_feasibility
state_ok = RobotState(base_pose=(0, 0, 1.0, 0, 0, 0, 1), joint=JointState(position=[0]*31, velocity=[0]*31))
state_fall = RobotState(base_pose=(0, 0, 0.1, 0, 0, 0, 1))
ok, reason = lib.check_feasibility(Skill(id="SK_WALK"), state_ok)
assert ok, f"should be ok, got {reason}"
ok, reason = lib.check_feasibility(Skill(id="SK_WALK"), state_fall)
assert not ok
print('SkillLibrary check_feasibility OK')

# 3. TaskDecomposer fallback
decomposer = TaskDecomposer(llm=_FailingLLM())
seq = decomposer.decompose(
    TaskGoal(type="transport", target="box", location="warehouse"),
    SceneGraph(),
)
assert len(seq.skills) > 0
assert seq.skills[0].id == "SK_STAND"
print('TaskDecomposer fallback OK, skills:', [s.id for s in seq.skills])

# 4. Replanner fallback
replanner = Replanner(llm=_FailingLLM())
for ft in [FailureType.F_STABILITY, FailureType.F_GRASP, FailureType.F_COLLISION, FailureType.F_TIMEOUT]:
    seq = replanner.replan(ft, "SK_WALK", state_ok)
    assert len(seq.skills) > 0
print('Replanner fallback OK for all 4 failure types')

# 5. make_llm_client / make_vlm_client factory
c1 = make_llm_client({"llm": {"mode": "remote", "endpoint": "https://x", "api_key": "k"}})
assert isinstance(c1, RemoteLLMClient)
c2 = make_llm_client({"llm": {"mode": "local", "endpoint": "http://localhost:8000"}})
assert isinstance(c2, LocalLLMClient)
v1 = make_vlm_client({"vlm": {"mode": "local"}})
assert v1.endpoint == "http://127.0.0.1:8001"
print('LLM/VLM factory OK')

# 6. FeedbackMonitor L1-L5
fb = FeedbackMonitor(config={"enabled_levels": ["L1", "L2", "L3", "L4", "L5"]})
cmd_ok = JointCommand(effort=[10.0]*31)
cmd_over = JointCommand(effort=[200.0]*31)
assert fb.check_joint_limits(cmd_ok) is True
assert fb.check_joint_limits(cmd_over) is False
clamped = fb.clamp(cmd_over)
assert all(abs(t) <= 150.0 for t in clamped.effort)
print('L1 OK')

assert fb.check_stability(state_ok) is True
# pitch 70° 绕 Y 轴(浮点精度,60° 算出来是 59.997°)
import math
q_pitch_70 = (0.0, math.sin(math.radians(35)), 0.0, math.cos(math.radians(35)))
state_p70 = RobotState(base_pose=(0, 0, 1.0, 0, q_pitch_70[1], 0, q_pitch_70[3]), imu=IMUState(orientation_xyzw=q_pitch_70))
assert fb.check_stability(state_p70) is False
print('L2 OK')

state_coll = RobotState(joint=JointState(position=[0]*31, velocity=[0]*31, effort=[200, 5, 5, 5, 5, 5]+[5]*25))
for _ in range(3):
    fb.check_collision(state_coll)
assert fb._consecutive_collision >= 3
print('L3 OK')

fb.emergency_stop()
assert fb.estop_active is True
print('L4 OK')

assert fb.check_comm_timeout(0, int(1e9)) is True  # last_msg=0 表示"还没收到",跳过检查
now_ns = int(1e10)
assert fb.check_comm_timeout(now_ns - int(5e9), now_ns) is False  # 5s 前有过消息,超时
assert fb.check_comm_timeout(now_ns - int(1e9), now_ns) is True   # 1s 前,OK
print('L5 OK')

# 7. SkillExecutor
class _DummyMPC:
    def plan(self, skill, state):
        from common.types import Trajectory
        return Trajectory(waypoints=[state.base_pose])

executor = SkillExecutor(mpc=_DummyMPC())
executor.start(Skill(id="SK_WALK"), state_ok)
assert executor.is_running
traj = executor.tick(state_ok)
assert len(traj.waypoints) > 0
executor.complete(success=True)
assert not executor.is_running
print('SkillExecutor OK')

# 8. ControlNode
ctrl = ControlNode(config={"robot": {"dof": 31}, "mpc_freq_hz": 100, "wbc_freq_hz": 1000, "servo_freq_hz": 2000})
ctrl.on_skill_sequence(SkillSequence(skills=[Skill(id="SK_STAND")]))
status = ctrl.step(state_ok)
assert status is not None
ctrl.shutdown()
print('ControlNode OK')

# 9. WBC zero-order
wbc = WBCController(config={"robot": {"dof": 31}})
cmd = wbc.control(Trajectory(waypoints=[state_ok.base_pose]), state_ok)
assert len(cmd.position) == 31
assert cmd.kp[0] > 0
print('WBC zero-order OK')

# 10. MPC zero-order
mpc = MPCPlanner()
traj = mpc.plan(Skill(id="SK_WALK"), state_ok)
assert len(traj.waypoints) > 0
print('MPC zero-order OK')

# 11. PlanningNode
pn = PlanningNode(config={"llm": {"mode": "remote", "endpoint": "", "api_key": ""}})
seq = pn.on_instruction("把箱子搬到工作台", SceneGraph())
assert seq is not None
pn.shutdown()
print('PlanningNode OK')

print('ALL BATCH-3 TESTS PASSED')
