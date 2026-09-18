"""motion_control/ 单元测试:FeedbackMonitor L1-L5 + SkillExecutor + ControlNode。"""
import sys
sys.path.insert(0, ".")
import math
import time

from common.types import (
    RobotState, JointState, IMUState, JointCommand,
    Skill, SkillSequence, Trajectory, FailureType,
)
from motion_control import (
    MPCPlanner, WBCController, LowLevelServo,
    SkillExecutor, FeedbackMonitor, ControlNode,
)


def _state_ok(num_dof: int = 31) -> RobotState:
    return RobotState(
        base_pose=(0, 0, 1.0, 0, 0, 0, 1),
        joint=JointState(position=[0.0] * num_dof, velocity=[0.0] * num_dof,
                         effort=[0.0] * num_dof),
        imu=IMUState(orientation_xyzw=(0, 0, 0, 1)),
    )


def test_feedback_l1_limit():
    fb = FeedbackMonitor()
    ok = JointCommand(effort=[10.0] * 31)
    over = JointCommand(effort=[200.0] * 31)
    assert fb.check_joint_limits(ok) is True
    assert fb.check_joint_limits(over) is False
    clamped = fb.clamp(over)
    assert all(abs(t) <= 150.0 for t in clamped.effort)


def test_feedback_l2_tilt_detection():
    fb = FeedbackMonitor()
    # pitch 70° 应触发
    q = (0.0, math.sin(math.radians(35)), 0.0, math.cos(math.radians(35)))
    s_tilt = RobotState(base_pose=(0, 0, 1.0, 0, q[1], 0, q[3]),
                        imu=IMUState(orientation_xyzw=q))
    assert fb.check_stability(s_tilt) is False
    assert fb.check_stability(_state_ok()) is True


def test_feedback_l5_comm_timeout():
    fb = FeedbackMonitor()
    now = int(1e10)
    assert fb.check_comm_timeout(0, now) is True  # 未收到过消息
    assert fb.check_comm_timeout(now - int(5e9), now) is False  # 5s 前超时
    assert fb.check_comm_timeout(now - int(1e9), now) is True   # 1s 前 OK


def test_feedback_l4_estop():
    fb = FeedbackMonitor()
    assert fb.estop_active is False
    fb.emergency_stop()
    assert fb.estop_active is True


def test_skill_executor_state_machine():
    class _DummyMPC:
        def plan(self, skill, state):
            return Trajectory(waypoints=[state.base_pose])

    ex = SkillExecutor(mpc=_DummyMPC())
    state = _state_ok()
    ex.start(Skill(id="SK_WALK"), state)
    assert ex.is_running
    traj = ex.tick(state)
    assert len(traj.waypoints) > 0
    ex.complete(success=True)
    assert not ex.is_running


def test_mpc_wbc_zero_order():
    state = _state_ok()
    mpc = MPCPlanner()
    traj = mpc.plan(Skill(id="SK_WALK"), state)
    assert len(traj.waypoints) > 0

    wbc = WBCController(config={"robot": {"dof": 31}})
    cmd = wbc.control(traj, state)
    assert len(cmd.position) == 31
    assert cmd.kp[0] > 0


def test_control_node_step_runs():
    ctrl = ControlNode(config={"robot": {"dof": 31}, "mpc_freq_hz": 100,
                              "wbc_freq_hz": 1000, "servo_freq_hz": 2000})
    ctrl.on_skill_sequence(SkillSequence(skills=[Skill(id="SK_STAND")]))
    status = ctrl.step(_state_ok())
    assert status is not None
    ctrl.shutdown()
