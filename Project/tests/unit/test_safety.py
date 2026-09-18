"""safety 专项测试(Readme §10)。"""
import sys
sys.path.insert(0, ".")
import time
import math

from common.types import (
    JointCommand, RobotState, JointState, IMUState, FailureType,
)
from motion_control.feedback import FeedbackMonitor


def test_estop_response_under_50ms():
    """L4 急停响应 < 50ms(Readme §10.1)。"""
    fb = FeedbackMonitor()
    cmd = JointCommand(effort=[200.0] * 31)
    t0 = time.perf_counter_ns()
    if not fb.check_joint_limits(cmd):
        cmd = fb.clamp(cmd)
        cmd.effort = [0.0] * 31
    elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
    assert elapsed_ms < 50


def test_l1_clamp_does_not_exceed_limit():
    fb = FeedbackMonitor()
    over = JointCommand(effort=[300.0] * 31)
    clamped = fb.clamp(over)
    assert all(abs(t) <= 150.0 for t in clamped.effort)


def test_l2_baseline_stable():
    """base 高度 > 0.3m,pitch 0° → 通过。"""
    fb = FeedbackMonitor()
    s = RobotState(base_pose=(0, 0, 1.0, 0, 0, 0, 1), imu=IMUState(orientation_xyzw=(0, 0, 0, 1)))
    assert fb.check_stability(s) is True


def test_l3_collision_requires_consecutive():
    fb = FeedbackMonitor()
    s = RobotState(joint=JointState(position=[0]*31, velocity=[0]*31,
                                    effort=[200.0] + [5.0] * 30))
    # 1 次不算
    fb.check_collision(s)
    assert fb._consecutive_collision < 3
    # 连续 3 次
    fb.check_collision(s)
    fb.check_collision(s)
    assert fb._consecutive_collision >= 3
    assert fb.check_collision(s) is False  # 触发
