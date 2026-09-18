"""model_training/ 单元测试:rewards + terminations + domain rand。"""
import sys
sys.path.insert(0, ".")
import math
import random

from common.types import RobotState, JointState, JointCommand
from model_training.mdp.rewards import (
    StairClimbReward, GraspReward, TrackingReward, StabilityReward,
)
from model_training.mdp.terminations import (
    FallTermination, CollisionTermination, TimeoutTermination, check_all,
)
from model_training.mdp.events import DomainRandomization


def test_stair_climb_reward():
    r = StairClimbReward()
    s1 = RobotState(base_pose=(0, 0, 1.0, 0, 0, 0, 1))
    s2 = RobotState(base_pose=(1.0, 0, 1.5, 0, 0, 0, 1))
    cmd = JointCommand(effort=[1.0] * 31, position=[0.1] * 31)
    s1.joint = JointState(position=[0.0] * 31)
    score = r(s2, s1, cmd)
    assert score > 0


def test_grasp_reward():
    r = GraspReward()
    s = RobotState()
    assert r(s, grasp_force=10.0, target_force=10.0) == 1.0
    assert r(s, grasp_force=5.0) < 1.0


def test_tracking_reward_zero_error():
    r = TrackingReward()
    s = RobotState(base_pose=(0, 0, 1.0, 0, 0, 0, 1))
    assert r(s, s) == 0.0


def test_stability_reward_ok():
    r = StabilityReward()
    s = RobotState(base_pose=(0, 0, 1.0, 0, 0, 0, 1))
    assert r(s) >= 0


def test_fall_termination_low_height():
    fall = FallTermination(min_height=0.3)
    assert fall(RobotState(base_pose=(0, 0, 0.1, 0, 0, 0, 1))) is True
    assert fall(RobotState(base_pose=(0, 0, 1.0, 0, 0, 0, 1))) is False


def test_collision_termination_torque_spike():
    coll = CollisionTermination(max_torque_nm=200.0)
    assert coll(RobotState(joint=JointState(effort=[300.0] * 31))) is True
    assert coll(RobotState(joint=JointState(effort=[10.0] * 31))) is False


def test_timeout_termination():
    t = TimeoutTermination(max_steps=1000)
    assert t(1000) is True
    assert t(500) is False


def test_check_all_returns_reason():
    fall = FallTermination()
    coll = CollisionTermination(max_torque_nm=200.0)
    t = TimeoutTermination(max_steps=1000)
    # 跌倒触发
    done, reason = check_all([fall, coll, t], RobotState(base_pose=(0, 0, 0.1, 0, 0, 0, 1)), 0)
    assert done is True and reason == "fall"
    # 没异常
    done, reason = check_all([fall, coll, t], RobotState(base_pose=(0, 0, 1.0, 0, 0, 0, 1)), 50)
    assert done is False
    # 超时
    done, reason = check_all([fall, coll, t], RobotState(base_pose=(0, 0, 1.0, 0, 0, 0, 1)), 1001)
    assert done is True and reason == "timeout"


def test_domain_randomization_ranges():
    random.seed(0)
    dr = DomainRandomization()
    samples = [dr.sample() for _ in range(50)]
    for s in samples:
        assert 0.8 <= s["mass"] <= 1.2
        assert 0.5 <= s["friction"] <= 1.5
        assert 0.001 <= s["imu_noise_std"] <= 0.02
