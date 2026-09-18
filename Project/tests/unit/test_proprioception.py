"""proprioception/ 单元测试:state estimator + IMU + joint + node。"""
import sys
sys.path.insert(0, ".")
import math

from common.types import IMUState, JointState, RobotState, JointCommand
from proprioception import (
    StateEstimator, IMUReader, JointStateReader, ProprioceptionNode,
    quat_normalize, quat_rotate, quat_to_euler_zyx,
)


def test_quat_normalize_identity():
    assert quat_normalize((0, 0, 0, 1)) == (0, 0, 0, 1)
    n = quat_normalize((0, 0, 0, 2))
    assert abs(n[3] - 1.0) < 1e-9


def test_quat_rotate_zero():
    """零旋转应该不变。"""
    v = quat_rotate((0, 0, 0, 1), (1.5, 2.0, 3.0))
    assert all(abs(v[i] - (1.5, 2.0, 3.0)[i]) < 1e-9 for i in range(3))


def test_quat_rotate_180_around_z():
    """绕 Z 180° 旋转 (x,y,z) → (-x,-y,z)。"""
    v = quat_rotate((0, 0, 1, 0), (1, 0, 0))
    assert abs(v[0] - (-1)) < 1e-9 and abs(v[1]) < 1e-9 and abs(v[2]) < 1e-9


def test_state_estimator_fallback():
    """无 KinematicsBackend 时 fallback 到零阶。"""
    est = StateEstimator(config={"robot": {"dof": 31}})
    imu = IMUState(orientation_xyzw=(0, 0, 0, 1))
    joint = JointState(position=[0.0] * 31, velocity=[0.0] * 31, effort=[0.0] * 31)
    state = est.estimate(imu, joint, timestamp_ns=1000000)
    assert state.imu.orientation_xyzw == (0, 0, 0, 1)
    assert state.base_pose[2] == 1.0  # standing height


def test_imu_reader_mock():
    imu = IMUReader(mode="mock")
    s = imu.read()
    assert s.angular_velocity is not None
    imu.calibrate()  # 不报错


def test_joint_reader_l1_limit():
    """L1 限位:任何模式都强制。"""
    jr = JointStateReader(num_dof=31, mode="mock")
    # 200 N·m 超 150 限位
    assert jr.send_command(JointCommand(effort=[200.0] * 31)) is False
    # 10 N·m 合规
    assert jr.send_command(JointCommand(effort=[10.0] * 31)) is True


def test_proprioception_node_e2e():
    node = ProprioceptionNode(config={"mode": "mock", "robot": {"dof": 31}})
    node.calibrate()
    for _ in range(5):
        s = node.step()
        assert s is not None
    node.shutdown()
