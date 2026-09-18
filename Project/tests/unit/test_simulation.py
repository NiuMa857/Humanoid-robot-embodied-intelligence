"""simulation/ 单元测试:MockSimulator + SimInterface。"""
import sys
sys.path.insert(0, ".")

from common.types import JointCommand, RobotState
from simulation import SimInterface, MockSimulator


def test_mock_simulator_basic():
    sim = MockSimulator(num_dof=31)
    sim.load_robot("dummy.urdf")
    sim.load_scene("stair_30")
    state = sim.reset()
    assert state.base_pose[2] == 1.0
    cmd = JointCommand(position=[0.1] * 31, effort=[1.0] * 31)
    new_state = sim.step(cmd)
    assert new_state is not None
    assert sim.get_ground_truth()["joint_count"] == 31


def test_sim_interface_abstract():
    """SimInterface 不能直接实例化(抽象)。"""
    try:
        SimInterface()  # type: ignore
        assert False, "should have raised TypeError"
    except TypeError:
        pass


def test_mock_simulation_step_count():
    sim = MockSimulator(num_dof=6)
    sim.reset()
    for _ in range(10):
        sim.step(JointCommand(position=[0.01] * 6))
    assert sim._t > 0.05  # 0.01s × 10 steps
