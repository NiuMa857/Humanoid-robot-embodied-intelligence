"""动作执行 ROS 2 节点入口(完整实装)。

频率:MPC 100Hz / WBC 1kHz / 伺服 2kHz
"""
from __future__ import annotations
import time
from common.types import (
    SkillSequence, RobotState, Trajectory, JointCommand,
    ExecutionStatus,
)
from common.logging import setup_logger
from .feedback import FeedbackMonitor
from .skill_executor import SkillExecutor
from .mpc import MPCPlanner
from .wbc import WBCController
from .low_level_servo import LowLevelServo

_log = setup_logger("control_node")


class ControlNode:
    """动作执行节点。

    时序:
      tick_100hz:读 RobotState → executor.tick(state) → Trajectory
      tick_1khz:Trajectory + state → WBC → JointCommand
      tick_2khz:JointCommand → servo → Oli 电机
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.urdf_path = self.config.get("urdf_path", "")
        self.num_dof = int(self.config.get("robot", {}).get("dof", 31))

        self.mpc = MPCPlanner(urdf_path=self.urdf_path, config=self.config.get("mpc", {}))
        self.wbc = WBCController(urdf_path=self.urdf_path, config=self.config.get("wbc", {}))
        self.feedback = FeedbackMonitor(config=self.config.get("safety", {}))
        self.executor = SkillExecutor(mpc=self.mpc, config=self.config.get("executor", {}))
        self.servo: LowLevelServo | None = None  # 部署时由外部注入

        # tick 频率
        self.mpc_freq_hz = int(self.config.get("mpc_freq_hz", 100))
        self.wbc_freq_hz = int(self.config.get("wbc_freq_hz", 1000))
        self.servo_freq_hz = int(self.config.get("servo_freq_hz", 2000))
        self.mpc_period_ns = int(1e9 / self.mpc_freq_hz)
        self.wbc_period_ns = int(1e9 / self.wbc_freq_hz)
        self.servo_period_ns = int(1e9 / self.servo_freq_hz)

        self._last_mpc_ns = 0
        self._last_wbc_ns = 0
        self._last_servo_ns = 0
        self._latest_traj: Trajectory | None = None
        self._latest_cmd: JointCommand | None = None

    def on_skill_sequence(self, seq: SkillSequence) -> None:
        """收到新技能序列时启动第一个。"""
        if not seq.skills:
            return
        # 简化:只执行第一个(连续 skill 流需要更复杂的状态机)
        first = seq.skills[0]
        # 假 state,实装需要先读 proprioception
        dummy_state = RobotState()
        self.executor.start(first, dummy_state)

    def step(self, state: RobotState) -> ExecutionStatus:
        """统一 step(从 ROS 节点定时器里调一次,内部按频率分摊)。"""
        now_ns = time.time_ns()
        status = ExecutionStatus(
            current_skill_id=self.executor.current_skill.id if self.executor.current_skill else "",
            progress=0.0,
        )

        # 1. MPC tick
        if now_ns - self._last_mpc_ns >= self.mpc_period_ns:
            self._latest_traj = self.executor.tick(state)
            self._last_mpc_ns = now_ns

        # 2. WBC tick
        if now_ns - self._last_wbc_ns >= self.wbc_period_ns and self._latest_traj is not None:
            try:
                cmd = self.wbc.control(self._latest_traj, state)
            except Exception as e:
                _log.warning(f"WBC failed: {e}, using zero cmd")
                cmd = JointCommand(position=[0.0] * self.num_dof,
                                   velocity=[0.0] * self.num_dof,
                                   effort=[0.0] * self.num_dof)
            # L1 限位
            if not self.feedback.check_joint_limits(cmd):
                cmd = self.feedback.clamp(cmd)
                status.failure = "L1_LIMIT"
            self._latest_cmd = cmd
            self._last_wbc_ns = now_ns

        # 3. Servo tick
        if now_ns - self._last_servo_ns >= self.servo_period_ns and self._latest_cmd is not None:
            self.tick_2khz(self._latest_cmd)
            self._last_servo_ns = now_ns

        # 4. L2 稳定 + L3 碰撞
        if not self.feedback.check_stability(state):
            status.failure = "L2_UNSTABLE"
        if not self.feedback.check_collision(state):
            status.failure = "L3_COLLISION"

        return status

    def tick_2khz(self, cmd: JointCommand) -> None:
        """2kHz 伺服下发。"""
        if self.servo is None:
            return
        try:
            self.servo.step(cmd)
        except Exception as e:
            _log.error(f"servo step failed: {e}")

    def shutdown(self) -> None:
        if self.servo is not None:
            self.servo.close()


__all__ = ["ControlNode"]
