"""执行反馈 / 安全监测(Readme §7 + §10,L1-L5 完整实装)。"""
from __future__ import annotations
from common.types import JointCommand, RobotState, FailureType
from common.logging import setup_logger

_log = setup_logger("feedback")


class FeedbackMonitor:
    """安全监测器:L1-L5 通用规则。"""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        # 默认阈值
        self.max_torque_nm = float(self.config.get("max_torque_nm", 150.0))
        self.max_tilt_deg = float(self.config.get("max_tilt_deg", 60.0))
        self.min_base_height = float(self.config.get("min_base_height", 0.3))
        self.comm_timeout_s = float(self.config.get("comm_timeout_s", 3.0))
        self._enabled_levels = set(self.config.get("enabled_levels", ["L1", "L2", "L3", "L4", "L5"]))

    # ============================================================
    # L1:关节力矩限位
    # ============================================================
    def check_joint_limits(self, cmd: JointCommand) -> bool:
        """L1:检查关节指令是否超力矩限位。"""
        if "L1" not in self._enabled_levels:
            return True
        if not cmd.effort:
            return True
        return all(abs(t) <= self.max_torque_nm for t in cmd.effort)

    def clamp(self, cmd: JointCommand) -> JointCommand:
        """把超限 effort 钳到 ±max_torque_nm。"""
        cmd.effort = [max(-self.max_torque_nm, min(self.max_torque_nm, t)) for t in cmd.effort]
        return cmd

    # ============================================================
    # L2:姿态稳定(简化:base 高度 + 倾角)
    # ============================================================
    def check_stability(self, state: RobotState) -> bool:
        """L2:检查 base 高度和姿态角。"""
        if "L2" not in self._enabled_levels:
            return True
        if state.base_pose[2] < self.min_base_height:
            _log.warning(f"L2: base too low: {state.base_pose[2]:.2f}m")
            return False
        # 倾角:四元数 → roll/pitch
        qx, qy, qz, qw = state.imu.orientation_xyzw if state.imu else (0, 0, 0, 1)
        import math
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) > 1:
            pitch_deg = 90.0
        else:
            pitch_deg = abs(math.degrees(math.asin(sinp)))
        if pitch_deg >= self.max_tilt_deg:
            _log.warning(f"L2: pitch too high: {pitch_deg:.1f}°")
            return False
        return True

    # ============================================================
    # L3:碰撞检测(简化:力矩突变)
    # ============================================================
    def __init_collision_state(self) -> None:
        if not hasattr(self, "_last_effort"):
            self._last_effort = None
        if not hasattr(self, "_consecutive_collision"):
            self._consecutive_collision = 0

    def check_collision(self, state: RobotState) -> bool:
        """L3:检测力矩突变(> 2x 平均视为碰撞)。"""
        if "L3" not in self._enabled_levels:
            return True
        self.__init_collision_state()
        if state.joint is None or not state.joint.effort:
            return True
        eff = state.joint.effort
        avg = sum(abs(e) for e in eff) / len(eff) if eff else 0
        max_eff = max(abs(e) for e in eff)
        if avg > 0 and max_eff > 2 * avg and max_eff > 50:
            self._consecutive_collision += 1
            if self._consecutive_collision >= 3:
                _log.warning(f"L3: collision detected (max_eff={max_eff:.1f}, avg={avg:.1f})")
                return False
        else:
            self._consecutive_collision = 0
        self._last_effort = eff[:]
        return True

    # ============================================================
    # L4:紧急停止(标记位,真停在低层 servo 里实现)
    # ============================================================
    def emergency_stop(self) -> None:
        """L4:触发紧急停止标志。"""
        if "L4" not in self._enabled_levels:
            return
        _log.critical("L4: EMERGENCY STOP")
        self._estop_active = True

    @property
    def estop_active(self) -> bool:
        return getattr(self, "_estop_active", False)

    # ============================================================
    # L5:通信超时
    # ============================================================
    def check_comm_timeout(self, last_msg_ts_ns: int, now_ts_ns: int) -> bool:
        """L5:检查是否超时(> 3s 默认)。True = OK,False = 超时。"""
        if "L5" not in self._enabled_levels:
            return True
        if last_msg_ts_ns <= 0:
            return True
        elapsed = (now_ts_ns - last_msg_ts_ns) * 1e-9
        return elapsed <= self.comm_timeout_s

    # ============================================================
    # 失败分类(上层调用)
    # ============================================================
    def classify_failure(self, expected: RobotState, actual: RobotState) -> str:
        """根据实际状态 vs 期望,归类失败类型。"""
        if actual.base_pose[2] < self.min_base_height:
            return FailureType.F_STABILITY
        if self.joint_tracking_error(expected, actual) > 1.0:
            return FailureType.F_TRAJECTORY
        return FailureType.F_TIMEOUT

    @staticmethod
    def joint_tracking_error(expected: RobotState, actual: RobotState) -> float:
        """关节跟踪误差(RMSE rad)。"""
        if expected.joint is None or actual.joint is None:
            return 0.0
        if not expected.joint.position or not actual.joint.position:
            return 0.0
        n = min(len(expected.joint.position), len(actual.joint.position))
        err = sum((expected.joint.position[i] - actual.joint.position[i]) ** 2 for i in range(n))
        return (err / n) ** 0.5 if n > 0 else 0.0


__all__ = ["FeedbackMonitor"]
