"""技能执行器:Skill 状态机 + 推进。"""
from __future__ import annotations
import time
from common.types import Skill, RobotState, Trajectory, JointCommand
from common.logging import setup_logger

_log = setup_logger("skill_executor")


class SkillExecutor:
    """单 skill 状态机:idle → running → success/failed。

    MPC 抽象(由 config["mpc"] 注入);无 MPC 时给零阶 fallback。
    """

    STATES = ("idle", "running", "success", "failed")

    def __init__(self, mpc=None, config: dict | None = None):
        self.mpc = mpc
        self.config = config or {}
        self._state = "idle"
        self._current_skill: Skill | None = None
        self._start_time_ns: int = 0
        self._last_progress: float = 0.0

    def start(self, skill: Skill, state: RobotState) -> None:
        if self._state == "running":
            _log.warning(f"start() called while already running {self._current_skill.id if self._current_skill else '?'}, ignoring")
            return
        self._current_skill = skill
        self._state = "running"
        self._start_time_ns = time.time_ns()
        self._last_progress = 0.0
        _log.info(f"skill started: {skill.id}")

    def tick(self, state: RobotState) -> Trajectory:
        """单步:让 MPC 重新规划一次(或 fallback)。"""
        if self._state != "running" or self._current_skill is None:
            return Trajectory()
        if self.mpc is not None and hasattr(self.mpc, "plan"):
            try:
                return self.mpc.plan(self._current_skill, state)
            except Exception as e:
                _log.warning(f"MPC.plan failed: {e}, using zero-order fallback")
        return self._fallback_trajectory(state)

    def check_timeout(self, timeout_s: float) -> bool:
        """检查是否超时。返回 True = 超时了。"""
        if self._state != "running":
            return False
        elapsed = (time.time_ns() - self._start_time_ns) * 1e-9
        return elapsed > timeout_s

    def complete(self, success: bool = True) -> None:
        self._state = "success" if success else "failed"
        _log.info(f"skill {self._current_skill.id if self._current_skill else '?'} -> {self._state}")

    def _fallback_trajectory(self, state: RobotState) -> Trajectory:
        """零阶轨迹:返回当前状态作为唯一 waypoint(没有 MPC 时的占位)。"""
        return Trajectory(
            waypoints=[state.base_pose],
            horizon_s=0.0,
            dt_s=0.0,
        )

    @property
    def is_running(self) -> bool:
        return self._state == "running"

    @property
    def current_skill(self) -> Skill | None:
        return self._current_skill

    @property
    def state(self) -> str:
        return self._state

    def reset(self) -> None:
        self._state = "idle"
        self._current_skill = None
        self._start_time_ns = 0


__all__ = ["SkillExecutor"]
