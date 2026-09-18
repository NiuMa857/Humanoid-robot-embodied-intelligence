"""域随机化(Readme §9.2)。"""
from __future__ import annotations
import random
from dataclasses import dataclass, field, asdict


@dataclass
class DomainRandomization:
    """每 episode 随机化物理参数。

    Args:
        ranges: 字段名 → (min, max) 范围
    """
    ranges: dict = field(default_factory=lambda: {
        "mass": (0.8, 1.2),
        "friction": (0.5, 1.5),
        "motor_strength": (0.9, 1.1),
        "imu_noise_std": (0.001, 0.02),
        "vision_noise_std": (0.001, 0.01),
        "action_delay_steps": (0, 3),
    })

    def sample(self) -> dict[str, float]:
        """为每个字段独立采样(连续用 uniform,整数用 randint)。"""
        out: dict = {}
        for k, v in self.ranges.items():
            if isinstance(v, tuple) and len(v) == 2:
                lo, hi = v
                if isinstance(lo, int) and isinstance(hi, int):
                    out[k] = random.randint(lo, hi)
                else:
                    out[k] = random.uniform(lo, hi)
            else:
                out[k] = v
        return out

    def to_dict(self) -> dict:
        return asdict(self)


__all__ = ["DomainRandomization"]
