"""semver 版本管理(major.minor.patch + 预发布标签)。"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class VersionMetadata:
    semver: str = "0.0.1"
    trained_at: str = ""                # ISO-8601 UTC
    train_dataset: str = ""
    metrics: dict = field(default_factory=dict)    # {"success_rate": 0.82, "ep_len": 480}
    git_commit: str = ""
    notes: str = ""

    @staticmethod
    def now() -> str:
        # 含微秒,确保同一秒内多次 register 仍可区分
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


_SEMVER_RE = re.compile(
    r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
    r"(?:-(?P<prerelease>[0-9A-Za-z\-\.]+))?"
    r"(?:\+(?P<build>[0-9A-Za-z\-\.]+))?$"
)


def parse_semver(v: str) -> tuple[int, int, int, str, str]:
    """解析 semver 字符串。

    Returns:
        (major, minor, patch, prerelease, build)
    """
    m = _SEMVER_RE.match(v)
    if not m:
        raise ValueError(f"invalid semver: {v!r}")
    return (
        int(m.group("major")),
        int(m.group("minor")),
        int(m.group("patch")),
        m.group("prerelease") or "",
        m.group("build") or "",
    )


def cmp_semver(a: str, b: str) -> int:
    """比较两个 semver,返回 -1 / 0 / 1。"""
    A = parse_semver(a)
    B = parse_semver(b)
    for x, y in zip(A[:3], B[:3]):
        if x < y:
            return -1
        if x > y:
            return 1
    if A[3] != B[3]:
        # 预发布版本 < 正式版本;预发布之间按字符串
        if not A[3]:
            return 1
        if not B[3]:
            return -1
        return -1 if A[3] < B[3] else 1
    if A[4] != B[4]:
        return -1 if A[4] < B[4] else 1
    return 0


class VersionManager:
    """版本号管理:基于初始版本自动 bump。"""

    def __init__(self, initial: str = "0.0.0"):
        self._current = initial

    @property
    def current(self) -> str:
        return self._current

    def bump(self, level: str = "patch") -> str:
        """bump 版本并返回新版本号。

        level: "major" / "minor" / "patch"。
        """
        major, minor, patch, pre, build = parse_semver(self._current)
        if level == "major":
            major += 1
            minor = 0
            patch = 0
        elif level == "minor":
            minor += 1
            patch = 0
        elif level == "patch":
            patch += 1
        else:
            raise ValueError(f"invalid level: {level!r}, expected major/minor/patch")
        self._current = f"{major}.{minor}.{patch}"
        return self._current

    def set(self, version: str) -> None:
        """直接设版本(校验格式)。"""
        parse_semver(version)  # 校验
        self._current = version

    def stamp(self, **kwargs) -> VersionMetadata:
        """生成当前版本的元数据快照。"""
        return VersionMetadata(semver=self._current, trained_at=VersionMetadata.now(), **kwargs)


__all__ = [
    "VersionMetadata",
    "VersionManager",
    "parse_semver",
    "cmp_semver",
]
