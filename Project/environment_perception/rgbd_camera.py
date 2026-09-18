"""RGB-D 相机驱动 + 标定(Readme §3.1)。

支持 mock / real 两种模式。real 用 pyrealsense2(Oli 用 RealSense D435i,Readme §3.1)。
"""
from __future__ import annotations
import time
from dataclasses import dataclass


@dataclass
class CameraIntrinsics:
    """相机内参(简化版,RealSense 标准)。"""
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    width: int = 640
    height: int = 480


class RGBDCamera:
    """RealSense D435i 抽象(Oli 头部 + 胸部各 1 台,Readme §3.1)。"""

    def __init__(
        self,
        serial: str = "",
        mode: str = "mock",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        self.serial = serial
        self.mode = mode
        self.width = width
        self.height = height
        self.fps = fps
        self._pipeline = None
        self._intrinsics = CameraIntrinsics(width=width, height=height)

        if mode == "real":
            self._init_realsense()

    def _init_realsense(self) -> None:
        """初始化 RealSense 设备。"""
        try:
            import pyrealsense2 as rs
            ctx = rs.context()
            if self.serial:
                # 选指定序列号的设备
                for dev in ctx.query_devices():
                    if dev.get_info(rs.camera_info.serial_number) == self.serial:
                        break
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            self._pipeline = rs.pipeline()
            self._pipeline.start(cfg)
            # 取内参
            profile = self._pipeline.get_active_profile()
            color_stream = profile.get_stream(rs.stream.color)
            intr = color_stream.as_video_stream_profile().get_intrinsics()
            self._intrinsics = CameraIntrinsics(
                fx=intr.fx, fy=intr.fy, cx=intr.ppx, cy=intr.ppy,
                width=intr.width, height=intr.height,
            )
        except ImportError:
            raise RuntimeError("pyrealsense2 not installed; install with `pip install pyrealsense2` or set mode='mock'")
        except Exception as e:
            raise RuntimeError(f"failed to init RealSense: {e}")

    def read(self) -> tuple:
        """同步读一帧 RGB + Depth。

        Returns:
            rgb: HxWx3 uint8 (BGR 顺序,跟 OpenCV 一致)
            depth: HxW uint16(单位 mm)
            timestamp_ns: int
        """
        if self.mode == "real":
            return self._read_realsense()
        return self._read_mock()

    def _read_realsense(self) -> tuple:
        import numpy as np
        frames = self._pipeline.wait_for_frames()
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        rgb = np.asanyarray(color.get_data())
        d = np.asanyarray(depth.get_data())
        ts = int(frames.get_timestamp() * 1e6)   # ms → ns
        return rgb, d, ts

    def _read_mock(self) -> tuple:
        """mock 模式:返回固定大小黑色图 + 零深度。"""
        try:
            import numpy as np
        except ImportError:
            # 没 numpy 就返回 None
            return None, None, time.time_ns()
        rgb = np.zeros((self.height, self.width, 3), dtype="uint8")
        depth = np.zeros((self.height, self.width), dtype="uint16")
        return rgb, depth, time.time_ns()

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None

    @property
    def intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics


__all__ = ["RGBDCamera", "CameraIntrinsics"]
