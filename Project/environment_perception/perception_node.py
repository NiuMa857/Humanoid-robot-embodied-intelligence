"""环境感知 ROS 2 节点入口。

订阅:head_rgbd / chest_rgbd
发布:SceneGraph / ObjectList(Readme §3.3)

频率:30 Hz
"""
from __future__ import annotations
import time
from common.types import SceneGraph
from .rgbd_camera import RGBDCamera
from .object_detector import ObjectDetector, MockObjectDetector
from .scene_graph import SceneGraphBuilder


class PerceptionNode:
    """环境感知节点(支持 mock / sim / real 模式)。"""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.mode = self.config.get("mode", "mock")
        self.freq_hz = int(self.config.get("freq_hz", 30))
        self.dt_s = 1.0 / self.freq_hz

        self.head_cam = RGBDCamera(
            serial=self.config.get("head_cam", {}).get("serial", ""),
            mode=self.mode,
            width=self.config.get("head_cam", {}).get("width", 640),
            height=self.config.get("head_cam", {}).get("height", 480),
        )
        self.chest_cam = RGBDCamera(
            serial=self.config.get("chest_cam", {}).get("serial", ""),
            mode=self.mode,
        )

        # 检测器:从 model_storage 加载 ONNX,或用 mock
        detector_cfg = self.config.get("detector", {})
        if detector_cfg.get("model_path"):
            from .object_detector import OnnxObjectDetector
            self.detector: ObjectDetector = OnnxObjectDetector(
                model_path=detector_cfg["model_path"],
                conf_threshold=detector_cfg.get("conf_threshold", 0.5),
            )
        else:
            self.detector = MockObjectDetector()

        self.graph_builder = SceneGraphBuilder(
            config=self.config.get("scene_graph", {})
        )

    def step(self) -> SceneGraph | None:
        """单步:取帧 → 检测 → 构图。"""
        head_rgb, head_depth, ts = self.head_cam.read()
        chest_rgb, chest_depth, _ = self.chest_cam.read()
        if head_rgb is None:
            return None
        head_objs = self.detector.detect(head_rgb, head_depth)
        chest_objs = self.detector.detect(chest_rgb, chest_depth) if chest_rgb is not None else None
        return self.graph_builder.build(head_objs, chest_objs, timestamp_ns=ts or time.time_ns())

    def run(self, max_steps: int | None = None) -> None:
        try:
            i = 0
            while max_steps is None or i < max_steps:
                sg = self.step()
                if sg is not None:
                    # TODO: 发布到 ROS 2 topic
                    pass
                time.sleep(self.dt_s)
                i += 1
        except KeyboardInterrupt:
            pass

    def shutdown(self) -> None:
        self.head_cam.close()
        self.chest_cam.close()


__all__ = ["PerceptionNode"]
