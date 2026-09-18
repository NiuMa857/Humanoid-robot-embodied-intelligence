"""目标检测器抽象 + ONNX Runtime 实装。

支持:
  - onnxruntime(默认,跨平台)
  - 抽象基类让用户注入 PyTorch / TensorRT 等其他后端
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import numpy as np

from common.types import DetectedObject


class ObjectDetector(ABC):
    """目标检测器抽象接口。"""

    @abstractmethod
    def detect(self, rgb: np.ndarray, depth: np.ndarray | None = None) -> list[DetectedObject]:
        """从 RGB(+可选 Depth)输出物体列表。"""
        ...

    @property
    @abstractmethod
    def input_size(self) -> tuple[int, int]:
        """(H, W)"""
        ...

    def preprocess(self, rgb: np.ndarray) -> np.ndarray:
        """默认预处理:resize + normalize + NCHW。子类可覆盖。"""
        h, w = self.input_size
        if rgb.shape[:2] != (h, w):
            import cv2
            rgb = cv2.resize(rgb, (w, h))
        # BGR → RGB, [0,255] → [0,1]
        x = rgb[:, :, ::-1].astype(np.float32) / 255.0
        # HWC → NCHW
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, axis=0)
        return x


class OnnxObjectDetector(ObjectDetector):
    """ONNX Runtime 检测器。

    假设模型输出格式:list[DetectedObject] 已解析的(NMS 后处理由模型导出时包含)。
    如果你的模型是 YOLO 风格(输出 raw boxes),重写 `postprocess`。
    """

    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int] = (640, 480),
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        providers: list[str] | None = None,
    ):
        self.model_path = model_path
        self._input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._session = None
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                model_path,
                providers=providers or ort.get_available_providers(),
            )
            self._input_name = self._session.get_inputs()[0].name
            self._output_names = [o.name for o in self._session.get_outputs()]
        except ImportError as e:
            raise RuntimeError("onnxruntime not installed: `pip install onnxruntime`") from e

    def detect(self, rgb: np.ndarray, depth: np.ndarray | None = None) -> list[DetectedObject]:
        x = self.preprocess(rgb)
        outputs = self._session.run(self._output_names, {self._input_name: x})
        return self.postprocess(outputs, depth)

    def postprocess(self, model_outputs: list, depth: np.ndarray | None = None) -> list[DetectedObject]:
        """默认实现:假设模型直接输出 [{name, bbox, conf}, ...](已在 ONNX 导出时做 NMS)。

        YOLO / Faster R-CNN 等需要在此重写。
        """
        if not model_outputs:
            return []
        # 取第一个输出,期望是 list[dict] 或 numpy structured array
        out = model_outputs[0]
        results: list[DetectedObject] = []
        if isinstance(out, list):
            for item in out:
                if not isinstance(item, dict):
                    continue
                if item.get("confidence", 0.0) < self.conf_threshold:
                    continue
                # 6D 位姿需要另外跑位姿估计模型(FoundationPose / PoseCNN)
                pose = item.get("pose_xyz_rpy", (0.0,) * 6)
                results.append(DetectedObject(
                    name=item.get("name", "object"),
                    bbox_xyxy=tuple(item.get("bbox", (0, 0, 0, 0))),
                    pose_xyz_rpy=tuple(pose),
                    confidence=float(item.get("confidence", 0.0)),
                ))
        return results

    @property
    def input_size(self) -> tuple[int, int]:
        return self._input_size


class MockObjectDetector(ObjectDetector):
    """Mock 检测器(无 ONNX 时用,返回空列表或固定结果)。"""

    def __init__(self, fixed_results: list[DetectedObject] | None = None):
        self._results = fixed_results or []
        self._input_size = (480, 640)

    def detect(self, rgb: np.ndarray, depth: np.ndarray | None = None) -> list[DetectedObject]:
        return self._results

    @property
    def input_size(self) -> tuple[int, int]:
        return self._input_size


__all__ = ["ObjectDetector", "OnnxObjectDetector", "MockObjectDetector"]
