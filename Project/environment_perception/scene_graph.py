"""场景图构建:多相机检测结果 → SceneGraph(Readme §3.3)。"""
from __future__ import annotations
import numpy as np
from common.types import DetectedObject, SceneGraph


class SceneGraphBuilder:
    """多相机检测融合。

    功能:
      1. 胸相机坐标系 → 世界坐标(用胸-头外参)
      2. 跟头相机结果做 NMS 去重
      3. 区分 objects / obstacles
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        # 外参(胸 → 头,Readme §3.1:两台相机在头部和胸部,需要外参对齐)
        # 简化:默认外参为单位矩阵(假设同坐标系)
        self.chest_to_head_tf = np.eye(4)   # 4x4 SE(3)
        # 障碍物判定:对地面附近 + 大尺寸物体
        self.obstacle_height_max = 0.3   # m
        self.obstacle_size_min = 0.1     # m

    def set_chest_to_head_transform(self, matrix4x4: np.ndarray) -> None:
        self.chest_to_head_tf = matrix4x4

    def build(
        self,
        head_objects: list[DetectedObject],
        chest_objects: list[DetectedObject] | None = None,
        timestamp_ns: int = 0,
    ) -> SceneGraph:
        """合并两视角检测结果。"""
        chest_objects = chest_objects or []
        # 1. 坐标变换
        chest_in_head = [
            self._transform_object(obj, self.chest_to_head_tf)
            for obj in chest_objects
        ]
        all_objects = list(head_objects) + chest_in_head

        # 2. NMS(简单版:按 (name, iou) 去重)
        merged = self._nms(all_objects, iou_threshold=0.5)

        # 3. 区分 objects / obstacles
        objects, obstacles = [], []
        for obj in merged:
            if self._is_obstacle(obj):
                obstacles.append(obj)
            else:
                objects.append(obj)

        return SceneGraph(
            objects=objects,
            obstacles=obstacles,
            timestamp_ns=timestamp_ns,
        )

    def _transform_object(self, obj: DetectedObject, tf4x4: np.ndarray) -> DetectedObject:
        """用 4x4 SE(3) 矩阵变换 6D 位姿(xyz 旋转, rpy 不变近似)。"""
        pose = np.array(obj.pose_xyz_rpy[:3] + (1.0,))   # xyz1
        new_xyz = tf4x4 @ pose
        new_pose = (
            float(new_xyz[0]), float(new_xyz[1]), float(new_xyz[2]),
            obj.pose_xyz_rpy[3], obj.pose_xyz_rpy[4], obj.pose_xyz_rpy[5],
        )
        return DetectedObject(
            name=obj.name, bbox_xyxy=obj.bbox_xyxy,
            pose_xyz_rpy=new_pose, confidence=obj.confidence,
        )

    def _nms(self, objs: list[DetectedObject], iou_threshold: float = 0.5) -> list[DetectedObject]:
        """简单 NMS(按 name 分组 + bbox IoU 去重,保留 conf 最高的)。"""
        if not objs:
            return []
        # 按 name 分组
        groups: dict[str, list[DetectedObject]] = {}
        for o in objs:
            groups.setdefault(o.name, []).append(o)

        keep: list[DetectedObject] = []
        for name, items in groups.items():
            # 按 confidence 降序
            items_sorted = sorted(items, key=lambda x: -x.confidence)
            kept: list[DetectedObject] = []
            for item in items_sorted:
                dup = False
                for k in kept:
                    if self._iou(item.bbox_xyxy, k.bbox_xyxy) > iou_threshold:
                        dup = True
                        break
                if not dup:
                    kept.append(item)
            keep.extend(kept)
        return keep

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        """bbox IoU(xyxy 格式)。"""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _is_obstacle(self, obj: DetectedObject) -> bool:
        """简化判定:z < threshold 且 bbox 大 → 障碍物(地面物体)。"""
        z = obj.pose_xyz_rpy[2]
        bw = max(0.0, obj.bbox_xyxy[2] - obj.bbox_xyxy[0]) / 640.0
        return (z < self.obstacle_height_max) and (bw > self.obstacle_size_min)


__all__ = ["SceneGraphBuilder"]
