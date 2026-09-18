"""感知流水线集成测试(无 ROS 2 依赖)。"""
import sys
sys.path.insert(0, ".")
import numpy as np

from common.types import DetectedObject, SceneGraph
from environment_perception.scene_graph import SceneGraphBuilder
from environment_perception.perception_node import PerceptionNode


def test_scene_graph_build_merges_two_views():
    """头 + 胸 两个视角的检测结果合并成 SceneGraph。"""
    builder = SceneGraphBuilder(config={"merge_strategy": "nms"})

    head_objs = [
        DetectedObject(name="screwdriver", bbox_xyxy=(10, 20, 50, 80),
                       pose_xyz_rpy=(0.5, 0.3, 0.8, 0, 0, 0), confidence=0.9),
    ]
    chest_objs = [
        DetectedObject(name="screwdriver", bbox_xyxy=(100, 200, 130, 240),
                       pose_xyz_rpy=(0.5, 0.3, 0.8, 0, 0, 0), confidence=0.85),
    ]
    sg = builder.build(head_objs, chest_objs, timestamp_ns=1234567890)
    assert isinstance(sg, SceneGraph)
    # 两个不同 bbox 的 screwdriver 都保留(不重叠,IoU=0)
    screwdriver_count = sum(1 for o in sg.objects + sg.obstacles if o.name == "screwdriver")
    assert screwdriver_count == 2


def test_perception_node_construction():
    node = PerceptionNode(config={"head_cam": {}, "chest_cam": {}, "detector": {}})
    assert node is not None
