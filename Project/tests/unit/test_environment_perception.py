"""environment_perception/ 单元测试:RGBD + 检测 + 场景图。"""
import sys
sys.path.insert(0, ".")
import numpy as np

from common.types import DetectedObject
from environment_perception import (
    RGBDCamera, SceneGraphBuilder, MockObjectDetector, PerceptionNode,
)


def test_rgbd_mock_shape():
    cam = RGBDCamera(mode="mock", width=640, height=480)
    rgb, depth, ts = cam.read()
    assert rgb.shape == (480, 640, 3)
    assert depth.shape == (480, 640)
    cam.close()


def test_scene_graph_nms_dedup():
    sgb = SceneGraphBuilder()
    objs = [
        DetectedObject(name="box", bbox_xyxy=(10, 20, 50, 80),
                       pose_xyz_rpy=(0.5, 0.3, 0.8, 0, 0, 0), confidence=0.9),
        DetectedObject(name="box", bbox_xyxy=(12, 22, 52, 82),
                       pose_xyz_rpy=(0.5, 0.3, 0.8, 0, 0, 0), confidence=0.85),
        DetectedObject(name="cup", bbox_xyxy=(300, 300, 350, 350),
                       pose_xyz_rpy=(1.0, 1.0, 1.0, 0, 0, 0), confidence=0.7),
    ]
    sg = sgb.build(objs, chest_objects=None, timestamp_ns=0)
    # box 应只剩 1 个(conf 0.9)
    box_count = sum(1 for o in sg.objects + sg.obstacles if o.name == "box")
    assert box_count == 1


def test_mock_detector_returns_fixed():
    det = MockObjectDetector(fixed_results=[DetectedObject(name="x")])
    res = det.detect(None, None)
    assert len(res) == 1 and res[0].name == "x"


def test_perception_node_e2e():
    node = PerceptionNode(config={"mode": "mock"})
    sg = node.step()
    assert sg is not None
    assert sg.timestamp_ns > 0
    node.shutdown()
