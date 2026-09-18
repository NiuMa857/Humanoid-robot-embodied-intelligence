"""Batch 2 smoke test - proprioception + environment_perception."""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')
import logging
logging.disable(logging.CRITICAL)

import time
import math
from common.types import IMUState, JointState
from proprioception import (
    StateEstimator, KinematicsBackend,
    quat_normalize, quat_rotate, quat_to_euler_zyx,
    IMUReader, JointStateReader, ProprioceptionNode,
)
from environment_perception import (
    RGBDCamera, SceneGraphBuilder, MockObjectDetector,
    PerceptionNode, OnnxObjectDetector,
)
from common.types import DetectedObject

# 1. quat utils
q = quat_normalize((0.0, 0.0, 0.0, 1.0))
assert q == (0.0, 0.0, 0.0, 1.0)
rpy = quat_to_euler_zyx((0.0, 0.0, 0.0, 1.0))
assert all(abs(v) < 1e-9 for v in rpy)
v = quat_rotate((0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
assert all(abs(v[i] - (1.0, 0.0, 0.0)[i]) < 1e-9 for i in range(3))
print('quat utils OK')

# 2. StateEstimator
est = StateEstimator(config={"robot": {"dof": 31}})
imu = IMUState(orientation_xyzw=(0.0, 0.0, 0.0, 1.0), angular_velocity=(0.1, 0.0, 0.0))
joint = JointState(position=[0.0]*31, velocity=[0.0]*31, effort=[0.0]*31, timestamp_ns=1000000)
state = est.estimate(imu, joint, timestamp_ns=1000000000)
assert state.imu.angular_velocity[0] == 0.1
print('StateEstimator OK')

# 3. IMUReader mock
imu_r = IMUReader(mode='mock')
imu_state = imu_r.read()
assert imu_state.angular_velocity is not None
imu_r.close()
print('IMUReader mock OK')

# 4. JointStateReader mock
joint_r = JointStateReader(num_dof=31, mode='mock')
js = joint_r.read()
assert len(js.position) == 31
assert js.timestamp_ns > 0
# 超限应拒绝
from common.types import JointCommand
ok = joint_r.send_command(JointCommand(effort=[200.0]*31))  # 超过 150 N·m
assert ok is False
ok = joint_r.send_command(JointCommand(effort=[10.0]*31))   # 合规
assert ok is True
joint_r.close()
print('JointStateReader mock + L1 limit OK')

# 5. ProprioceptionNode end-to-end
node = ProprioceptionNode(config={"mode": "mock", "robot": {"dof": 31}})
node.calibrate()
for _ in range(10):
    s = node.step()
node.shutdown()
print('ProprioceptionNode OK')

# 6. SceneGraphBuilder NMS
import numpy as np
sgb = SceneGraphBuilder()
# 两个重叠的 screwdrivers + 一个 box
objs = [
    DetectedObject(name='screwdriver', bbox_xyxy=(10, 20, 50, 80), pose_xyz_rpy=(0.5, 0.3, 0.8, 0, 0, 0), confidence=0.9),
    DetectedObject(name='screwdriver', bbox_xyxy=(12, 22, 52, 82), pose_xyz_rpy=(0.5, 0.3, 0.8, 0, 0, 0), confidence=0.85),
    DetectedObject(name='box', bbox_xyxy=(200, 200, 300, 300), pose_xyz_rpy=(1.0, 1.0, 0.2, 0, 0, 0), confidence=0.7),
]
sg = sgb.build(head_objects=objs, chest_objects=None, timestamp_ns=0)
# NMS 后:1 个 screwdriver(conf 0.9)+ 1 个 box(bbox 大 z 小 → obstacle)
assert len(sg.objects) + len(sg.obstacles) == 2
print('SceneGraphBuilder NMS OK')

# 7. RGBDCamera mock
cam = RGBDCamera(mode='mock', width=640, height=480)
rgb, depth, ts = cam.read()
assert rgb is not None and rgb.shape == (480, 640, 3)
assert depth is not None and depth.shape == (480, 640)
cam.close()
print('RGBDCamera mock OK')

# 8. MockObjectDetector
det = MockObjectDetector(fixed_results=[DetectedObject(name='cup')])
res = det.detect(rgb, depth)
assert len(res) == 1
print('MockObjectDetector OK')

# 9. PerceptionNode end-to-end
pnode = PerceptionNode(config={"mode": "mock"})
sg = pnode.step()
assert sg is not None
pnode.shutdown()
print('PerceptionNode end-to-end OK')

print('ALL BATCH-2 TESTS PASSED')
