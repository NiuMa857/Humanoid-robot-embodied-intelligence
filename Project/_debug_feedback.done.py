"""Debug FeedbackMonitor"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')
import math
from common.types import RobotState, IMUState
from motion_control import FeedbackMonitor

q = (0.0, 0.5, 0.0, 0.866)
state = RobotState(
    base_pose=(0, 0, 1.0, 0, 0.5, 0, 0.866),
    imu=IMUState(orientation_xyzw=q),
)
print('state.imu:', state.imu)
print('state.imu.orientation_xyzw:', state.imu.orientation_xyzw)

fb = FeedbackMonitor(config={"enabled_levels": ["L1", "L2", "L3", "L4", "L5"]})
print('max_tilt_deg:', fb.max_tilt_deg)

# 手动跑
qx, qy, qz, qw = state.imu.orientation_xyzw
print(f'qx, qy, qz, qw: {qx}, {qy}, {qz}, {qw}')
sinp = 2 * (qw * qy - qz * qx)
print(f'sinp: {sinp}')
if abs(sinp) > 1:
    pitch_deg = 90.0
else:
    pitch_deg = abs(math.degrees(math.asin(sinp)))
print(f'pitch_deg: {pitch_deg}')

print('check_stability result:', fb.check_stability(state))
