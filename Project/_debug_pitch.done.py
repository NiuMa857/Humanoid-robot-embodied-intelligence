"""Debug: 检查 pitch 计算"""
import sys
sys.path.insert(0, '.')
import math

q = (0.0, 0.5, 0.0, 0.866)
qx, qy, qz, qw = q
sinp = 2 * (qw * qy - qz * qx)
print('sinp:', sinp)
clamped = max(-1, min(1, sinp))
print('clamped sinp:', clamped)
print('pitch_deg:', abs(math.degrees(math.asin(clamped))))

# 尝试一些值
for angle in [10, 30, 45, 60, 70, 80]:
    half = math.radians(angle) / 2
    q_pitch = (0.0, math.sin(half), 0.0, math.cos(half))
    qx, qy, qz, qw = q_pitch
    sinp = 2 * (qw * qy - qz * qx)
    print(f'target {angle} deg: sinp={sinp:.4f}, computed={abs(math.degrees(math.asin(max(-1, min(1, sinp))))):.1f}')
