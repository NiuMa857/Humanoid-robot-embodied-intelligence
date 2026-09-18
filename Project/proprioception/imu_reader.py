"""6 轴 IMU 读取 + 滤波(Readme §3.1)。

通用接口 + 留 Oli SDK 适配点。带 mock 模式(无硬件时也能跑测试)。
"""
from __future__ import annotations
import time
import math
from dataclasses import dataclass

from common.types import IMUState


class IMUReader:
    """IMU 读取器。

    两种模式:
      - real: 调 Oli SDK(待补)读真实 IMU
      - mock: 生成虚拟数据(用于 sim / 测试)
    """

    def __init__(
        self,
        port: str = "",
        mode: str = "mock",
        config: dict | None = None,
    ):
        self.port = port
        self.mode = mode
        self.config = config or {}
        # 标定数据(零位偏移)
        self._gyro_bias = (0.0, 0.0, 0.0)
        self._accel_bias = (0.0, 0.0, 0.0)
        # mock 状态
        self._mock_t0 = time.time()

        if mode == "real":
            # TODO: 打开 Oli IMU 端口
            # 例如:self._dev = OliIMUDevice(port)
            # 或者:import serial; self._dev = serial.Serial(port, baudrate=...)
            self._dev = None
            self.calibrate()

    def calibrate(self, samples: int = 500, sample_rate_hz: float = 1000.0) -> None:
        """采样 N 次取平均,作为 bias 偏移。

        要求机器人**完全静止**时调用。
        """
        if self.mode == "mock":
            self._gyro_bias = (0.0, 0.0, 0.0)
            self._accel_bias = (0.0, 0.0, 0.0)
            return

        # 真实模式:采样 N 次平均
        try:
            gx_sum = gy_sum = gz_sum = 0.0
            ax_sum = ay_sum = az_sum = 0.0
            interval = 1.0 / sample_rate_hz
            for _ in range(samples):
                raw = self._read_raw()
                if raw is None:
                    continue
                gx_sum += raw[0]; gy_sum += raw[1]; gz_sum += raw[2]
                ax_sum += raw[3]; ay_sum += raw[4]; az_sum += raw[5]
                time.sleep(interval)
            self._gyro_bias = (gx_sum / samples, gy_sum / samples, gz_sum / samples)
            self._accel_bias = (ax_sum / samples, ay_sum / samples, az_sum / samples)
        except Exception as e:
            # 标定失败不致命,继续用零 bias
            print(f"[IMUReader] calibration failed: {e}")

    def _read_raw(self) -> tuple | None:
        """读原始 6 轴数据(未校正)。真实模式:从设备读。"""
        if self.mode == "mock":
            # mock:模拟静止 IMU,只有重力(0,0,-9.81)
            t = time.time() - self._mock_t0
            return (
                0.0, 0.0, 0.0,                      # gyro
                0.001 * math.sin(t),                  # accel x
                0.001 * math.sin(2 * t),              # accel y
                -9.81 + 0.001 * math.cos(t),          # accel z (重力 + 抖动)
            )
        # 真实模式 TODO
        # if self._dev is not None:
        #     return self._dev.read()
        return None

    def read(self) -> IMUState:
        """读一帧 IMU 数据(已校准 + 滤波)。

        TODO: 加 ESKF / 互补滤波(目前是简单 bias 减法)。
        """
        raw = self._read_raw()
        if raw is None:
            # 没读到数据,返回零
            return IMUState(timestamp_ns=time.time_ns())

        gx, gy, gz, ax, ay, az = raw
        # 减 bias
        gx -= self._gyro_bias[0]; gy -= self._gyro_bias[1]; gz -= self._gyro_bias[2]
        ax -= self._accel_bias[0]; ay -= self._accel_bias[1]; az -= self._accel_bias[2]

        # 简单姿态估计(用加速度计估计 roll/pitch, yaw 不可观测)
        roll = math.atan2(ay, az)
        pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))
        # 简单四元数 (x, y, z, w)
        cy = math.cos(yaw := 0.0) * 0.0  # yaw 暂时 0
        sy = 0.0
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        # ZYX 内旋:q = qz * qy * qx
        # 这里 yaw=0,简化为:qw=cr*cp, qx=sr*cp, qy=cr*sp, qz=-sr*sp
        qw = cr * cp
        qx = sr * cp
        qy = cr * sp
        qz = -sr * sp

        return IMUState(
            orientation_xyzw=(qx, qy, qz, qw),
            angular_velocity=(gx, gy, gz),
            linear_acceleration=(ax, ay, az),
        )

    def close(self) -> None:
        if self.mode == "real" and self._dev is not None:
            # TODO: self._dev.close()
            pass


__all__ = ["IMUReader"]
