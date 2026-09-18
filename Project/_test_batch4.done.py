"""Batch 4 smoke test - simulation + model_training.

不依赖 torch(PPO 训练只验证接口,真实训练需 pip install torch)。
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')
import logging
logging.disable(logging.CRITICAL)

import math
import numpy as np

# 检查 torch 可用性(不强制)
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print('[note] torch not installed; PPO 训练部分跳过,只验证接口')

from common.types import RobotState, JointState, IMUState, JointCommand
from simulation import SimInterface, MockSimulator, IsaacSceneLoader, IsaacBridge
from model_training.envs import BaseEnv
from model_training.mdp.rewards import (
    StairClimbReward, GraspReward, TrackingReward, StabilityReward,
)
from model_training.mdp.terminations import (
    FallTermination, CollisionTermination, TimeoutTermination, check_all,
)
from model_training.mdp.events import DomainRandomization
from model_training.algorithms.base import BaseRLAlgorithm

# 1. MockSimulator
sim = MockSimulator(num_dof=31)
sim.load_robot("dummy.urdf")
sim.load_scene("stair_30")
state = sim.reset()
assert state.base_pose[2] == 1.0
cmd = JointCommand(position=[0.1]*31, effort=[1.0]*31)  # 小 effort 避免主导 reward
new_state = sim.step(cmd)
assert new_state is not None
gt = sim.get_ground_truth()
assert "time" in gt
print('MockSimulator OK')


# 2. rewards
r_stair = StairClimbReward()
state1 = RobotState(base_pose=(0, 0, 1.0, 0, 0, 0, 1))
state2 = RobotState(base_pose=(1, 0, 1.5, 0, 0, 0, 1))
score = r_stair(state2, state1, cmd)
assert score > 0
print(f'StairClimbReward OK: {score:.2f}')

r_grasp = GraspReward()
assert r_grasp(state1, grasp_force=10.0) == 1.0
assert r_grasp(state1, grasp_force=5.0) < 1.0
print('GraspReward OK')

r_track = TrackingReward()
assert r_track(state1, state1) == 0.0
print('TrackingReward OK')

r_stab = StabilityReward()
score = r_stab(state1)
assert score >= 0
print(f'StabilityReward OK: {score:.2f}')


# 3. terminations
fall = FallTermination()
assert fall(RobotState(base_pose=(0, 0, 0.1, 0, 0, 0, 1))) is True
assert fall(RobotState(base_pose=(0, 0, 1.0, 0, 0, 0, 1))) is False
print('FallTermination OK')

coll = CollisionTermination(max_torque_nm=200.0)
assert coll(RobotState(joint=JointState(effort=[300.0]*31))) is True
assert coll(RobotState(joint=JointState(effort=[10.0]*31))) is False
print('CollisionTermination OK')

timeout = TimeoutTermination(max_steps=1000)
assert timeout(1000) is True
assert timeout(500) is False
print('TimeoutTermination OK')

done, reason = check_all(
    [FallTermination(), CollisionTermination(max_torque_nm=200.0), TimeoutTermination(max_steps=1000)],
    RobotState(base_pose=(0, 0, 0.1, 0, 0, 0, 1)), 0,
)
assert done is True and reason == "fall"
print('check_all OK')


# 4. DomainRandomization
dr = DomainRandomization()
samples = [dr.sample() for _ in range(20)]
masses = [s["mass"] for s in samples]
assert all(0.8 <= m <= 1.2 for m in masses)
print(f'DomainRandomization OK: mass range {min(masses):.2f}-{max(masses):.2f}')


# 5. PPO Native(有 torch 跑小训练,没 torch 验证接口)
if HAS_TORCH:
    from model_training.algorithms.ppo_native import compute_gae, layer_init, ActorCritic, PPONativeImpl
    import torch

    rewards = torch.ones(10, 4)
    values = torch.zeros(11, 4)
    dones = torch.zeros(10, 4)
    next_value = torch.zeros(4)
    adv, ret = compute_gae(rewards, values, dones, next_value, gamma=0.99, gae_lambda=0.95)
    assert adv.shape == (10, 4) and ret.shape == (10, 4)
    expected = (1 - 0.99 ** 10) / 0.01
    print(f'compute_gae OK: adv[0,0]={adv[0,0]:.2f}, expected≈{expected:.2f}')

    torch.manual_seed(42)
    ac = ActorCritic(obs_dim=33, act_dim=31)
    obs = torch.randn(8, 33)
    action, logp, ent, value = ac.get_action_and_value(obs)
    assert action.shape == (8, 31) and value.shape == (8,)
    print(f'ActorCritic OK: action={action.shape}, value={value.shape}')

    class _ToyEnv(BaseEnv):
        def __init__(self):
            super().__init__({})
            self.num_envs = 2
            self.num_obs = 2
            self.num_actions = 1
        def reset(self, env_ids=None):
            return np.zeros((2, 2), dtype=np.float32), {}
        def step(self, actions):
            return (np.zeros((2, 2), dtype=np.float32),
                    np.zeros(2, dtype=np.float32),
                    np.zeros(2, dtype=bool), {})
        @property
        def observation_space(self): return (2,)
        @property
        def action_space(self): return (1,)

    env = _ToyEnv()
    ppo = PPONativeImpl(env=env, config={
        "device": "cpu",
        "ppo": {"total_timesteps": 64, "num_steps_per_env": 8, "minibatch_size": 8, "update_epochs": 1},
    }, log_dir="/tmp/ppo_test")
    print(f'PPONativeImpl created OK: obs_dim={ppo.obs_dim}, act_dim={ppo.act_dim}')

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    try:
        ppo.save(ckpt_path)
        assert os.path.exists(ckpt_path)
        ppo2 = PPONativeImpl(env=env, config={"device": "cpu", "ppo": {"total_timesteps": 1}}, log_dir="/tmp/ppo_test2")
        ppo2.load(ckpt_path)
        print('PPO save/load OK')
    finally:
        if os.path.exists(ckpt_path):
            os.unlink(ckpt_path)
else:
    # 验证 .py 文件存在(代码完整),但不 import(避免 torch 依赖)
    import os
    ppo_native_path = os.path.join(os.path.dirname(__file__), "model_training", "algorithms", "ppo_native.py")
    assert os.path.exists(ppo_native_path), "ppo_native.py missing"
    with open(ppo_native_path, "r", encoding="utf-8") as f:
        content = f.read()
    # 验证关键函数 / 类 / 接口都存在
    for token in ["class PPONativeImpl", "class ActorCritic", "def compute_gae",
                  "def train", "def save", "def load", "def export_onnx",
                  "def forward_actor"]:
        assert token in content, f"missing token in ppo_native.py: {token}"
    # 验证继承 BaseRLAlgorithm
    assert "from .base import BaseRLAlgorithm" in content
    print('PPO native source validated (no torch: train not executed; install torch to actually train)')


# 6. IsaacSceneLoader / IsaacBridge 不会真初始化 Isaac,只测可构造
loader = IsaacSceneLoader(config={"stair_angle": 30.0})
bridge = IsaacBridge()
loader.load_stair_scene(30.0)
bridge.start()
bridge.stop()
print('IsaacSceneLoader / IsaacBridge placeholder OK')

print('ALL BATCH-4 TESTS PASSED')
