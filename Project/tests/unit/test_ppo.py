"""model_training.algorithms.ppo_native 单元测试。

依赖 torch(可选)。无 torch 时只验证源码完整性。
"""
import sys
sys.path.insert(0, ".")
import os

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from model_training.algorithms.base import BaseRLAlgorithm


def test_ppo_native_interface_exists():
    """不 import ppo_native(避免 torch 强依赖),只验证关键 token。"""
    ppo_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "model_training", "algorithms", "ppo_native.py"
    )
    ppo_path = os.path.normpath(ppo_path)
    assert os.path.exists(ppo_path), f"missing: {ppo_path}"
    with open(ppo_path, "r", encoding="utf-8") as f:
        src = f.read()
    for token in [
        "class PPONativeImpl",
        "class ActorCritic",
        "def compute_gae",
        "def train",
        "def save",
        "def load",
        "def export_onnx",
        "def forward_actor",
        "BaseRLAlgorithm",
        "torch.onnx.export",
        "compute_gae",  # GAE 函数
    ]:
        assert token in src, f"ppo_native.py 缺少关键 token: {token}"


def test_ppo_base_class_interface():
    """BaseRLAlgorithm 抽象接口完整。"""
    assert hasattr(BaseRLAlgorithm, "train")
    assert hasattr(BaseRLAlgorithm, "save")
    assert hasattr(BaseRLAlgorithm, "load")
    assert hasattr(BaseRLAlgorithm, "export_onnx")
    assert hasattr(BaseRLAlgorithm, "close")


def test_ppo_native_inherits_base():
    """PPONativeImpl 必须继承 BaseRLAlgorithm(看源码)。"""
    ppo_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "model_training", "algorithms", "ppo_native.py"
    )
    with open(ppo_path, "r", encoding="utf-8") as f:
        src = f.read()
    assert "class PPONativeImpl(BaseRLAlgorithm)" in src


# 跑得动 torch 时,做端到端小训练(可选)
if HAS_TORCH:
    def test_ppo_mini_train_e2e():
        import numpy as np
        from model_training.algorithms.ppo_native import (
            ActorCritic, PPONativeImpl, compute_gae,
        )
        from model_training.envs import BaseEnv

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
            "ppo": {"total_timesteps": 64, "num_steps_per_env": 8,
                    "minibatch_size": 8, "update_epochs": 1},
        })
        ppo.train()  # 跑 1 次 update
        print("PPO mini-train OK (1 update)")
