"""CleanRL 风格 PPO 自实现(完整主循环)。

参考:
  - https://github.com/vwxyzjn/cleanrl
  - spinningup PPO

不需要 rsl_rl 依赖,纯 PyTorch。便于教学 / 兜底 / 深度定制。
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader

from model_training.envs.base_env import BaseEnv
from model_training.mdp.events import DomainRandomization
from common.types import JointCommand
from common.logging import setup_logger
from .base import BaseRLAlgorithm

_log = setup_logger("ppo_native")


def layer_init(layer: nn.Linear, std: float = math.sqrt(2.0), bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    """Actor-Critic 网络(MLP actor + MLP critic)。"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple = (256, 256, 128)):
        super().__init__()
        h0, h1, h2 = hidden
        # actor:obs -> mean; 单独一个 log_std 参数(每维)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, h0)), nn.ELU(),
            layer_init(nn.Linear(h0, h1)), nn.ELU(),
            layer_init(nn.Linear(h1, h2)), nn.ELU(),
            layer_init(nn.Linear(h2, act_dim), std=0.01),
        )
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        # critic:obs -> value
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, h0)), nn.ELU(),
            layer_init(nn.Linear(h0, h1)), nn.ELU(),
            layer_init(nn.Linear(h1, h2)), nn.ELU(),
            layer_init(nn.Linear(h2, 1), std=1.0),
        )

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor(obs)
        std = self.actor_log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        if action is None:
            if deterministic:
                action = mean
            else:
                action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, entropy, value

    def forward_actor(self, obs: torch.Tensor) -> torch.Tensor:
        """导出 ONNX 时用(只 actor)。"""
        return self.actor(obs)


def compute_gae(
    rewards: torch.Tensor,        # (T, N)
    values: torch.Tensor,         # (T+1, N)  含 V(T)  bootstrap
    dones: torch.Tensor,          # (T, N)    done[t] 后是 t+1 的起点
    next_value: torch.Tensor,     # (N,)      V(T)
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE(Generalized Advantage Estimation)。"""
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(N, device=rewards.device)
    for t in reversed(range(T)):
        if t == T - 1:
            nextvalues = next_value
        else:
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * (1 - dones[t]) - values[t]
        lastgaelam = delta + gamma * gae_lambda * (1 - dones[t]) * lastgaelam
        advantages[t] = lastgaelam
    returns = advantages + values[:T]
    return advantages, returns


class PPONativeImpl(BaseRLAlgorithm):
    """自实现 PPO 训练器(零 Isaac 依赖,可纯 PyTorch 跑)。"""

    def __init__(self, env: BaseEnv, config: dict, log_dir: str = "logs/"):
        super().__init__(env, config, log_dir)
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        # 假设 env 暴露 num_obs / num_actions
        self.obs_dim = int(getattr(env, "num_obs", 33))
        self.act_dim = int(getattr(env, "num_actions", 31))
        # PPO 超参
        p = config.get("ppo", {})
        self.lr = float(p.get("learning_rate", 3e-4))
        self.gamma = float(p.get("gamma", 0.99))
        self.gae_lambda = float(p.get("gae_lambda", 0.95))
        self.clip_eps = float(p.get("clip_eps", 0.2))
        self.entropy_coef = float(p.get("entropy_coef", 0.005))
        self.vf_coef = float(p.get("vf_coef", 0.5))
        self.max_grad_norm = float(p.get("max_grad_norm", 0.5))
        self.update_epochs = int(p.get("update_epochs", 4))
        self.minibatch_size = int(p.get("minibatch_size", 256))
        self.num_steps_per_env = int(p.get("num_steps_per_env", 24))
        self.total_timesteps = int(p.get("total_timesteps", 1_000_000))
        self.num_envs = int(getattr(env, "num_envs", 1))

        self.ac = ActorCritic(self.obs_dim, self.act_dim).to(self.device)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=self.lr, eps=1e-5)
        self._checkpoint = {"ac": self.ac.state_dict(), "opt": self.opt.state_dict()}

    # ---------- 训练 ----------
    def train(self) -> None:
        env = self.env
        obs_buf = torch.zeros((self.num_steps_per_env, self.num_envs, self.obs_dim), device=self.device)
        actions_buf = torch.zeros((self.num_steps_per_env, self.num_envs, self.act_dim), device=self.device)
        logprobs_buf = torch.zeros((self.num_steps_per_env, self.num_envs), device=self.device)
        rewards_buf = torch.zeros((self.num_steps_per_env, self.num_envs), device=self.device)
        dones_buf = torch.zeros((self.num_steps_per_env, self.num_envs), device=self.device)
        values_buf = torch.zeros((self.num_steps_per_env, self.num_envs), device=self.device)

        # reset
        obs, _info = env.reset()
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        global_step = 0
        num_updates = max(1, self.total_timesteps // (self.num_steps_per_env * self.num_envs))

        for update in range(num_updates):
            # ---------- 1. rollout ----------
            for step in range(self.num_steps_per_env):
                with torch.no_grad():
                    action, logprob, _, value = self.ac.get_action_and_value(obs)
                # 跟 env 交互(简化:env.step 期望 numpy/cpu)
                action_np = action.cpu().numpy()
                next_obs, reward, done, info = env.step(action_np)
                next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
                reward_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
                done_t = torch.as_tensor(done, dtype=torch.float32, device=self.device)

                obs_buf[step] = obs
                actions_buf[step] = action
                logprobs_buf[step] = logprob
                rewards_buf[step] = reward_t
                dones_buf[step] = done_t
                values_buf[step] = value
                obs = next_obs
                global_step += self.num_envs

            # ---------- 2. bootstrap ----------
            with torch.no_grad():
                next_value = self.ac.get_value(obs)
            advantages, returns = compute_gae(
                rewards_buf, values_buf, dones_buf, next_value,
                self.gamma, self.gae_lambda,
            )

            # ---------- 3. flatten ----------
            b_obs = obs_buf.reshape((-1, self.obs_dim))
            b_actions = actions_buf.reshape((-1, self.act_dim))
            b_logprobs = logprobs_buf.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values_buf.reshape(-1)

            # ---------- 4. PPO update ----------
            b_inds = np.arange(len(b_obs))
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, len(b_obs), self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]
                    mb_inds_t = torch.as_tensor(mb_inds, dtype=torch.long, device=self.device)

                    _, newlogprob, entropy, newvalue = self.ac.get_action_and_value(
                        b_obs[mb_inds_t], b_actions[mb_inds_t]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds_t]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()

                    mb_advantages = b_advantages[mb_inds_t]
                    norm_adv = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue_clipped = b_values[mb_inds_t] + torch.clamp(
                        newvalue - b_values[mb_inds_t], -self.clip_eps, self.clip_eps,
                    )
                    v_loss_unclipped = (newvalue - b_returns[mb_inds_t]) ** 2
                    v_loss_clipped = (newvalue_clipped - b_returns[mb_inds_t]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.entropy_coef * entropy_loss + self.vf_coef * v_loss

                    self.opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                    self.opt.step()

                if approx_kl > 0.03:
                    break  # 提前停止

            if (update + 1) % 50 == 0:
                _log.info(f"update {update+1}/{num_updates} step={global_step} pg={pg_loss.item():.3f} v={v_loss.item():.3f}")

        # 训练完同步 checkpoint
        self._checkpoint = {"ac": self.ac.state_dict(), "opt": self.opt.state_dict()}

    # ---------- checkpoint ----------
    def save(self, path: str) -> None:
        torch.save(self._checkpoint, path)
        _log.info(f"checkpoint saved: {path}")

    def load(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.ac.load_state_dict(ck["ac"])
        self.opt.load_state_dict(ck["opt"])
        self._checkpoint = ck
        _log.info(f"checkpoint loaded: {path}")

    # ---------- ONNX 导出 ----------
    def export_onnx(self, out_path: str, opset: int = 17) -> None:
        self.ac.eval()
        dummy = torch.zeros(1, self.obs_dim, device=self.device)
        torch.onnx.export(
            self.ac.forward_actor, dummy, out_path,
            input_names=["obs"], output_names=["action"],
            dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
            opset_version=opset,
        )
        _log.info(f"ONNX exported: {out_path}")


__all__ = ["PPONativeImpl", "ActorCritic", "compute_gae", "layer_init"]
