"""Isaac Sim + Isaac Lab 场景加载器。

需装 NVIDIA Isaac Sim + Isaac Lab:
  pip install isaaclab

此处给接口骨架 + 留 Isaac 特定实现点。
"""
from __future__ import annotations
from common.logging import setup_logger

_log = setup_logger("scene_loader")


class IsaacSceneLoader:
    """加载 Oli + 楼梯/工作台 USD 场景到 Isaac Sim。

    使用前需要:
      1. 安装 Isaac Sim
      2. 把 Oli URDF 转为 USD(用 Isaac Lab 提供的 urdf2usd 工具)
      3. 准备楼梯 / 工作台 USD(从 Isaac Lab Assets 下载或自制)
    """

    def __init__(self, sim_app=None, config: dict | None = None):
        self.sim_app = sim_app
        self.config = config or {}
        self._scene = None

    def load_stair_scene(self, angle_deg: float = 30.0) -> None:
        """加载楼梯场景。

        实装:
          from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
          # 用 TerrainGenerator 生成楼梯
        """
        _log.info(f"loading stair scene, angle={angle_deg}° (placeholder)")
        # TODO:
        # terrain_cfg = HfRandomUniformTerrainCfg(...)
        # scene = InteractiveSceneCfg(terrain=terrain_cfg)

    def load_industrial_scene(self) -> None:
        """加载工业工作台(搬运 / 装配)。"""
        _log.info("loading industrial scene (placeholder)")
        # TODO

    def get_ground_truth_objects(self) -> list[dict]:
        """从 Isaac 拉取所有物体的 ground truth 位姿(用于评估 / sim2real 验证)。"""
        # TODO: 用 Isaac Lab 的 SceneEntityCfg
        return []


__all__ = ["IsaacSceneLoader"]
