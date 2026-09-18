# MISSING_DATA — 阻塞 100% 实装的待提供数据

> 本文档列出 Project/ 里所有**通用代码已实装**、但仍需用户提供关键数据才能 100% 跑通的项。
> 按优先级排序。

## A. 阻塞实机部署(M3 之前必须有)

| # | 数据 | 用途 | 影响文件 |
|---|---|---|---|
| A1 | **LimX Oli URDF**(或 MJCF/USD) | 关节名 / 拓扑 / 限位 / 质量 / 惯性 | `proprioception/state_estimator.py`(KinematicsBackend)、`motion_control/{mpc,wbc}.py`、训练 env |
| A2 | **LimX Oli SDK API 文档 + Python 绑定** | 关节驱动 / IMU / 力矩读取 | `proprioception/{imu_reader,joint_state}.py`、`motion_control/low_level_servo.py`(待建) |
| A3 | **关节默认站姿**(`default_joint_pos`) | 部署缓动目标 | `motion_control/wbc.py`、`deployment/`(待建) |
| A4 | **关节 PD 刚度/阻尼**(`kps` / `kds`) | 真机 PD 控制 | `motion_control/{wbc,low_level_servo}.py` |
| A5 | **关节力矩限位**(per-joint,不是单一 max_torque_nm) | L1 安全 | `proprioception/joint_state.py`、`motion_control/feedback.py` |
| A6 | **相机-机体系外参**(头相机→base, 胸相机→base) | 场景图坐标变换 | `environment_perception/scene_graph.py` |
| A7 | **IMU 型号 / 通信协议** / 接口库 | 真实模式读取 | `proprioception/imu_reader.py` |

## B. 阻塞训练跑通(M2 之前必须有)

| # | 数据 | 用途 | 影响文件 |
|---|---|---|---|
| B1 | Oli 的关节维度、限位 → **填到 `configs/default.yaml`** | env 构造 | `model_training/envs/{stair,industrial}_env.py`(待建) |
| B2 | 楼梯 / 工作台 USD 资产 | Isaac Sim 场景 | `simulation/isaac_sim/scene_loader.py` |
| B3 | 训练超参(actor/critic 网络结构、PPO 超参) | 实跑训练 | `model_training/algorithms/ppo_native.py`、`configs/*.yaml` |

## C. 阻塞规划层(M1/M2)

| # | 数据 | 用途 | 影响文件 |
|---|---|---|---|
| C1 | **LLM endpoint** + API key | 调云端/本地 LLM | `configs/default.yaml` 里 `planning.llm.endpoint / api_key`;或环境变量 `OLI_LLM_API_KEY` |
| C2 | VLM 是否启用 / 哪家 | 场景理解 | `configs/default.yaml` 里 `planning.vlm.*` |
| C3 | LLM system prompt(指令理解/任务分解/重规划) | LLM 输出质量 | `task_planning/llm_client.py` 里 hard-coded 的 system prompt |

## D. 软依赖(可选)

| # | 数据 | 用途 | 影响 |
|---|---|---|---|
| D1 | 是否需要 Git LFS(训练策略 .pt / .onnx) | 模型版本管理 | 若需要,在 `.gitattributes` 配 LFS |
| D2 | TensorRT / onnxruntime GPU backend | 加速推理 | 当前默认 onnxruntime CPU,RK3588 没 GPU |
| D3 | ONNX 模型来源 | 物体检测器 | 自训 YOLO / 用公开模型 / 用 VLM |

## E. 当前实装状态

| 模块 | 通用代码 | Oli 特定实装 |
|---|---|---|
| `common/` | ✅ 100% | n/a |
| `model_storage/` | ✅ 100% | n/a |
| `proprioception/state_estimator.py` | ✅ 通用融合逻辑 | ⏳ 等 A1 注入 KinematicsBackend |
| `proprioception/imu_reader.py` | ✅ mock + 校准逻辑 | ⏳ 等 A2/A7 接入 Oli SDK |
| `proprioception/joint_state.py` | ✅ mock + L1 限位 | ⏳ 等 A2/A4/A5 接入 Oli SDK |
| `environment_perception/rgbd_camera.py` | ✅ RealSense 抽象 | ⏳ 等 A2 配置真机 serial |
| `environment_perception/object_detector.py` | ✅ ONNX + Mock | ⏳ 等 D3 选型 |
| `environment_perception/scene_graph.py` | ✅ NMS + 坐标变换 | ⏳ 等 A6 填外参 |
| `task_planning/llm_client.py` | ✅ Remote/Local 双接口 | ⏳ 等 C1/C3 配 endpoint + prompt |
| `task_planning/{task_decomposer,replanner}.py` | ✅ fallback 模板 | 跑通,LLM 接上效果更好 |
| `motion_control/feedback.py` | ✅ L1-L5 完整 | 默认阈值即可用 |
| `motion_control/{mpc,wbc,low_level_servo}.py` | ✅ 零阶 fallback | ⏳ 等 A1/A3/A4 实装真实控制 |
| `motion_control/skill_executor.py` | ✅ 状态机 | n/a |
| `simulation/sim_interface.py` | ✅ Mock + 抽象 | ⏳ Isaac 特定(见 B) |
| `simulation/isaac_sim/` | 占位 | ⏳ 等 B1/B2 |
| `model_training/envs/base_env.py` | ✅ 抽象 | ⏳ 等 B1 |
| `model_training/mdp/{rewards,terminations,events}.py` | ✅ 完整 | 调权重即可 |
| `model_training/algorithms/ppo_native.py` | ✅ CleanRL 完整 | ⏳ 需要 `pip install torch` |
| `model_training/algorithms/ppo_rsl_rl.py` | 占位 | ⏳ 装 rsl_rl 后启用 |
| `tests/` | ✅ 17/17 通过 | n/a |

## F. 怎么给我这些数据

最快的几种方式:
- A1 (URDF): 直接放到 `Project/oli_hardware/urdf/` 下,告诉我文件名
- A2 (SDK): 安装 Python 包,然后告诉我 import 路径
- C1 (LLM endpoint): 在 PowerShell 里 `setx OLI_LLM_API_KEY <key>`,然后我改 config
- D3 (检测模型): 放 `.onnx` 到 `Project/models/` 下,告诉我文件名

或者:`pip install <package>` 让 Project 自己能 import,我帮你接进去。

## G. 验证:我能跑什么

不需要任何上面数据就能跑的:
- ✅ `python _run_all_tests.py`(17/17 测试通过)
- ✅ mock 模式下的所有感知 / 本体感知 / 规划 / 控制流程
- ✅ 模型存储 / 配置加载 / 日志 / 半完整测试
- ✅ PPO 接口完整,源码 token 验证通过(但真训练需要 torch)

需要数据才能跑的:
- 真机部署 → A1-A7
- 真训练 → B1-B3 + torch
- 真实 LLM 调用 → C1
- 真实检测 → D3