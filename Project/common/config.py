"""统一配置加载(YAML + 环境变量覆盖)。

特性:
  - YAML 加载(用 PyYAML)
  - 环境变量覆盖:支持 OLI_KEY__SUBKEY 格式自动展开
  - 缓存(避免重复 IO)
  - 容错:找不到时返回空 dict
"""
from __future__ import annotations
import os
import threading
from pathlib import Path
from typing import Any

try:
    import yaml  # PyYAML
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CFG_PATH = _PROJECT_ROOT / "configs" / "default.yaml"

_cache: dict[str, Any] = {}
_lock = threading.Lock()


def load_config(name: str = "default", path: str | os.PathLike | None = None) -> dict[str, Any]:
    """加载 YAML 配置(带缓存)。

    Args:
        name: 配置名(对应 configs/<name>.yaml),默认 "default"。
        path: 显式路径,优先级最高(覆盖 name)。

    Returns:
        dict;文件不存在时返回 {}。
    """
    with _lock:
        if name in _cache and path is None:
            return _cache[name]

    if path is not None:
        cfg_path = Path(path)
    else:
        cfg_path = _PROJECT_ROOT / "configs" / f"{name}.yaml"

    if not cfg_path.exists():
        cfg: dict[str, Any] = {}
    else:
        if yaml is None:
            raise RuntimeError("PyYAML not installed; run `pip install pyyaml`")
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    # 环境变量覆盖
    cfg = env_override(cfg, prefix="OLI_")

    with _lock:
        _cache[name] = cfg
    return cfg


def env_override(cfg: dict[str, Any], prefix: str = "OLI_") -> dict[str, Any]:
    """用环境变量覆盖配置。

    约定:环境变量 `OLI_<KEY1>__<KEY2>__<KEY3>` 自动展开到 `cfg[key1][key2][key3]`。
    例:OLI_LLM__ENDPOINT → cfg["llm"]["endpoint"]

    转换规则:
      - 字符串 "true" / "false" → bool
      - 字符串数字 → float / int
      - 字符串列表(逗号分隔) → list
      - 其余保持 str
    """
    out = _deepcopy(cfg)
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        keys = env_key[len(prefix):].lower().split("__")
        coerced = _coerce(env_val)
        _set_nested(out, keys, coerced)
    return out


def _coerce(v: str) -> Any:
    """把字符串值转成合适的 Python 类型。"""
    if v.lower() in ("true", "yes", "1"):
        return True
    if v.lower() in ("false", "no", "0"):
        return False
    # 试 int / float
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    # 逗号分隔列表
    if "," in v:
        return [_coerce(x) for x in v.split(",")]
    return v


def _set_nested(d: dict[str, Any], keys: list[str], val: Any) -> dict[str, Any]:
    """按 keys 链把 val 写到嵌套 dict 里的最后一级,返回最外层 d。"""
    root = d
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = val
    return root


def _deepcopy(d: Any) -> Any:
    """轻量 deepcopy(避免引入 copy 模块依赖,只支持 dict/list/scalar)。"""
    if isinstance(d, dict):
        return {k: _deepcopy(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_deepcopy(x) for x in d]
    return d


def clear_cache() -> None:
    """清空配置缓存(测试用)。"""
    with _lock:
        _cache.clear()


def get_project_root() -> Path:
    """返回 Project/ 根目录。"""
    return _PROJECT_ROOT


__all__ = ["load_config", "env_override", "clear_cache", "get_project_root"]
