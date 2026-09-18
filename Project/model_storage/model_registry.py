"""模型注册表:按"模型名 + 版本"管理 ONNX / TorchScript。

manifest 用 JSON 存于 `models/registry.json`,记录每个模型的所有历史版本。
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .storage_backend import StorageBackend, LocalStorageBackend
from .version_manager import VersionMetadata


@dataclass
class ModelEntry:
    name: str
    version: str
    key: str                      # 在 storage backend 里的 key,例 "ppo/v0.1.0/model.onnx"
    format: str = "onnx"          # onnx / torchscript / pt
    size_bytes: int = 0
    input_shape: list = field(default_factory=list)   # e.g. [1, 33]
    output_shape: list = field(default_factory=list)  # e.g. [1, 31]
    metadata: VersionMetadata = field(default_factory=VersionMetadata)
    created_at: str = ""


class ModelRegistry:
    """模型注册表。"""

    def __init__(self, storage: StorageBackend, manifest_key: str = "registry.json"):
        self.storage = storage
        self.manifest_key = manifest_key
        self._manifest: dict[str, list[ModelEntry]] = {}
        self._load()

    # ---------- 内部:manifest 加载/保存 ----------

    def _load(self) -> None:
        if not self.storage.exists(self.manifest_key):
            self._manifest = {}
            return
        try:
            raw = json.loads(self.storage.get(self.manifest_key).decode("utf-8"))
            self._manifest = {
                name: [ModelEntry(**e) for e in entries]
                for name, entries in raw.items()
            }
        except (json.JSONDecodeError, TypeError) as e:
            raise RuntimeError(f"corrupt registry manifest: {e}") from e

    def _save(self) -> None:
        raw = {
            name: [asdict(e) for e in entries]
            for name, entries in self._manifest.items()
        }
        self.storage.put(self.manifest_key, json.dumps(raw, indent=2, ensure_ascii=False).encode("utf-8"))

    # ---------- CRUD ----------

    def register(
        self,
        name: str,
        version: str,
        model_bytes: bytes,
        fmt: str = "onnx",
        input_shape: list | None = None,
        output_shape: list | None = None,
        metadata: VersionMetadata | None = None,
    ) -> ModelEntry:
        """上传新模型 + 写入 manifest。"""
        key = f"{name}/{version}/model.{fmt}"
        self.storage.put(key, model_bytes)
        entry = ModelEntry(
            name=name,
            version=version,
            key=key,
            format=fmt,
            size_bytes=len(model_bytes),
            input_shape=input_shape or [],
            output_shape=output_shape or [],
            metadata=metadata or VersionMetadata(semver=version, trained_at=VersionMetadata.now()),
            created_at=VersionMetadata.now(),
        )
        self._manifest.setdefault(name, []).append(entry)
        self._save()
        return entry

    def get(self, name: str, version: str = "latest") -> ModelEntry | None:
        """获取 entry,version="latest" 解析到最新版本(按 semver 字符串排序,简化用字符串比较)。"""
        entries = self._manifest.get(name, [])
        if not entries:
            return None
        if version == "latest":
            return self._latest(entries)
        for e in entries:
            if e.version == version:
                return e
        return None

    def get_bytes(self, name: str, version: str = "latest") -> bytes:
        """便捷:获取模型文件字节。"""
        entry = self.get(name, version)
        if entry is None:
            raise KeyError(f"model {name}@{version} not found")
        return self.storage.get(entry.key)

    def list_models(self) -> list[str]:
        return list(self._manifest.keys())

    def list_versions(self, name: str) -> list[str]:
        return [e.version for e in self._manifest.get(name, [])]

    def delete(self, name: str, version: str) -> None:
        """删除指定版本(从 manifest + storage 一起删)。"""
        entries = self._manifest.get(name, [])
        for i, e in enumerate(entries):
            if e.version == version:
                self.storage.delete(e.key)
                entries.pop(i)
                if not entries:
                    del self._manifest[name]
                self._save()
                return
        raise KeyError(f"model {name}@{version} not found")

    # ---------- 辅助 ----------

    @staticmethod
    def _latest(entries: list[ModelEntry]) -> ModelEntry:
        """挑 latest:按 created_at ISO 字符串排序(字符串可比较,新版本后加)。"""
        return max(entries, key=lambda e: e.created_at)


__all__ = ["ModelRegistry", "ModelEntry"]
