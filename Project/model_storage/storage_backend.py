"""存储后端:本地文件系统 + S3/MinIO 抽象。

LocalStorageBackend 已完整实装。
S3StorageBackend 用 boto3 包装,需要 `pip install boto3`。
"""
from __future__ import annotations
import os
from abc import ABC, abstractmethod
from pathlib import Path


class StorageBackend(ABC):
    """存储后端抽象。"""

    @abstractmethod
    def put(self, key: str, data: bytes) -> Path:
        """存数据,返回本地路径(本地后端)或 S3 URI(云端后端)。"""
        ...

    @abstractmethod
    def get(self, key: str) -> bytes:
        """读数据。"""
        ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """删除(不存在时静默忽略)。"""
        ...

    @abstractmethod
    def list(self, prefix: str = "") -> list[str]:
        """列出 prefix 下的所有 key。"""
        ...


class LocalStorageBackend(StorageBackend):
    """本地文件系统后端(默认)。"""

    def __init__(self, root: str | os.PathLike = "models/"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # 防止 key 越出 root(key 不能含 ..)
        if ".." in key.split("/"):
            raise ValueError(f"unsafe key (contains '..'): {key}")
        return self.root / key

    def put(self, key: str, data: bytes) -> Path:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return p

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete(self, key: str) -> None:
        p = self._path(key)
        if p.exists():
            p.unlink()

    def list(self, prefix: str = "") -> list[str]:
        base = self.root / prefix if prefix else self.root
        if not base.exists():
            return []
        return sorted(
            str(p.relative_to(self.root)).replace(os.sep, "/")
            for p in base.rglob("*") if p.is_file()
        )


class S3StorageBackend(StorageBackend):
    """S3 / MinIO 后端(可选,需 boto3)。

    留 S3 兼容层,生产环境用。
    """

    def __init__(
        self,
        bucket: str,
        endpoint: str = "",
        access_key: str = "",
        secret_key: str = "",
        region: str = "us-east-1",
    ):
        self.bucket = bucket
        self.endpoint = endpoint
        self.region = region
        self._client = None
        self._access_key = access_key
        self._secret_key = secret_key
        # TODO: 初始化 boto3
        # import boto3
        # self._client = boto3.client(
        #     "s3",
        #     endpoint_url=endpoint or None,
        #     aws_access_key_id=access_key or None,
        #     aws_secret_access_key=secret_key or None,
        #     region_name=region,
        # )

    def _require_client(self):
        if self._client is None:
            raise NotImplementedError(
                "S3StorageBackend 需要 boto3:`pip install boto3` 然后取消 TODO。"
                f"当前 bucket={self.bucket!r} endpoint={self.endpoint!r}"
            )

    def put(self, key: str, data: bytes) -> Path:
        # TODO: self._client.put_object(Bucket=self.bucket, Key=key, Body=data)
        self._require_client()
        raise NotImplementedError

    def get(self, key: str) -> bytes:
        # TODO: obj = self._client.get_object(Bucket=self.bucket, Key=key); return obj["Body"].read()
        self._require_client()
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        # TODO: try self._client.head_object(Bucket=self.bucket, Key=key) except ClientError
        self._require_client()
        raise NotImplementedError

    def delete(self, key: str) -> None:
        # TODO: self._client.delete_object(Bucket=self.bucket, Key=key)
        self._require_client()
        raise NotImplementedError

    def list(self, prefix: str = "") -> list[str]:
        # TODO: paginator = self._client.get_paginator("list_objects_v2")
        #       for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
        #           yield obj["Key"]
        self._require_client()
        raise NotImplementedError
