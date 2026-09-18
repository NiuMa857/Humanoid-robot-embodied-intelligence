"""model_storage 单元测试。"""
import sys
sys.path.insert(0, ".")
import os
import tempfile

from model_storage import (
    LocalStorageBackend, ModelRegistry, VersionMetadata,
    VersionManager, parse_semver, cmp_semver,
)


def test_local_storage_crud():
    with tempfile.TemporaryDirectory() as d:
        be = LocalStorageBackend(root=d)
        path = be.put("ppo/v0.1.0/model.onnx", b"x")
        assert path.exists() and be.exists("ppo/v0.1.0/model.onnx")
        assert be.get("ppo/v0.1.0/model.onnx") == b"x"
        be.delete("ppo/v0.1.0/model.onnx")
        assert not be.exists("ppo/v0.1.0/model.onnx")


def test_local_storage_list():
    with tempfile.TemporaryDirectory() as d:
        be = LocalStorageBackend(root=d)
        be.put("a/b/c.onnx", b"x")
        be.put("a/d.pt", b"y")
        keys = be.list("a")
        assert "a/b/c.onnx" in keys
        assert "a/d.pt" in keys


def test_model_registry_register_and_retrieve():
    with tempfile.TemporaryDirectory() as d:
        be = LocalStorageBackend(root=d)
        reg = ModelRegistry(be)
        reg.register("walker", "0.1.0", b"bytes1", fmt="onnx",
                     input_shape=[1, 33], output_shape=[1, 31])
        reg.register("walker", "0.1.1", b"bytes2", fmt="onnx")
        assert reg.get("walker", "latest").version == "0.1.1"
        assert reg.get_bytes("walker", "0.1.0") == b"bytes1"


def test_model_registry_delete():
    with tempfile.TemporaryDirectory() as d:
        be = LocalStorageBackend(root=d)
        reg = ModelRegistry(be)
        reg.register("walker", "0.1.0", b"bytes1", fmt="onnx")
        reg.delete("walker", "0.1.0")
        assert reg.get("walker", "0.1.0") is None


def test_semver_compare():
    assert cmp_semver("0.1.0", "0.1.1") == -1
    assert cmp_semver("1.0.0", "0.99.99") == 1
    assert cmp_semver("1.0.0-alpha", "1.0.0") == -1
    assert cmp_semver("2.0.0", "2.0.0") == 0


def test_version_manager_bump():
    vm = VersionManager("1.0.0")
    assert vm.bump("patch") == "1.0.1"
    assert vm.bump("minor") == "1.1.0"
    assert vm.bump("major") == "2.0.0"
