"""Batch 1 smoke test - common + model_storage."""
import sys
sys.path.insert(0, '.')
import logging
logging.disable(logging.CRITICAL)  # 关闭 JSON log 干扰

from common.config import load_config
from model_storage import LocalStorageBackend, ModelRegistry, cmp_semver
from model_storage.version_manager import VersionMetadata
import tempfile

cfg = load_config('default')
print('config keys:', list(cfg.keys()))

with tempfile.TemporaryDirectory() as d:
    be = LocalStorageBackend(root=d)
    be.put('ppo/v0.1.0/model.onnx', b'fake-onnx')
    print('local storage OK')

    reg = ModelRegistry(be)
    e1 = reg.register('walker_ppo', '0.1.0', b'onnx-bytes', fmt='onnx',
                      input_shape=[1, 33], output_shape=[1, 31],
                      metadata=VersionMetadata(semver='0.1.0', metrics={'sr': 0.8}))
    e2 = reg.register('walker_ppo', '0.1.1', b'onnx-bytes-v2', fmt='onnx',
                      metadata=VersionMetadata(semver='0.1.1', metrics={'sr': 0.85}))
    assert reg.get('walker_ppo', 'latest').version == '0.1.1'
    assert reg.get_bytes('walker_ppo', '0.1.0') == b'onnx-bytes'
    print('registry OK, versions:', reg.list_versions('walker_ppo'))

print('semver 0.1.0 vs 0.1.1:', cmp_semver('0.1.0', '0.1.1'))
print('semver 1.0.0-alpha vs 1.0.0:', cmp_semver('1.0.0-alpha', '1.0.0'))
print('ALL BATCH-1 TESTS PASSED')
