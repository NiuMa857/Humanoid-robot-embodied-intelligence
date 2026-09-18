"""pytest 共用 fixtures / 配置。

为 Project 内的测试提供 sys.path 注入(让 `from common.config import ...` 这种绝对导入能工作)。
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
