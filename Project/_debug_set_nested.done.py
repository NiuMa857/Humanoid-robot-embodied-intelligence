"""Debug _set_nested."""
import sys
sys.path.insert(0, '.')
from common.config import _set_nested

# 测试 1:简单 2 级
out1 = _set_nested({}, ["test", "val"], 123)
print('test1:', out1, type(out1))

# 测试 2:3 级
out2 = _set_nested({}, ["a", "b", "c"], "x")
print('test2:', out2, type(out2))
