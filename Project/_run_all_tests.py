"""统一测试入口(不需要 pytest)。

跑 Project/tests/ 下所有 test_*.py,报告通过/失败。
"""
import sys
import os
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
TESTS_DIR = PROJECT_ROOT / "tests"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def discover_tests():
    """发现所有 test_*.py。"""
    tests = []
    for p in TESTS_DIR.rglob("test_*.py"):
        # 跳过 conftest 和 __init__
        if p.name in ("conftest.py", "__init__.py"):
            continue
        if "sim_e2e" in p.parts:
            # sim_e2e 默认 skip
            continue
        tests.append(p)
    return sorted(tests)


def run_one(test_path: Path) -> tuple[bool, str]:
    """跑单个测试文件。"""
    # 把 test_path 写成模块名(相对 tests/ 父目录)
    rel = test_path.relative_to(PROJECT_ROOT).with_suffix("")
    mod_name = str(rel).replace(os.sep, ".")
    try:
        mod = __import__(mod_name, fromlist=["*"])
    except Exception as e:
        return False, f"IMPORT FAILED: {type(e).__name__}: {e}"

    # 找所有以 test_ 开头的 callable
    test_funcs = [
        (name, getattr(mod, name))
        for name in dir(mod)
        if name.startswith("test_") and callable(getattr(mod, name))
    ]
    if not test_funcs:
        return False, "NO TEST FUNCTIONS FOUND"

    passed = []
    failed = []
    for name, fn in test_funcs:
        t0 = time.time()
        try:
            fn()
            dt = (time.time() - t0) * 1000
            passed.append((name, dt))
        except Exception as e:
            dt = (time.time() - t0) * 1000
            tb = traceback.format_exc(limit=2)
            failed.append((name, type(e).__name__, str(e)[:100], dt, tb))

    summary = f"  passed: {len(passed)}, failed: {len(failed)}"
    if failed:
        details = "\n    " + "\n    ".join(
            f"{n}: {etype}: {emsg} ({dt:.0f}ms)" for n, etype, emsg, dt, _ in failed
        )
        return False, summary + details
    return True, summary + f" ({sum(d for _, d in passed):.0f}ms total)"


def main():
    tests = discover_tests()
    print(f"discovered {len(tests)} test files\n")

    passed_files = []
    failed_files = []

    for p in tests:
        rel = p.relative_to(PROJECT_ROOT)
        print(f"--- {rel} ---")
        ok, msg = run_one(p)
        if ok:
            passed_files.append(rel)
            print(f"  PASS {msg}")
        else:
            failed_files.append(rel)
            print(f"  FAIL {msg}")
        print()

    # 总结
    print("=" * 60)
    print(f"summary: {len(passed_files)}/{len(tests)} test files passed")
    if failed_files:
        print("failed files:")
        for f in failed_files:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
