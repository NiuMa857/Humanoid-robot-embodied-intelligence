"""Check pytest."""
try:
    import pytest
    print("pytest:", pytest.__version__)
except ImportError:
    import subprocess, sys
    print("installing pytest...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "pytest"])
    import pytest
    print("pytest installed:", pytest.__version__)
