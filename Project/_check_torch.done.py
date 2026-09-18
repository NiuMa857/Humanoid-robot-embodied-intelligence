"""Install torch if missing, then verify."""
try:
    import torch
    print("torch already installed:", torch.__version__)
except ImportError:
    import subprocess, sys
    print("torch not found, installing (CPU)...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "torch", "--index-url", "https://download.pytorch.org/whl/cpu",
    ])
    import torch
    print("torch installed:", torch.__version__)
