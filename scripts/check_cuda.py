from __future__ import annotations

import sys

try:
    import torch
except ImportError:
    print("PyTorch is not installed in this environment.", file=sys.stderr)
    raise SystemExit(1)

print(f"torch={torch.__version__}")
print(f"torch_cuda_runtime={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Check the NVIDIA driver and installed PyTorch wheel.")
print(f"device={torch.cuda.get_device_name(0)}")
a = torch.randn((1024, 1024), device="cuda")
b = a @ a.T
checksum = float(b.mean().item())
if not torch.isfinite(b).all():
    raise SystemExit("CUDA tensor operation produced non-finite output")
print(f"cuda_matrix_checksum={checksum:.6f}")
