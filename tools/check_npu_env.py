"""Check the Ascend runtime without constructing the full CROG model."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from utils.npu import get_torch_npu, set_device


def main() -> int:
    adapter = get_torch_npu()
    device = set_device(0)
    print(f"torch={torch.__version__}")
    print(f"torch_npu={getattr(adapter, '__version__', '<unknown>')}")
    print(f"device={device}")
    print(f"ASCEND_HOME_PATH={os.environ.get('ASCEND_HOME_PATH', '<unset>')}")
    print(
        "HCCL available="
        f"{getattr(torch.distributed, 'is_hccl_available', lambda: False)()}"
    )
    left = torch.randn(64, 64, device=device)
    right = torch.randn(64, 64, device=device)
    result = left @ right
    adapter.npu.synchronize()
    print(f"NPU matmul OK: shape={tuple(result.shape)} dtype={result.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
