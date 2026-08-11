"""Small, explicit Ascend runtime used by the official CROG training code.

The port intentionally does not use ``torch_npu.contrib.transfer_to_npu``:
CUDA calls are replaced explicitly so device mistakes remain visible.
"""

from contextlib import nullcontext
from typing import Optional

import torch


try:
    import torch_npu  # type: ignore
except Exception as exc:
    torch_npu = None
    _IMPORT_ERROR: Optional[BaseException] = exc
else:
    _IMPORT_ERROR = None


def get_torch_npu():
    if torch_npu is None:
        raise RuntimeError(
            "torch_npu is unavailable. Install the torch/torch_npu pair matching "
            "the server CANN release and source the CANN set_env.sh first."
        ) from _IMPORT_ERROR
    return torch_npu


def require_npu() -> None:
    adapter = get_torch_npu()
    if not adapter.npu.is_available():
        raise RuntimeError(
            "torch_npu imported, but no Ascend NPU is available. Check "
            "npu-smi info, ASCEND_RT_VISIBLE_DEVICES, and the CANN environment."
        )


def device_count() -> int:
    require_npu()
    return int(get_torch_npu().npu.device_count())


def set_device(index: int) -> torch.device:
    require_npu()
    index = int(index)
    get_torch_npu().npu.set_device(f"npu:{index}")
    return torch.device(f"npu:{index}")


def autocast(enabled: bool = True):
    if not enabled:
        return nullcontext()
    return get_torch_npu().npu.amp.autocast(enabled=True)


class NoOpGradScaler:
    """FP32 optimizer path that never enters torch_npu AMP overflow checks."""

    enabled = False

    @staticmethod
    def scale(loss):
        return loss

    @staticmethod
    def unscale_(optimizer):
        return None

    @staticmethod
    def step(optimizer):
        return optimizer.step()

    @staticmethod
    def update():
        return None


def build_grad_scaler(enabled: bool = True):
    if not enabled:
        return NoOpGradScaler()
    return get_torch_npu().npu.amp.GradScaler(enabled=True)


def empty_cache() -> None:
    get_torch_npu().npu.empty_cache()


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    get_torch_npu().npu.manual_seed_all(seed)
