import torch.nn as nn

from .drog_layers import MultiTaskProjector, OffsetMultiTaskProjector
from .drog_layers import FiLMProjector

def build_projector(cfg, with_offset=False) -> nn.Module:
    """
    根据 config 里 MODEL.PROJECTOR.TYPE 构建不同的 head：
      - "dynamic" -> MultiTaskProjector (你现在的版本)
      - "film"    -> FiLMProjector (FiLM 版本)
    """
    proj_cfg = cfg.projector # 你按自己 config 结构改
    vis_dim  = cfg.vis_dim
    word_dim = cfg.word_dim

    proj_type   = proj_cfg.lower()
    if proj_type == "dynamic":
        projector_cls = OffsetMultiTaskProjector if with_offset else MultiTaskProjector
        projector_kwargs = {
            "word_dim": word_dim,
            "in_dim": vis_dim // 2,
            "kernel_size": 3,
        }
        if with_offset:
            projector_kwargs["with_short_side"] = bool(
                getattr(cfg, "predict_grasp_short_side", False)
            )
        projector = projector_cls(**projector_kwargs)
    elif proj_type == "film" and not with_offset:
        projector = FiLMProjector(
            word_dim=word_dim,
            in_dim=vis_dim//2,
            kernel_size=3
        )
    else:
        suffix = " with offset" if with_offset else ""
        raise ValueError(f"Unknown projector type {proj_type!r}{suffix}")

    print(f"[build_projector] Use projector type: {proj_type}, "
          f"word_dim={word_dim}, in_dim={vis_dim//2}, k={3}")
    return projector
