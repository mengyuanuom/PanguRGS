"""ToolRGS variants with legacy and Pangu-prefixed public names."""

from importlib import import_module


MODEL_REGISTRY = {
    "etrg": ("etrg", "ETRG"),
    "etrg_rgb": ("etrg", "ETRG"),
    "pangu_etrg": ("pangu_etrg", "PanguETRG"),
    "crogoff": ("crogoff", "CROGOFF"),
    "pangu_crogoff": ("pangu_crogoff", "PanguCROGOFF"),
    "ggcnnclip": ("ggcnnclip", "GGCNN_CLIP"),
    "ggcnn_clip": ("ggcnnclip", "GGCNN_CLIP"),
    "pangu_ggcnnclip": ("pangu_ggcnnclip", "PanguGGCNNCLIP"),
    "grconvnetclip": ("grconvnetclip", "GenerativeResnet_CLIP"),
    "grconvnet_clip": ("grconvnetclip", "GenerativeResnet_CLIP"),
    "pangu_grconvnetclip": (
        "pangu_grconvnetclip",
        "PanguGRConvNetCLIP",
    ),
    "graspmamba": ("graspmamba", "GraspMamba"),
    "grasp_mamba": ("graspmamba", "GraspMamba"),
    "pangu_graspmamba": ("pangu_graspmamba", "PanguGraspMamba"),
    "lgd": ("lgd", "LGD"),
    "pangu_lgd": ("pangu_lgd", "PanguLGD"),
    "maplegrasp": ("maplegrasp", "MapleGrasp"),
    "maple_grasp": ("maplegrasp", "MapleGrasp"),
    "pangu_maplegrasp": ("pangu_maplegrasp", "PanguMapleGrasp"),
}


def build_toolrgs_model(name, cfg):
    normalized = str(name).strip().lower()
    try:
        module_name, class_name = MODEL_REGISTRY[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown ToolRGS model {name!r}; available: {available}"
        ) from exc
    module = import_module(f"{__name__}.{module_name}")
    return getattr(module, class_name)(cfg)
