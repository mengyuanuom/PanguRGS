from .crog import CROG
from .pangu_crog import PanguCROG
from .pangu_ssg import PanguSSG
from .ssg import SSG
from loguru import logger


def _build_crog_family(model_class, args):
    model = model_class(args)
    backbone = []
    head = []
    for k, v in model.named_parameters():
        if k.startswith('backbone') and 'positional_embedding' not in k:
            backbone.append(v)
        else:
            head.append(v)
    logger.info('Backbone with decay={}, Head={}'.format(len(backbone), len(head)))
    param_list = [{
        'params': backbone,
        'initial_lr': args.lr_multi * args.base_lr
    }, {
        'params': head,
        'initial_lr': args.base_lr
    }]
    return model, param_list


def build_crog(args):
    return _build_crog_family(CROG, args)


def build_pangu_crog(args):
    return _build_crog_family(PanguCROG, args)


def _build_drog_family(model_class, args):
    model = model_class(args)
    backbone = []
    head = []
    frozen = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            frozen.append(parameter)
        elif name.startswith(("txt_backbone", "dinov2")):
            backbone.append(parameter)
        else:
            head.append(parameter)
    logger.info(
        "{}: Backbone={}, Head={}, Frozen={}".format(
            model_class.__name__, len(backbone), len(head), len(frozen)
        )
    )
    param_list = [
        {
            "params": backbone,
            "initial_lr": args.lr_multi * args.base_lr,
        },
        {
            "params": head,
            "initial_lr": args.base_lr,
        },
    ]
    return model, param_list


def build_drog(args):
    from .drog import DROG

    return _build_drog_family(DROG, args)


def build_pangu_drog(args):
    from .pangu_drog import PanguDROG

    return _build_drog_family(PanguDROG, args)


def build_drogoff(args):
    from .drogoff import DROGOFF

    return _build_drog_family(DROGOFF, args)


def build_pangu_drogoff(args):
    from .pangu_drogoff import PanguDROGOFF

    return _build_drog_family(PanguDROGOFF, args)


def _build_toolrgs_family(args):
    from .toolrgs import build_toolrgs_model

    architecture = str(args.architecture).strip().lower()
    model = build_toolrgs_model(architecture, args)
    backbone = []
    head = []
    frozen = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            frozen.append(parameter)
        elif name.startswith(
            ("backbone", "bridger", "txt_backbone", "dinov2")
        ) and "positional_embedding" not in name:
            backbone.append(parameter)
        else:
            head.append(parameter)
    logger.info(
        "{}: Backbone={}, Head={}, Frozen={}".format(
            type(model).__name__,
            len(backbone),
            len(head),
            len(frozen),
        )
    )
    return model, [
        {
            "params": backbone,
            "lr": args.lr_multi * args.base_lr,
            "initial_lr": args.lr_multi * args.base_lr,
        },
        {
            "params": head,
            "lr": args.base_lr,
            "initial_lr": args.base_lr,
        },
    ]


def build_model(args):
    """Select either the legacy or Pangu-prefixed architecture namespace."""
    architecture = str(getattr(args, "architecture", "crog")).lower()
    builders = {
        "crog": build_crog,
        "pangu_crog": build_pangu_crog,
        "drog": build_drog,
        "pangu_drog": build_pangu_drog,
        "drogoff": build_drogoff,
        "pangu_drogoff": build_pangu_drogoff,
    }
    if architecture in builders:
        return builders[architecture](args)
    toolrgs_models = {
        "crogoff",
        "pangu_crogoff",
        "etrg",
        "etrg_rgb",
        "pangu_etrg",
        "ggcnnclip",
        "ggcnn_clip",
        "pangu_ggcnnclip",
        "grconvnetclip",
        "grconvnet_clip",
        "pangu_grconvnetclip",
        "graspmamba",
        "grasp_mamba",
        "pangu_graspmamba",
        "lgd",
        "pangu_lgd",
        "maplegrasp",
        "maple_grasp",
        "pangu_maplegrasp",
    }
    if architecture in toolrgs_models:
        return _build_toolrgs_family(args)
    choices = ", ".join(sorted(set(builders) | toolrgs_models))
    raise ValueError(
        f"Unknown MODEL.architecture {architecture!r}; choose one of: {choices}"
    )


def build_ssg(args):
    model = SSG(args)
    return model, model.parameters()


def build_pangu_ssg(args):
    model = PanguSSG(args)
    return model, model.parameters()
