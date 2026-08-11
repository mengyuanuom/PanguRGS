"""Dataset selection shared by CROG-NPU training and evaluation."""

from .dataset import OCIDVLGDataset
from .grasp_tool_dataset import GraspToolDataset
from .vcot_dataset import VCoTDataset


def normalize_dataset_name(name):
    return str(name or "OCID-VLG").strip().lower().replace("_", "-")


def build_referring_grasp_dataset(args, split, with_grasp_offset=False):
    """Build a supported referring-grasp dataset with the common batch contract."""
    dataset_name = normalize_dataset_name(getattr(args, "dataset", "OCID-VLG"))
    common = {
        "root_dir": args.root_path,
        "input_size": args.input_size,
        "word_length": args.word_len,
        "split": split,
    }
    if dataset_name in {"vcot", "vcot-grasp", "grasp-anything"}:
        return VCoTDataset(
            **common,
            split_root=getattr(args, "split_root", None),
            prompt_template=getattr(
                args, "prompt_template", "Grasp the {object_name}"
            ),
            with_offset=with_grasp_offset,
            offset_radius=float(getattr(args, "offset_r", 20.0)),
            offset_sigma=getattr(args, "offset_sigma", None),
            grasp_size_factor=float(getattr(args, "grasp_size_factor", 100.0)),
        )
    if dataset_name in {"grasptool", "grasp-tool", "grasp-tools"}:
        return GraspToolDataset(
            **common,
            with_offset=with_grasp_offset,
            with_short_side=bool(
                getattr(args, "predict_grasp_short_side", False)
            ),
            offset_radius=float(getattr(args, "offset_r", 20.0)),
            offset_sigma=getattr(args, "offset_sigma", None),
            dynamic_train_prompts=bool(
                getattr(args, "dynamic_train_prompts", False)
            ),
            dynamic_prompt_seed=int(
                getattr(args, "dynamic_prompt_seed", 2025)
            ),
        )
    if dataset_name in {"ocid-vlg", "ocidvlg"}:
        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"Unsupported OCID-VLG split {split!r}; choose train, val, or test."
            )
        return OCIDVLGDataset(
            **common,
            with_depth=bool(getattr(args, "with_depth", False)),
            version=args.version,
            with_grasp_offset=with_grasp_offset,
            offset_r=float(getattr(args, "offset_r", 20.0)),
            offset_sigma=getattr(args, "offset_sigma", None),
        )
    raise ValueError(
        f"Unsupported DATA.dataset {getattr(args, 'dataset', None)!r}; "
        "choose OCID-VLG, vcot, or GraspTool."
    )
