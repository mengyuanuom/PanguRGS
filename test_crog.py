import argparse
import os
import warnings

import cv2
import torch
import torch.distributed as dist
import torch.utils.data
from loguru import logger

import utils.config as config
from engine.crog_engine import inference_with_grasp
from model import build_model
from utils.data_builder import build_referring_grasp_dataset
from utils.misc import setup_logger
from utils.npu import set_device

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument('--config',
                        default='path to xxx.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


@logger.catch(reraise=True)
def main():
    args = get_parser()
    evaluation_protocol = str(
        getattr(args, "evaluation_protocol", "crog_legacy")
    ).strip().lower()
    protocol_aliases = {
        "crog": "crog_legacy",
        "crog_source": "crog_legacy",
        "crog_legacy": "crog_legacy",
        "vcot": "vcot_official",
        "vcot_source": "vcot_official",
        "vcot_official": "vcot_official",
    }
    if evaluation_protocol not in protocol_aliases:
        raise ValueError(
            "Unsupported evaluation protocol; choose crog_legacy or vcot_official."
        )
    args.evaluation_protocol = protocol_aliases[evaluation_protocol]
    args.npu = int(os.environ.get("LOCAL_RANK", 0))
    args.rank = int(os.environ.get("RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.distributed = args.world_size > 1
    args.device = set_device(args.npu)
    if args.distributed:
        dist.init_process_group(
            backend="hccl",
            init_method="env://",
            rank=args.rank,
            world_size=args.world_size,
        )
    args.output_dir = os.path.join(args.output_folder, args.exp_name)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.visualize:
        args.vis_dir = os.path.join(args.output_dir, "vis")
        os.makedirs(args.vis_dir, exist_ok=True)

    # logger
    setup_logger(args.output_dir,
                 distributed_rank=args.rank,
                 filename="test.log",
                 mode="a")
    logger.info(args)

    # build dataset & dataloader
    test_split = str(getattr(args, "test_split", "test"))
    full_test_data = build_referring_grasp_dataset(
        args,
        split=test_split,
        with_grasp_offset=False,
    )
    logger.info(
        "Dataset: {} (test_split={}, {} samples; protocol={})",
        getattr(args, "dataset", "OCID-VLG"),
        test_split,
        len(full_test_data),
        args.evaluation_protocol,
    )
    if args.distributed:
        indices = range(args.rank, len(full_test_data), args.world_size)
        test_data = torch.utils.data.Subset(full_test_data, indices)
    else:
        test_data = full_test_data
    test_batch_size = int(getattr(args, "test_batch_size", 1))
    test_workers = int(getattr(args, "test_workers", 1))
    if test_batch_size <= 0:
        raise ValueError("TEST.test_batch_size must be positive")
    if test_workers < 0:
        raise ValueError("TEST.test_workers must be non-negative")
    logger.info(
        "Inference loader: per-device batch_size={}, workers={}",
        test_batch_size,
        test_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=test_workers,
        pin_memory=bool(getattr(args, "pin_memory", False)),
        collate_fn=full_test_data.collate_fn,
    )
    # build model
    model, _ = build_model(args)
    model = model.to(args.device)
    logger.info(model)

    save_path = os.path.join("./results", args.exp_name)
    os.makedirs(save_path, exist_ok=True)

    if os.path.isfile(args.resume):
        logger.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location="cpu")
        requested_activation = getattr(
            args, "grasp_size_activation", "sigmoid"
        )
        args.grasp_size_activation = config.resolve_grasp_size_activation(
            requested_activation, checkpoint
        )
        logger.info(
            "Grasp-size activation: requested={}, resolved={}",
            requested_activation, args.grasp_size_activation,
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
        model.load_state_dict(state_dict, strict=True)
        logger.info("=> loaded checkpoint '{}'".format(args.resume))
    else:
        raise ValueError(
            "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
            .format(args.resume))

    # inference
    try:
        inference_with_grasp(test_loader, model, args)
    finally:
        if args.distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
