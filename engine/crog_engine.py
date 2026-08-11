import os
import time
from tqdm import tqdm

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger
from utils.dataset import tokenize
from utils.misc import (AverageMeter, ProgressMeter, trainMetricGPU, get_seg_image)
from utils.npu import autocast
from utils.grasp_eval import (detect_grasps, calculate_iou, calculate_max_iou, calculate_jacquard_index, visualization)
from utils.grasp_ablation import (
    filter_grasp_centres as filter_centres_inside_mask,
    mask_grasp_quality as mask_quality_outside_segmentation,
)
from utils.offset_eval import refine_with_offset, resample_grasp_geometry
from utils.vcot_eval import calculate_vcot_grasp_success


def _is_vcot_official(args):
    return str(getattr(args, "evaluation_protocol", "crog_legacy")).lower() == "vcot_official"


def _model_predicts_short_side(model):
    unwrapped_model = getattr(model, "module", model)
    return bool(getattr(unwrapped_model, "predicts_grasp_short_side", False))


def _split_grasp_predictions(model, predictions):
    predicts_short = _model_predicts_short_side(model)
    short_side = predictions[5] if predicts_short else None
    offset_index = 6 if predicts_short else 5
    offset = (
        predictions[offset_index]
        if len(predictions) > offset_index
        else None
    )
    return (*predictions[:5], short_side, offset)


def _grasp_size_scale(inverse_matrix, args):
    if not bool(getattr(args, "restore_grasp_size_scale", False)):
        return 1.0
    linear = np.asarray(inverse_matrix, dtype=np.float32)[:, :2]
    scale_x = float(np.linalg.norm(linear[:, 0]))
    scale_y = float(np.linalg.norm(linear[:, 1]))
    return max(1e-6, 0.5 * (scale_x + scale_y))


def _grasp_size_factor(args):
    factor = float(getattr(args, "grasp_size_factor", 100.0))
    if factor <= 0:
        raise ValueError("grasp_size_factor must be positive")
    return factor


def _grasp_size_activation(args):
    activation = str(
        getattr(args, "grasp_size_activation", "sigmoid")
    ).strip().lower()
    aliases = {
        "sigmoid": "sigmoid",
        "clamp": "clamp",
        "raw_clamp": "clamp",
    }
    if activation not in aliases:
        raise ValueError(
            "grasp_size_activation must be sigmoid, clamp, or raw_clamp"
        )
    return aliases[activation]


def _decode_grasp_size_map(prediction, args):
    if _grasp_size_activation(args) == "sigmoid":
        return torch.sigmoid(prediction)
    return torch.clamp(prediction, 0.0, 1.0)


def _reduce_iou_statistics(values, device):
    local_values = torch.as_tensor(
        values,
        dtype=torch.float32,
        device=device,
    ).reshape(-1)
    thresholds = torch.arange(0.5, 1.0, 0.1, device=device)
    stats = torch.stack(
        [
            local_values.sum(),
            torch.tensor(float(local_values.numel()), device=device),
            *((local_values > threshold).float().sum() for threshold in thresholds),
        ]
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats)
    count = stats[1].clamp_min(1.0)
    iou = stats[0] / count
    precision = [stats[index] / count for index in range(2, 7)]
    return iou, precision


def _inverse_interpolation(args):
    return cv2.INTER_NEAREST if _is_vcot_official(args) else cv2.INTER_CUBIC


def _segmentation_threshold(args):
    return 0.5 if _is_vcot_official(args) else 0.35


def _calculate_grasp_success(grasp_predictions, grasp_targets, args):
    if _is_vcot_official(args):
        prediction = grasp_predictions[0] if len(grasp_predictions) else None
        return calculate_vcot_grasp_success(prediction, grasp_targets)
    return calculate_jacquard_index(grasp_predictions, grasp_targets)


def _evaluation_topk(args):
    if not _is_vcot_official(args):
        return [1, 5]
    topk = [int(value) for value in getattr(args, "grasp_topk", [1])]
    if topk != [1]:
        raise ValueError(
            "vcot_official evaluates exactly one prediction; set TEST.grasp_topk: [1]."
        )
    return topk

def _apply_model_offset(
    grasps,
    offset_map,
    inverse_matrix,
    sine_map,
    cosine_map,
    width_map,
    short_side_map,
    size_scale,
    input_hw,
    args,
):
    """Apply DROG-OFF post-processing before the unchanged CROG scorer."""
    if not bool(getattr(args, "use_offset_at_inference", True)):
        return grasps
    if offset_map is None or not grasps:
        return grasps
    radius = float(getattr(args, "offset_r", min(input_hw) / 20.0))
    refined = refine_with_offset(
        grasps, offset_map, inverse_matrix, max(1.0, radius)
    )
    if bool(getattr(args, "offset_resample_geometry", False)):
        refined = resample_grasp_geometry(
            refined,
            sine_map,
            cosine_map,
            width_map,
            short_side=short_side_map,
            size_scale=size_scale,
            width_factor=_grasp_size_factor(args),
        )
    return refined


def _mask_grasp_quality(quality_map, segmentation_mask, args):
    """Suppress grasp candidates whose centres are outside the predicted mask."""
    if not bool(getattr(args, "filter_grasps_by_segmentation", False)):
        return quality_map
    return mask_quality_outside_segmentation(quality_map, segmentation_mask)


def _filter_grasp_centres(grasps, segmentation_mask, args):
    """Keep only final grasp rectangles centred inside the predicted mask."""
    if not bool(getattr(args, "filter_grasps_by_segmentation", False)):
        return grasps
    return filter_centres_inside_mask(grasps, segmentation_mask)


def _log_drogoff_inference_options(model, args):
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    unwrapped_model = getattr(model, "module", model)
    if rank == 0 and bool(getattr(unwrapped_model, "supports_offset", False)):
        logger.info(
            "DROG-OFF inference: offset={}, segmentation-centre-filter={}, "
            "resample-geometry={}",
            bool(getattr(args, "use_offset_at_inference", True)),
            bool(getattr(args, "filter_grasps_by_segmentation", False)),
            bool(getattr(args, "offset_resample_geometry", False)),
        )


def train_with_grasp(train_loader, model, optimizer, scheduler, scaler, epoch, args):
    batch_time = AverageMeter('Batch', ':2.2f')
    data_time = AverageMeter('Data', ':2.2f')
    lr = AverageMeter('Lr', ':1.6f')
    loss_meter = AverageMeter('Loss', ':2.4f')
    ins_loss_meter = AverageMeter('Loss_ins', ':2.4f')
    qua_loss_metter = AverageMeter('Loss_qua', ':2.4f')
    sin_loss_metter = AverageMeter('Loss_sin', ':2.4f')
    cos_loss_metter = AverageMeter('Loss_cos', ':2.4f')
    wid_loss_metter = AverageMeter('Loss_wid', ':2.4f')
    unwrapped_model = getattr(model, "module", model)
    off_loss_metter = (
        AverageMeter('Loss_off', ':2.4f')
        if bool(getattr(unwrapped_model, "supports_offset", False))
        else None
    )
    short_loss_metter = (
        AverageMeter('Loss_short', ':2.4f')
        if _model_predicts_short_side(model)
        else None
    )
    lgd_contrast_loss_meter = (
        AverageMeter('Loss_lgd_contrast', ':2.4f')
        if str(getattr(args, "architecture", "")).lower() == "lgd"
        else None
    )
    iou_meter = AverageMeter('IoU', ':2.2f')
    pr_meter = AverageMeter('Prec@50', ':2.2f')
    component_loss_meters = [
        ins_loss_meter,
        qua_loss_metter,
        sin_loss_metter,
        cos_loss_metter,
        wid_loss_metter,
    ]
    if off_loss_metter is not None:
        component_loss_meters.append(off_loss_metter)
    if short_loss_metter is not None:
        component_loss_meters.append(short_loss_metter)
    if lgd_contrast_loss_meter is not None:
        component_loss_meters.append(lgd_contrast_loss_meter)
    progress = ProgressMeter(
        len(train_loader),
        [
            batch_time, data_time, lr, loss_meter,
            *component_loss_meters,
            iou_meter, pr_meter
        ],
        prefix="Training: Epoch=[{}/{}] ".format(epoch, args.epochs))

    model.train()
    time.sleep(2)
    end = time.time()
    accumulation_steps = int(
        getattr(args, "gradient_accumulation_steps", 1)
    )
    if accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    optimizer.zero_grad()

    # size_list = [320, 352, 384, 416, 448, 480, 512]
    # idx = np.random.choice(len(size_list))
    # new_size = size_list[idx]

    for i, data in enumerate(train_loader):
        # image, target, text = data
        # ins_mask, grasp_quality_mask, grasp_sin_mask, grasp_cos_mask, grasp_width_mask = target

        image = data["img"]
        text = data["word_vec"]
        ins_mask = data["mask"]
        grasp_qua_mask = data["grasp_masks"]["qua"]
        grasp_sin_mask = data["grasp_masks"]["sin"]
        grasp_cos_mask = data["grasp_masks"]["cos"]
        grasp_wid_mask = data["grasp_masks"]["wid"]
        grasp_off_mask = data["grasp_masks"].get("off")
        grasp_off_weight = data["grasp_masks"].get("off_w")
        grasp_short_mask = data["grasp_masks"].get("short")


        data_time.update(time.time() - end)
        # data
        image = image.to(args.device, non_blocking=True)
        text = text.to(args.device, non_blocking=True)
        ins_mask = ins_mask.to(args.device, non_blocking=True).unsqueeze(1)
        grasp_qua_mask = grasp_qua_mask.to(args.device, non_blocking=True).unsqueeze(1)
        grasp_sin_mask = grasp_sin_mask.to(args.device, non_blocking=True).unsqueeze(1)
        grasp_cos_mask = grasp_cos_mask.to(args.device, non_blocking=True).unsqueeze(1)
        grasp_wid_mask = grasp_wid_mask.to(args.device, non_blocking=True).unsqueeze(1)
        if grasp_off_mask is not None:
            grasp_off_mask = grasp_off_mask.to(args.device, non_blocking=True)
            grasp_off_weight = grasp_off_weight.to(args.device, non_blocking=True)
        if grasp_short_mask is not None:
            grasp_short_mask = grasp_short_mask.to(
                args.device, non_blocking=True
            ).unsqueeze(1)

        # # multi-scale training
        # image = F.interpolate(image, size=(new_size, new_size), mode='bilinear')

        # forward
        with autocast(enabled=bool(getattr(args, "amp", True))):
            model_inputs = (
                image, text, ins_mask, grasp_qua_mask, grasp_sin_mask,
                grasp_cos_mask, grasp_wid_mask,
            )
            if grasp_off_mask is not None:
                model_inputs = (*model_inputs, grasp_off_mask, grasp_off_weight)
            if _model_predicts_short_side(model):
                model_inputs = (*model_inputs, grasp_short_mask)
            pred, target, loss, loss_dict = model(*model_inputs)

        if not bool(torch.isfinite(loss.detach()).item()):
            components = {
                name: float(torch.as_tensor(value).detach().float().cpu())
                for name, value in loss_dict.items()
            }
            raise FloatingPointError(
                "Non-finite training loss before backward: "
                f"epoch={epoch}, step={i + 1}, components={components}"
            )

        ins_mask_pred = pred[0]
        ins_mask_target = target[0]

        # Accumulate smaller micro-batches to reduce the activation-memory peak
        # while retaining the configured effective global batch size.
        window_start = (i // accumulation_steps) * accumulation_steps
        window_size = min(
            accumulation_steps, len(train_loader) - window_start
        )
        should_step = (
            (i + 1) % accumulation_steps == 0
            or (i + 1) == len(train_loader)
        )
        scaler.scale(loss / window_size).backward()
        if should_step:
            if args.max_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.max_norm,
                    error_if_nonfinite=True,
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # metric
        iou, pr5 = trainMetricGPU(ins_mask_pred, ins_mask_target, 0.35, 0.5)
        dist.all_reduce(loss.detach())
        dist.all_reduce(iou)
        dist.all_reduce(pr5)
        loss = loss / dist.get_world_size()
        iou = iou / dist.get_world_size()
        pr5 = pr5 / dist.get_world_size()

        loss_meter.update(loss.item(), image.size(0))
        ins_loss_meter.update(loss_dict["m_ins"], image.size(0))
        qua_loss_metter.update(loss_dict["m_qua"], image.size(0))
        sin_loss_metter.update(loss_dict["m_sin"], image.size(0))
        cos_loss_metter.update(loss_dict["m_cos"], image.size(0))
        wid_loss_metter.update(loss_dict["m_wid"], image.size(0))
        if off_loss_metter is not None:
            off_loss_metter.update(loss_dict["m_off"], image.size(0))
        if short_loss_metter is not None:
            short_loss_metter.update(loss_dict["m_short"], image.size(0))
        if lgd_contrast_loss_meter is not None:
            lgd_contrast_loss_meter.update(
                loss_dict["m_lgd_contrast"], image.size(0)
            )
        iou_meter.update(iou.item(), image.size(0))
        pr_meter.update(pr5.item(), image.size(0))
        lr.update(scheduler.get_last_lr()[-1])
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i + 1)
            # if dist.get_rank() in [-1, 0]:
            #     wandb.log(
            #         {
            #             "time/batch": batch_time.val,
            #             "time/data": data_time.val,
            #             "training/lr": lr.val,
            #             "training/loss": loss_meter.val,
            #             "training/loss_qua": qua_loss_metter.val,
            #             "training/loss_sin": sin_loss_metter.val,
            #             "training/loss_cos": cos_loss_metter.val,
            #             "training/loss_wid": wid_loss_metter.val,
            #             "training/iou": iou_meter.val,
            #             "training/prec@50": pr_meter.val,
            #         },
            #         step=epoch * len(train_loader) + (i + 1))


@torch.no_grad()
def validate_with_grasp(val_loader, model, epoch, args):
    def inverse(img, mat, w, h):
        inv_img = cv2.warpAffine(img, mat, (w, h),
                                    flags=_inverse_interpolation(args),
                                    borderValue=0.)
        return inv_img

    iou_list = []
    num_correct_grasps = 0
    num_total_grasps = 0
    model.eval()
    evaluation_model = getattr(model, "module", model)
    _log_drogoff_inference_options(model, args)
    time.sleep(2)

    num_grasps = _evaluation_topk(args)
    num_correct_grasps = [0 for _ in num_grasps]
    num_total_grasps = [0 for _ in num_grasps]

    pbar = tqdm(val_loader, disable=dist.get_rank() != 0)
    for data in pbar:
        # data
        image = data["img"]
        text = data["word_vec"]
        ins_mask = data["mask"]
        grasp_qua_mask = data["grasp_masks"]["qua"]
        grasp_sin_mask = data["grasp_masks"]["sin"]
        grasp_cos_mask = data["grasp_masks"]["cos"]
        grasp_wid_mask = data["grasp_masks"]["wid"]
        grasp_short_mask = data["grasp_masks"].get("short")
        inverse_matrix = data["inverse"]
        ori_sizes = data["ori_size"]
        grasp_targets = data["grasps"]

        image = image.to(args.device, non_blocking=True)
        text = text.to(args.device, non_blocking=True)
        ins_mask = ins_mask.to(args.device, non_blocking=True).unsqueeze(1)
        grasp_qua_mask = grasp_qua_mask.to(args.device, non_blocking=True).unsqueeze(1)
        grasp_sin_mask = grasp_sin_mask.to(args.device, non_blocking=True).unsqueeze(1)
        grasp_cos_mask = grasp_cos_mask.to(args.device, non_blocking=True).unsqueeze(1)
        grasp_wid_mask = grasp_wid_mask.to(args.device, non_blocking=True).unsqueeze(1)
        if grasp_short_mask is not None:
            grasp_short_mask = grasp_short_mask.to(
                args.device, non_blocking=True
            ).unsqueeze(1)

        # inference & get predictions from model
        eval_kwargs = {}
        if _model_predicts_short_side(model):
            eval_kwargs["grasp_short_mask"] = grasp_short_mask
        pred, _ = evaluation_model(image, text, ins_mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask, **eval_kwargs)

        # predictions
        (
            ins_mask_preds,
            grasp_qua_mask_preds,
            grasp_sin_mask_preds,
            grasp_cos_mask_preds,
            grasp_wid_mask_preds,
            grasp_short_mask_preds,
            grasp_off_mask_preds,
        ) = _split_grasp_predictions(model, pred)

        # Evaluation uses original input-resolution dataloader targets.
        # Model-returned targets may be resized for training loss computation.
        ins_mask_targets = ins_mask
        grasp_qua_mask_targets = grasp_qua_mask
        grasp_sin_mask_targets = grasp_sin_mask
        grasp_cos_mask_targets = grasp_cos_mask
        grasp_wid_mask_targets = grasp_wid_mask

        # Interpolate the predicted ins mask to the same size of input image
        ins_mask_preds = torch.sigmoid(ins_mask_preds)
        grasp_qua_mask_preds = torch.sigmoid(grasp_qua_mask_preds)
        grasp_wid_mask_preds = _decode_grasp_size_map(grasp_wid_mask_preds, args)
        if grasp_short_mask_preds is not None:
            grasp_short_mask_preds = _decode_grasp_size_map(grasp_short_mask_preds, args)
        if (
            grasp_off_mask_preds is not None
            and grasp_off_mask_preds.shape[-2:] != image.shape[-2:]
        ):
            grasp_off_mask_preds = F.interpolate(
                grasp_off_mask_preds,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if ins_mask_preds.shape[-2:] != image.shape[-2:]:
            ins_mask_preds = F.interpolate(ins_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)

            grasp_qua_mask_preds = F.interpolate(grasp_qua_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)

            grasp_sin_mask_preds = F.interpolate(grasp_sin_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)

            grasp_cos_mask_preds = F.interpolate(grasp_cos_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)

            grasp_wid_mask_preds = F.interpolate(grasp_wid_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)
            if grasp_short_mask_preds is not None:
                grasp_short_mask_preds = F.interpolate(
                    grasp_short_mask_preds,
                    size=image.shape[-2:],
                    mode='bicubic',
                    align_corners=True).squeeze(1)

        # iterate over the whole batch
        for idx in range(ins_mask_preds.shape[0]):
            inv_mat = inverse_matrix[idx]
            ori_size = ori_sizes[idx]
            h, w = ori_size

            ins_mask_pred = ins_mask_preds[idx].squeeze().cpu().numpy()
            grasp_qua_mask_pred = grasp_qua_mask_preds[idx].squeeze().cpu().numpy()
            grasp_sin_mask_pred = grasp_sin_mask_preds[idx].squeeze().cpu().numpy()
            grasp_cos_mask_pred = grasp_cos_mask_preds[idx].squeeze().cpu().numpy()
            grasp_wid_mask_pred = grasp_wid_mask_preds[idx].squeeze().cpu().numpy()
            grasp_short_mask_pred = (
                grasp_short_mask_preds[idx].squeeze().cpu().numpy()
                if grasp_short_mask_preds is not None
                else None
            )
            grasp_off_mask_pred = (
                grasp_off_mask_preds[idx : idx + 1].cpu().numpy()
                if grasp_off_mask_preds is not None
                else None
            )

            ins_mask_target = ins_mask_targets[idx].squeeze().cpu().numpy()
            grasp_target = grasp_targets[idx]
            grasp_qua_mask_target = grasp_qua_mask_targets[idx].squeeze().cpu().numpy()
            grasp_sin_mask_target = grasp_sin_mask_targets[idx].squeeze().cpu().numpy()
            grasp_cos_mask_target = grasp_cos_mask_targets[idx].squeeze().cpu().numpy()
            grasp_wid_mask_target = grasp_wid_mask_targets[idx].squeeze().cpu().numpy()

            # Inverse to original size
            ins_mask_pred = inverse(ins_mask_pred, inv_mat, w, h)
            ins_mask_pred = (ins_mask_pred > _segmentation_threshold(args))
            grasp_qua_mask_pred = inverse(grasp_qua_mask_pred, inv_mat, w, h)
            grasp_sin_mask_pred = inverse(grasp_sin_mask_pred, inv_mat, w, h)
            grasp_cos_mask_pred = inverse(grasp_cos_mask_pred, inv_mat, w, h)
            grasp_wid_mask_pred = inverse(grasp_wid_mask_pred, inv_mat, w, h)
            if grasp_short_mask_pred is not None:
                grasp_short_mask_pred = inverse(grasp_short_mask_pred, inv_mat, w, h)
            size_scale = _grasp_size_scale(inv_mat, args)

            ins_mask_target = inverse(ins_mask_target, inv_mat, w, h)
            grasp_qua_mask_target = inverse(grasp_qua_mask_target, inv_mat, w, h)
            grasp_sin_mask_target = inverse(grasp_sin_mask_target, inv_mat, w, h)
            grasp_cos_mask_target = inverse(grasp_cos_mask_target, inv_mat, w, h)
            grasp_wid_mask_target = inverse(grasp_wid_mask_target, inv_mat, w, h)

            # Calculate IoU between predicted instance mask and gt
            inter = np.logical_and(ins_mask_pred, ins_mask_target)
            union = np.logical_or(ins_mask_pred, ins_mask_target)

            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_list.append(iou)

            # Calculate grasp configurations
            for i in range(len(num_grasps)):
                num_g = num_grasps[i]
                detection_quality = _mask_grasp_quality(
                    grasp_qua_mask_pred, ins_mask_pred, args
                )
                grasp_preds, _ = detect_grasps(
                    detection_quality,
                    grasp_sin_mask_pred,
                    grasp_cos_mask_pred,
                    grasp_wid_mask_pred,
                    num_g,
                    grasp_short_mask=grasp_short_mask_pred,
                    size_scale=size_scale,
                    size_factor=_grasp_size_factor(args),
                )
                grasp_preds = _apply_model_offset(
                    grasp_preds,
                    grasp_off_mask_pred,
                    inv_mat,
                    grasp_sin_mask_pred,
                    grasp_cos_mask_pred,
                    grasp_wid_mask_pred,
                    grasp_short_mask_pred,
                    size_scale,
                    image.shape[-2:],
                    args,
                )
                grasp_preds = _filter_grasp_centres(
                    grasp_preds, ins_mask_pred, args
                )

                j_index = _calculate_grasp_success(grasp_preds, grasp_target, args)

                num_correct_grasps[i] += j_index
                num_total_grasps[i] += 1

    correct = torch.tensor(
        num_correct_grasps, dtype=torch.float32, device=image.device
    )
    total = torch.tensor(
        num_total_grasps, dtype=torch.float32, device=image.device
    )
    dist.all_reduce(correct)
    dist.all_reduce(total)
    J_index = (correct / total.clamp_min(1)).cpu().tolist()

    iou, prec_list = _reduce_iou_statistics(iou_list, image.device)
    prec = {}
    temp = '  '
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres * 10)
        value = prec_list[i].item()
        prec[key] = value
        temp += "{}: {:.2f}  ".format(key, 100. * value)
    if _is_vcot_official(args):
        head = (
            'Evaluation: Epoch=[{}/{}]  IoU={:.2f}  GraspSR: {:.2f}'
            .format(epoch, args.epochs, 100. * iou.item(), 100. * J_index[0])
        )
    else:
        head = (
            'Evaluation: Epoch=[{}/{}]  IoU={:.2f}  '
            'J_index@1: {:.2f}  J_index@5: {:.2f}'
            .format(
                epoch,
                args.epochs,
                100. * iou.item(),
                100. * J_index[0],
                100. * J_index[1],
            )
        )
    logger.info(head + temp)
    return iou.item(), prec, J_index


@torch.no_grad()
def validate_without_grasp(val_loader, model, epoch, args):
    def inverse(img, mat, w, h):
        inv_img = cv2.warpAffine(img, mat, (w, h),
                                    flags=_inverse_interpolation(args),
                                    borderValue=0.)
        return inv_img

    iou_list = []
    num_correct_grasps = 0
    num_total_grasps = 0
    model.eval()
    evaluation_model = getattr(model, "module", model)
    time.sleep(2)

    num_grasps = _evaluation_topk(args)
    num_correct_grasps = [0 for _ in num_grasps]
    num_total_grasps = [0 for _ in num_grasps]

    pbar = tqdm(val_loader, disable=dist.get_rank() != 0)
    for data in pbar:
        # data
        image = data["img"]
        text = data["word_vec"]
        ins_mask = data["mask"]
        grasp_qua_mask = data["grasp_masks"]["qua"]
        grasp_sin_mask = data["grasp_masks"]["sin"]
        grasp_cos_mask = data["grasp_masks"]["cos"]
        grasp_wid_mask = data["grasp_masks"]["wid"]
        inverse_matrix = data["inverse"]
        ori_sizes = data["ori_size"]
        grasp_targets = data["grasps"]

        image = image.to(args.device, non_blocking=True)
        text = text.to(args.device, non_blocking=True)
        ins_mask = ins_mask.to(args.device, non_blocking=True).unsqueeze(1)
        grasp_qua_mask = grasp_qua_mask.to(args.device, non_blocking=True).unsqueeze(1)
        grasp_sin_mask = grasp_sin_mask.to(args.device, non_blocking=True).unsqueeze(1)
        grasp_cos_mask = grasp_cos_mask.to(args.device, non_blocking=True).unsqueeze(1)
        grasp_wid_mask = grasp_wid_mask.to(args.device, non_blocking=True).unsqueeze(1)

        # inference & get predictions from model
        pred, _ = evaluation_model(image, text, ins_mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask)
        ins_mask_targets = ins_mask

        # Interpolate the predicted ins mask to the same size of input image
        ins_mask_preds = torch.sigmoid(pred)
        if ins_mask_preds.shape[-2:] != image.shape[-2:]:
            ins_mask_preds = F.interpolate(ins_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)

        # iterate over the whole batch
        for idx in range(ins_mask_preds.shape[0]):
            inv_mat = inverse_matrix[idx]
            ori_size = ori_sizes[idx]
            h, w = ori_size

            ins_mask_pred = ins_mask_preds[idx].squeeze().cpu().numpy()
            ins_mask_target = ins_mask_targets[idx].squeeze().cpu().numpy()

            # Inverse to original size
            ins_mask_pred = inverse(ins_mask_pred, inv_mat, w, h)
            ins_mask_pred = (ins_mask_pred > _segmentation_threshold(args))

            ins_mask_target = inverse(ins_mask_target, inv_mat, w, h)

            # Calculate IoU between predicted instance mask and gt
            inter = np.logical_and(ins_mask_pred, ins_mask_target)
            union = np.logical_or(ins_mask_pred, ins_mask_target)

            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_list.append(iou)

    J_index = [0, 0]

    iou, prec_list = _reduce_iou_statistics(iou_list, image.device)
    prec = {}
    temp = '  '
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres * 10)
        value = prec_list[i].item()
        prec[key] = value
        temp += "{}: {:.2f}  ".format(key, 100. * value)
    head = 'Evaluation: Epoch=[{}/{}]  IoU={:.2f}  J_index@1: {:.2f}  J_index@5: {:.2f}'.format(
        epoch, args.epochs, 100. * iou.item(), 100. * J_index[0], 100. * J_index[1])
    logger.info(head + temp)
    return iou.item(), prec, J_index



@torch.no_grad()
def inference_with_grasp(test_loader, model, args):
    def inverse(img, mat, w, h):
        inv_img = cv2.warpAffine(img, mat, (w, h),
                                    flags=_inverse_interpolation(args),
                                    borderValue=0.)
        return inv_img

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    thresholds = (0.5, 0.6, 0.7, 0.8, 0.9)
    iou_sum = 0.0
    sample_count = 0
    precision_counts = [0, 0, 0, 0, 0]
    model.eval()
    _log_drogoff_inference_options(model, args)
    time.sleep(2)

    num_grasps = _evaluation_topk(args)
    num_correct_grasps = [0 for _ in num_grasps]
    num_total_grasps = [0 for _ in num_grasps]

    tbar = tqdm(
        test_loader,
        desc='Inference:',
        ncols=100,
        disable=rank != 0,
    )
    for cnt, data in enumerate(tbar):

        # data
        image = data["img"]
        text = data["word_vec"]
        ins_mask = data["mask"]
        grasp_qua_mask = data["grasp_masks"]["qua"]
        grasp_sin_mask = data["grasp_masks"]["sin"]
        grasp_cos_mask = data["grasp_masks"]["cos"]
        grasp_wid_mask = data["grasp_masks"]["wid"]
        grasp_short_mask = data["grasp_masks"].get("short")
        inverse_matrix = data["inverse"]
        ori_sizes = data["ori_size"]
        grasp_targets = data["grasps"]
        sentences = data["sentence"]
        img_paths = data["img_path"]

        image = image.to(args.device, non_blocking=True)
        text = text.to(args.device, non_blocking=True)
        ins_mask = ins_mask.to(args.device, non_blocking=True).unsqueeze(1)
        grasp_qua_mask = grasp_qua_mask.to(args.device, non_blocking=True).unsqueeze(1)
        grasp_sin_mask = grasp_sin_mask.to(args.device, non_blocking=True).unsqueeze(1)
        grasp_cos_mask = grasp_cos_mask.to(args.device, non_blocking=True).unsqueeze(1)
        grasp_wid_mask = grasp_wid_mask.to(args.device, non_blocking=True).unsqueeze(1)
        if grasp_short_mask is not None:
            grasp_short_mask = grasp_short_mask.to(
                args.device, non_blocking=True
            ).unsqueeze(1)

        # inference & get predictions from model
        eval_kwargs = {}
        if _model_predicts_short_side(model):
            eval_kwargs["grasp_short_mask"] = grasp_short_mask
        pred, _ = model(image, text, ins_mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask, **eval_kwargs)

        # Legacy DROG-OFF uses pred[5] for offset; VCoT uses it for short side.
        # predictions
        (
            ins_mask_preds,
            grasp_qua_mask_preds,
            grasp_sin_mask_preds,
            grasp_cos_mask_preds,
            grasp_wid_mask_preds,
            grasp_short_mask_preds,
            grasp_off_mask_preds,
        ) = _split_grasp_predictions(model, pred)

        # Evaluation uses original input-resolution dataloader targets.
        # Model-returned targets may be resized for training loss computation.
        ins_mask_targets = ins_mask
        grasp_qua_mask_targets = grasp_qua_mask
        grasp_sin_mask_targets = grasp_sin_mask
        grasp_cos_mask_targets = grasp_cos_mask
        grasp_wid_mask_targets = grasp_wid_mask

        # Interpolate the predicted ins mask to the same size of input image
        ins_mask_preds = torch.sigmoid(ins_mask_preds)
        grasp_qua_mask_preds = torch.sigmoid(grasp_qua_mask_preds)
        grasp_wid_mask_preds = _decode_grasp_size_map(grasp_wid_mask_preds, args)
        if grasp_short_mask_preds is not None:
            grasp_short_mask_preds = _decode_grasp_size_map(grasp_short_mask_preds, args)
        if (
            grasp_off_mask_preds is not None
            and grasp_off_mask_preds.shape[-2:] != image.shape[-2:]
        ):
            grasp_off_mask_preds = F.interpolate(
                grasp_off_mask_preds,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if ins_mask_preds.shape[-2:] != image.shape[-2:]:
            ins_mask_preds = F.interpolate(ins_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)

            grasp_qua_mask_preds = F.interpolate(grasp_qua_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)

            grasp_sin_mask_preds = F.interpolate(grasp_sin_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)

            grasp_cos_mask_preds = F.interpolate(grasp_cos_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)

            grasp_wid_mask_preds = F.interpolate(grasp_wid_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)
            if grasp_short_mask_preds is not None:
                grasp_short_mask_preds = F.interpolate(
                    grasp_short_mask_preds,
                    size=image.shape[-2:],
                    mode='bicubic',
                    align_corners=True).squeeze(1)


        # iterate over the whole batch
        for idx in range(ins_mask_preds.shape[0]):
            inv_mat = inverse_matrix[idx]
            ori_size = ori_sizes[idx]
            h, w = ori_size
            sent = sentences[idx]
            img_path = img_paths[idx]

            ins_mask_pred = ins_mask_preds[idx].squeeze().cpu().numpy()
            grasp_qua_mask_pred = grasp_qua_mask_preds[idx].squeeze().cpu().numpy()
            grasp_sin_mask_pred = grasp_sin_mask_preds[idx].squeeze().cpu().numpy()
            grasp_cos_mask_pred = grasp_cos_mask_preds[idx].squeeze().cpu().numpy()
            grasp_wid_mask_pred = grasp_wid_mask_preds[idx].squeeze().cpu().numpy()
            grasp_short_mask_pred = (
                grasp_short_mask_preds[idx].squeeze().cpu().numpy()
                if grasp_short_mask_preds is not None
                else None
            )
            grasp_off_mask_pred = (
                grasp_off_mask_preds[idx : idx + 1].cpu().numpy()
                if grasp_off_mask_preds is not None
                else None
            )

            ins_mask_target = ins_mask_targets[idx].squeeze().cpu().numpy()
            grasp_target = grasp_targets[idx]
            grasp_qua_mask_target = grasp_qua_mask_targets[idx].squeeze().cpu().numpy()
            grasp_sin_mask_target = grasp_sin_mask_targets[idx].squeeze().cpu().numpy()
            grasp_cos_mask_target = grasp_cos_mask_targets[idx].squeeze().cpu().numpy()
            grasp_wid_mask_target = grasp_wid_mask_targets[idx].squeeze().cpu().numpy()

            # Inverse to original size
            ins_mask_pred = inverse(ins_mask_pred, inv_mat, w, h)
            ins_mask_pred = (ins_mask_pred > _segmentation_threshold(args))
            grasp_qua_mask_pred = inverse(grasp_qua_mask_pred, inv_mat, w, h)
            grasp_sin_mask_pred = inverse(grasp_sin_mask_pred, inv_mat, w, h)
            grasp_cos_mask_pred = inverse(grasp_cos_mask_pred, inv_mat, w, h)
            grasp_wid_mask_pred = inverse(grasp_wid_mask_pred, inv_mat, w, h)
            if grasp_short_mask_pred is not None:
                grasp_short_mask_pred = inverse(grasp_short_mask_pred, inv_mat, w, h)
            size_scale = _grasp_size_scale(inv_mat, args)

            ins_mask_target = inverse(ins_mask_target, inv_mat, w, h)
            grasp_qua_mask_target = inverse(grasp_qua_mask_target, inv_mat, w, h)
            grasp_sin_mask_target = inverse(grasp_sin_mask_target, inv_mat, w, h)
            grasp_cos_mask_target = inverse(grasp_cos_mask_target, inv_mat, w, h)
            grasp_wid_mask_target = inverse(grasp_wid_mask_target, inv_mat, w, h)

            # Calculate IoU between predicted instance mask and gt
            inter = np.logical_and(ins_mask_pred, ins_mask_target)
            union = np.logical_or(ins_mask_pred, ins_mask_target)

            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_sum += float(iou)
            sample_count += 1
            for threshold_idx, threshold in enumerate(thresholds):
                precision_counts[threshold_idx] += int(iou > threshold)

            # Calculate grasp configurations
            for i in range(len(num_grasps)):
                num_g = num_grasps[i]
                detection_quality = _mask_grasp_quality(
                    grasp_qua_mask_pred, ins_mask_pred, args
                )
                grasp_preds, grasp_ang_mask_pred = detect_grasps(
                    detection_quality,
                    grasp_sin_mask_pred,
                    grasp_cos_mask_pred,
                    grasp_wid_mask_pred,
                    num_g,
                    grasp_short_mask=grasp_short_mask_pred,
                    size_scale=size_scale,
                    size_factor=_grasp_size_factor(args),
                )
                grasp_preds = _apply_model_offset(
                    grasp_preds,
                    grasp_off_mask_pred,
                    inv_mat,
                    grasp_sin_mask_pred,
                    grasp_cos_mask_pred,
                    grasp_wid_mask_pred,
                    grasp_short_mask_pred,
                    size_scale,
                    image.shape[-2:],
                    args,
                )
                grasp_preds = _filter_grasp_centres(
                    grasp_preds, ins_mask_pred, args
                )

                j_index = _calculate_grasp_success(grasp_preds, grasp_target, args)

                num_correct_grasps[i] += j_index
                num_total_grasps[i] += 1

                # Visualization
                if args.visualize:
                    img_bgr = cv2.imread(img_path)
                    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    visualization(img, ins_mask_pred, (grasp_qua_mask_pred, grasp_ang_mask_pred, grasp_wid_mask_pred), grasp_preds, sent, save_path=os.path.join("./results", args.exp_name, f"results_rank{rank}_{cnt}_{num_g}_grasps.png"))

    grasp_stats = [
        value
        for pair in zip(num_correct_grasps, num_total_grasps)
        for value in pair
    ]
    stats = torch.tensor(
        [iou_sum, sample_count, *precision_counts, *grasp_stats],
        dtype=torch.float32,
        device=args.device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    total = stats[1].clamp_min(1.0)
    iou = stats[0] / total
    prec = {
        "Pr@{}".format(int(threshold * 100)): (
            stats[2 + index] / total
        ).item()
        for index, threshold in enumerate(thresholds)
    }
    grasp_offset = 2 + len(thresholds)
    J_index = [
        (
            stats[grasp_offset + 2 * index]
            / stats[grasp_offset + 2 * index + 1].clamp_min(1.0)
        ).item()
        for index in range(len(num_grasps))
    ]
    if rank == 0:
        logger.info('IoU={:.2f}'.format(100.*iou.item()))
        for key, value in prec.items():
            logger.info('{}: {:.2f}.'.format(key, 100.*value))
        if _is_vcot_official(args):
            logger.info("GraspSR: {:.2f}".format(100. * J_index[0]))
        else:
            logger.info(
                "J@1: {:.2f}, J@5: {:.2f}".format(
                    100. * J_index[0], 100. * J_index[1]
                )
            )

    return iou.item(), prec, J_index
