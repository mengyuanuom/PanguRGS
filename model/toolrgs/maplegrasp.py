"""Ascend NPU adaptation of the official two-stage MapleGrasp model.

Source: https://github.com/vineet2104/MapleGrasp
Reference commit: c1b1f48e7ff24caaf39daa127d47d9469b93c7a1

The model structure, parameter names, losses, hard 0.35 mask gate, and
stage-1/stage-2 contract follow the official release. The only model-level
changes are device-neutral execution, shape-safe mask resizing, and a fused
grouped convolution that avoids non-zero-storage-offset split views on Ascend.
An optional VCoT extension adds a predicted grasp short side while preserving
the official four-map Stage-2 head for legacy configurations and checkpoints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .crog_clip import build_model
from .crog_layers import FPN, TransformerDecoder, conv_layer


class MultiTaskProjectorPP(nn.Module):
    """Official MapleGrasp projector with an Ascend-safe grasp path."""

    def __init__(
        self,
        word_dim=1024,
        in_dim=256,
        kernel_size=3,
        stage1=False,
        stage2=False,
        use_gt_obj_masks=False,
        predict_short_side=False,
    ):
        super().__init__()
        if bool(stage1) == bool(stage2):
            raise ValueError(
                "MapleGrasp requires exactly one of stage1 or stage2 to be True"
            )
        self.in_dim = int(in_dim)
        self.kernel_size = int(kernel_size)
        self.stage1 = bool(stage1)
        self.stage2 = bool(stage2)
        self.use_gt_obj_masks = bool(use_gt_obj_masks)
        self.predict_short_side = bool(predict_short_side) and self.stage2
        self.num_grasp_outputs = 5 if self.predict_short_side else 4

        # Preserve the official names for Stage-1 -> Stage-2 loading.
        self.vis = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv_layer(in_dim * 2, in_dim * 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv_layer(in_dim * 2, in_dim, 3, padding=1),
        )
        self.vis_mask = nn.Conv2d(in_dim, in_dim, 1)
        if self.stage2:
            self.vis_grasp = nn.Conv2d(
                in_dim, in_dim * self.num_grasp_outputs, 1
            )
        self.txt = nn.Linear(
            word_dim, in_dim * kernel_size * kernel_size + 1
        )

    def _text_kernel(self, word):
        batch_size = word.shape[0]
        parameters = self.txt(word)
        weight = parameters[:, :-1].reshape(
            batch_size,
            self.in_dim,
            self.kernel_size,
            self.kernel_size,
        )
        return weight, parameters[:, -1]

    def _dynamic_mask_conv(self, features, weight, bias):
        batch_size, channels, height, width = features.shape
        output = F.conv2d(
            features.reshape(1, batch_size * channels, height, width),
            weight,
            bias=bias,
            padding=self.kernel_size // 2,
            groups=batch_size,
        )
        return output.transpose(0, 1).contiguous()

    def _grasp_gate(self, mask_out, gt_mask, output_size):
        # The detach, sigmoid, hard threshold and 0.35 value follow upstream.
        gate = (torch.sigmoid(mask_out.detach()) > 0.35).to(mask_out.dtype)
        if self.use_gt_obj_masks:
            if gt_mask is None:
                raise ValueError(
                    "use_gt_obj_masks=True requires a target object mask"
                )
            gate = gt_mask.detach().bool().to(dtype=mask_out.dtype)
        if gate.shape[-2:] != output_size:
            gate = F.interpolate(
                gate,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        return gate

    def forward(self, x, word, mask=None):
        x = self.vis(x)
        mask_features = self.vis_mask(x)
        batch_size, channels, height, width = mask_features.shape
        weight, bias = self._text_kernel(word)
        mask_out = self._dynamic_mask_conv(mask_features, weight, bias)

        if self.stage1:
            return mask_out, None, None, None, None

        gate = self._grasp_gate(mask_out, mask, (height, width))
        grasp_features = self.vis_grasp(x).reshape(
            batch_size, self.num_grasp_outputs, channels, height, width
        )
        grasp_features = grasp_features * gate.unsqueeze(1)

        # Equivalent to upstream's per-map convolutions, fused to avoid split
        # views with non-zero storage offsets on Ascend.
        grouped_input = grasp_features.reshape(
            1,
            batch_size * self.num_grasp_outputs * channels,
            height,
            width,
        )
        grouped_weight = (
            weight[:, None]
            .expand(-1, self.num_grasp_outputs, -1, -1, -1)
            .reshape(
                batch_size * self.num_grasp_outputs,
                channels,
                self.kernel_size,
                self.kernel_size,
            )
            .contiguous()
        )
        grouped_bias = (
            bias[:, None]
            .expand(-1, self.num_grasp_outputs)
            .reshape(batch_size * self.num_grasp_outputs)
            .contiguous()
        )
        grasp_out = F.conv2d(
            grouped_input,
            grouped_weight,
            bias=grouped_bias,
            padding=self.kernel_size // 2,
            groups=batch_size * self.num_grasp_outputs,
        ).reshape(
            batch_size, self.num_grasp_outputs, height, width
        )
        grasp_maps = tuple(
            grasp_out[:, index:index + 1].contiguous()
            for index in range(self.num_grasp_outputs)
        )
        return (mask_out, *grasp_maps)


class MapleGrasp(nn.Module):
    """Official MapleGrasp two-stage model adapted to the NPU runner."""

    def __init__(self, cfg):
        super().__init__()
        self.use_contrastive = bool(cfg.use_contrastive)
        self.use_pretrained_clip = bool(cfg.use_pretrained_clip)
        self.stage1 = bool(cfg.stage1)
        self.stage2 = bool(cfg.stage2)
        self.use_gt_obj_masks = bool(cfg.use_gt_obj_masks)
        self.predicts_grasp_short_side = self.stage2 and bool(
            getattr(cfg, "predict_grasp_short_side", False)
        )
        self.short_side_loss_weight = float(
            getattr(cfg, "short_side_loss_weight", 1.0)
        )
        self.align_grasp_size_loss = bool(
            getattr(cfg, "align_grasp_size_loss", False)
        )
        self.grasp_size_loss_activation = (
            "sigmoid" if self.align_grasp_size_loss else None
        )
        if self.stage1 == self.stage2:
            raise ValueError(
                "MapleGrasp requires exactly one of stage1 or stage2 to be True"
            )

        self.segmentation_only = self.stage1
        self.use_grasp_masks = self.stage2
        self.maplegrasp_stage = 1 if self.stage1 else 2

        clip_model = torch.jit.load(
            cfg.clip_pretrain, map_location="cpu"
        ).eval()
        print(f"Load pretrained CLIP: {self.use_pretrained_clip}")
        self.backbone = build_model(
            clip_model.state_dict(),
            cfg.word_len,
            self.use_pretrained_clip,
        ).float()
        self.neck = FPN(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out)
        if self.use_contrastive:
            print("Use contrastive learning module")
            self.decoder = TransformerDecoder(
                num_layers=cfg.num_layers,
                d_model=cfg.vis_dim,
                nhead=cfg.num_head,
                dim_ffn=cfg.dim_ffn,
                dropout=cfg.dropout,
                return_intermediate=cfg.intermediate,
            )
        else:
            print("Disable contrastive learning module")
        self.proj = MultiTaskProjectorPP(
            cfg.word_dim,
            cfg.vis_dim // 2,
            3,
            self.stage1,
            self.stage2,
            self.use_gt_obj_masks,
            self.predicts_grasp_short_side,
        )

    @staticmethod
    def _resize_target(target, output_size):
        if target is None:
            raise ValueError(
                "MapleGrasp Stage 2 training requires all grasp targets"
            )
        return F.interpolate(
            target, size=output_size, mode="nearest"
        ).detach()

    def forward(
        self,
        img,
        word,
        mask=None,
        grasp_qua_mask=None,
        grasp_sin_mask=None,
        grasp_cos_mask=None,
        grasp_wid_mask=None,
        grasp_short_mask=None,
        grasp_off_mask=None,
        grasp_off_weight=None,
    ):
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()
        visual = self.backbone.encode_image(img)
        word_features, text_state = self.backbone.encode_text(word)
        features = self.neck(visual, text_state)
        batch_size, channels, height, width = features.shape
        if self.use_contrastive:
            features = self.decoder(features, word_features, pad_mask)
            features = features.reshape(
                batch_size, channels, height, width
            )

        outputs = self.proj(features, text_state, mask)
        if mask is None:
            return outputs

        if self.stage1:
            if not self.training:
                return outputs[0].detach(), mask
            target_mask = self._resize_target(mask, outputs[0].shape[-2:])
            weight = target_mask * 0.5 + 1.0
            loss = F.binary_cross_entropy_with_logits(
                outputs[0], target_mask, weight=weight
            )
            loss_dict = {
                "m_ins": loss.item(),
                "m_qua": 0.0,
                "m_sin": 0.0,
                "m_cos": 0.0,
                "m_wid": 0.0,
            }
            return (
                (outputs[0].detach(), None, None, None, None),
                (target_mask, None, None, None, None),
                loss,
                loss_dict,
            )

        targets = (
            mask,
            grasp_qua_mask,
            grasp_sin_mask,
            grasp_cos_mask,
            grasp_wid_mask,
        )
        if self.predicts_grasp_short_side:
            targets = (*targets, grasp_short_mask)
        if not self.training:
            return tuple(item.detach() for item in outputs), targets

        targets = tuple(
            self._resize_target(target, outputs[0].shape[-2:])
            for target in targets
        )
        segmentation_loss = F.binary_cross_entropy_with_logits(
            outputs[0], targets[0], weight=targets[0] * 0.5 + 1.0
        )
        grasp_losses = [
            F.smooth_l1_loss(outputs[index], targets[index])
            for index in range(1, 4)
        ]
        size_losses = [
            F.smooth_l1_loss(
                torch.sigmoid(outputs[index])
                if self.align_grasp_size_loss
                else outputs[index],
                targets[index],
            )
            for index in range(4, len(outputs))
        ]
        grasp_losses.extend(size_losses)
        total_loss = segmentation_loss + sum(grasp_losses[:4])
        if self.predicts_grasp_short_side:
            total_loss = (
                total_loss
                + self.short_side_loss_weight * grasp_losses[4]
            )
        loss_dict = {
            "m_ins": segmentation_loss.item(),
            "m_qua": grasp_losses[0].item(),
            "m_sin": grasp_losses[1].item(),
            "m_cos": grasp_losses[2].item(),
            "m_wid": grasp_losses[3].item(),
        }
        if self.predicts_grasp_short_side:
            loss_dict["m_short"] = grasp_losses[4].item()
        return (
            tuple(item.detach() for item in outputs),
            targets,
            total_loss,
            loss_dict,
        )
