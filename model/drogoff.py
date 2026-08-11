"""DROG with dense grasp-center offset refinement."""

import torch
import torch.nn.functional as F

from .drog import DROG
from .drog_projector import build_projector


class DROGOFF(DROG):
    """Combine DROG's DINOv2/CLIP fusion with CROG-OFF-style offsets."""

    supports_offset = True
    grasp_size_loss_activation = "sigmoid"

    def __init__(self, cfg):
        super().__init__(cfg)
        if not self.use_grasp_masks:
            raise ValueError("DROGOFF requires use_grasp_masks=True")
        self.predicts_grasp_short_side = bool(
            getattr(cfg, "predict_grasp_short_side", False)
        )
        self.proj = build_projector(cfg, with_offset=True)
        self.offset_loss_weight = float(getattr(cfg, "offset_loss_weight", 1.0))
        self.short_side_loss_weight = float(
            getattr(cfg, "short_side_loss_weight", 1.0)
        )

    def forward(self, img, word, mask=None, grasp_qua_mask=None,
                grasp_sin_mask=None, grasp_cos_mask=None,
                grasp_wid_mask=None, grasp_off_mask=None,
                grasp_off_weight=None, grasp_short_mask=None):
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()
        vis, word, state = self.fusion(
            img, word, self.txt_backbone, self.dinov2
        )
        features = self.neck(vis, state)
        b, c, h, w = features.shape
        features = self.decoder(features, word, pad_mask).reshape(b, c, h, w)

        outputs = self.proj(features, state)
        if self.predicts_grasp_short_side:
            seg, qua, sin, cos, width, short_side, offset = outputs
        else:
            seg, qua, sin, cos, width, offset = outputs
            short_side = None

        if mask is None:
            return outputs

        targets = (
            mask,
            grasp_qua_mask,
            grasp_sin_mask,
            grasp_cos_mask,
            grasp_wid_mask,
            grasp_off_mask,
        )
        if self.predicts_grasp_short_side:
            targets = (
                mask,
                grasp_qua_mask,
                grasp_sin_mask,
                grasp_cos_mask,
                grasp_wid_mask,
                grasp_short_mask,
                grasp_off_mask,
            )
        if not self.training:
            return tuple(x.detach() for x in outputs), targets

        # Only loss computation needs targets at the prediction resolution.
        target_size = seg.shape[-2:]
        mask = F.interpolate(mask, target_size, mode="nearest").detach()
        grasp_qua_mask = F.interpolate(
            grasp_qua_mask, target_size, mode="nearest"
        ).detach()
        grasp_sin_mask = F.interpolate(
            grasp_sin_mask, target_size, mode="nearest"
        ).detach()
        grasp_cos_mask = F.interpolate(
            grasp_cos_mask, target_size, mode="nearest"
        ).detach()
        grasp_wid_mask = F.interpolate(
            grasp_wid_mask, target_size, mode="nearest"
        ).detach()

        if self.predicts_grasp_short_side:
            if grasp_short_mask is None:
                raise ValueError(
                    "Short-side DROGOFF training requires grasp short-side maps"
                )
            grasp_short_mask = F.interpolate(
                grasp_short_mask, target_size, mode="nearest"
            ).detach()

        targets = (
            mask,
            grasp_qua_mask,
            grasp_sin_mask,
            grasp_cos_mask,
            grasp_wid_mask,
            grasp_off_mask,
        )

        if self.predicts_grasp_short_side:
            targets = (
                mask,
                grasp_qua_mask,
                grasp_sin_mask,
                grasp_cos_mask,
                grasp_wid_mask,
                grasp_short_mask,
                grasp_off_mask,
            )

        if grasp_off_mask is None or grasp_off_weight is None:
            raise ValueError("DROGOFF training requires offset and offset-weight maps")
        grasp_off_mask = F.interpolate(
            grasp_off_mask, target_size, mode="bilinear", align_corners=False
        ).detach()
        grasp_off_weight = F.interpolate(
            grasp_off_weight, target_size, mode="nearest"
        ).detach()
        targets = (*targets[:-1], grasp_off_mask)

        seg_weight = mask * 0.5 + 1.0
        seg_loss = F.binary_cross_entropy_with_logits(seg, mask, weight=seg_weight)
        qua_loss = F.smooth_l1_loss(qua, grasp_qua_mask)
        sin_loss = F.smooth_l1_loss(sin, grasp_sin_mask)
        cos_loss = F.smooth_l1_loss(cos, grasp_cos_mask)
        width_loss = F.smooth_l1_loss(
            torch.sigmoid(width), grasp_wid_mask
        )

        short_side_loss = (
            F.smooth_l1_loss(torch.sigmoid(short_side), grasp_short_mask)
            if self.predicts_grasp_short_side
            else None
        )
        offset_error = F.smooth_l1_loss(
            offset, grasp_off_mask, reduction="none"
        )
        offset_weight = grasp_off_weight.expand_as(offset_error)
        offset_loss = (offset_error * offset_weight).sum() / offset_weight.sum().clamp_min(1.0)

        total_loss = (
            seg_loss + qua_loss + sin_loss + cos_loss + width_loss
            + self.offset_loss_weight * offset_loss
        )
        if short_side_loss is not None:
            total_loss = total_loss + self.short_side_loss_weight * short_side_loss
        loss_dict = {
            "m_ins": seg_loss.detach(),
            "m_qua": qua_loss.detach(),
            "m_sin": sin_loss.detach(),
            "m_cos": cos_loss.detach(),
            "m_wid": width_loss.detach(),
            "m_off": offset_loss.detach(),
        }
        if short_side_loss is not None:
            loss_dict["m_short"] = short_side_loss.detach()
        return tuple(x.detach() for x in outputs), targets, total_loss, loss_dict
