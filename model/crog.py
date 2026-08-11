import torch
import torch.nn as nn
import torch.nn.functional as F

from model.clip import build_model
from utils.pretrained import ensure_pretrained

from .layers import FPN, MultiTaskProjector, Projector, TransformerDecoder


class CROG(nn.Module):
    grasp_size_loss_activation = "clamp"

    def __init__(self, cfg):
        super().__init__()
        self.use_contrastive = cfg.use_contrastive
        self.use_pretrained_clip = cfg.use_pretrained_clip
        self.use_grasp_masks = cfg.use_grasp_masks
        self.predicts_grasp_short_side = bool(
            getattr(cfg, "predict_grasp_short_side", False)
        )
        self.short_side_loss_weight = float(
            getattr(cfg, "short_side_loss_weight", 1.0)
        )
        if self.predicts_grasp_short_side and not self.use_grasp_masks:
            raise ValueError(
                "CROG short-side prediction requires use_grasp_masks=True"
            )

        clip_pretrain = ensure_pretrained(cfg.clip_pretrain, "clip-rn50")
        clip_model = torch.jit.load(
            str(clip_pretrain), map_location="cpu"
        ).eval()
        print(f"Load pretrained CLIP: {self.use_pretrained_clip}")
        self.backbone = build_model(
            clip_model.state_dict(), cfg.word_len, self.use_pretrained_clip
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

        if self.use_grasp_masks:
            print("Use grasp masks")
            self.proj = MultiTaskProjector(
                cfg.word_dim,
                cfg.vis_dim // 2,
                3,
                predict_short_side=self.predicts_grasp_short_side,
            )
        else:
            print("Disable grasp masks")
            self.proj = Projector(cfg.word_dim, cfg.vis_dim // 2, 3)

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
    ):
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()
        vis = self.backbone.encode_image(img)
        word, state = self.backbone.encode_text(word)

        features = self.neck(vis, state)
        batch_size, channels, height, width = features.shape
        if self.use_contrastive:
            features = self.decoder(features, word, pad_mask).reshape(
                batch_size, channels, height, width
            )

        if not self.use_grasp_masks:
            pred = self.proj(features, state)
            if mask is None:
                return pred
            if self.training:
                mask = F.interpolate(
                    mask, pred.shape[-2:], mode="nearest"
                ).detach()
                loss = F.binary_cross_entropy_with_logits(pred, mask)
                loss_dict = {
                    "m_ins": loss.item(),
                    "m_qua": 0,
                    "m_sin": 0,
                    "m_cos": 0,
                    "m_wid": 0,
                }
                return (pred.detach(), None, None, None, None), (
                    mask, None, None, None, None
                ), loss, loss_dict
            return pred.detach(), mask

        outputs = self.proj(features, state)
        if self.predicts_grasp_short_side:
            pred, qua, sin, cos, width, short_side = outputs
        else:
            pred, qua, sin, cos, width = outputs
            short_side = None

        if mask is None:
            return outputs

        targets = (mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask,
                   grasp_wid_mask)
        if self.predicts_grasp_short_side:
            targets = (*targets, grasp_short_mask)
        if not self.training:
            return tuple(output.detach() for output in outputs), targets

        target_size = pred.shape[-2:]
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
                    "Short-side CROG training requires grasp short-side maps"
                )
            grasp_short_mask = F.interpolate(
                grasp_short_mask, target_size, mode="nearest"
            ).detach()

        targets = (mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask,
                   grasp_wid_mask)
        if self.predicts_grasp_short_side:
            targets = (*targets, grasp_short_mask)

        seg_weight = mask * 0.5 + 1.0
        seg_loss = F.binary_cross_entropy_with_logits(
            pred, mask, weight=seg_weight
        )
        qua_loss = F.smooth_l1_loss(qua, grasp_qua_mask)
        sin_loss = F.smooth_l1_loss(sin, grasp_sin_mask)
        cos_loss = F.smooth_l1_loss(cos, grasp_cos_mask)
        width_loss = F.smooth_l1_loss(width, grasp_wid_mask)
        short_side_loss = (
            F.smooth_l1_loss(short_side, grasp_short_mask)
            if self.predicts_grasp_short_side
            else None
        )

        total_loss = seg_loss + qua_loss + sin_loss + cos_loss + width_loss
        if short_side_loss is not None:
            total_loss = (
                total_loss
                + self.short_side_loss_weight * short_side_loss
            )

        loss_dict = {
            "m_ins": seg_loss.item(),
            "m_qua": qua_loss.item(),
            "m_sin": sin_loss.item(),
            "m_cos": cos_loss.item(),
            "m_wid": width_loss.item(),
        }
        if short_side_loss is not None:
            loss_dict["m_short"] = short_side_loss.item()
        return (
            tuple(output.detach() for output in outputs),
            targets,
            total_loss,
            loss_dict,
        )
