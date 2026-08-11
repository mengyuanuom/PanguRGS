import torch
import torch.nn as nn
import torch.nn.functional as F

from .crog_clip import build_model

from .crog_layers import FPN, ProjectorOff, TransformerDecoder, MultiTaskProjectorOff



class CROGOFF(nn.Module):
    supports_offset = True

    def __init__(self, cfg):
        super().__init__()
        
        # Flags for ablation study
        self.use_contrastive = cfg.use_contrastive
        self.use_pretrained_clip = cfg.use_pretrained_clip
        self.use_grasp_masks = cfg.use_grasp_masks
        self.offset_loss_weight = float(getattr(cfg, "offset_loss_weight", 1.0))
        
        # Vision & Text Encoder
        clip_model = torch.jit.load(cfg.clip_pretrain,
                                    map_location="cpu").eval()
        print(f"Load pretrained CLIP: {self.use_pretrained_clip}")
        self.backbone = build_model(clip_model.state_dict(), cfg.word_len, self.use_pretrained_clip).float()
        # Multi-Modal FPN
        self.neck = FPN(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out)
        
        # Decoder
        if self.use_contrastive:
            print("Use contrastive learning module")
            # Decoder
            self.decoder = TransformerDecoder(num_layers=cfg.num_layers,
                                            d_model=cfg.vis_dim,
                                            nhead=cfg.num_head,
                                            dim_ffn=cfg.dim_ffn,
                                            dropout=cfg.dropout,
                                            return_intermediate=cfg.intermediate)
        else:
            print("Disable contrastive learning module")
        if self.use_grasp_masks:
            # Projector
            print("Use grasp masks")
            self.proj = MultiTaskProjectorOff(cfg.word_dim, cfg.vis_dim // 2, 3)
        else:
            print("Disable grasp masks")
            self.proj = ProjectorOff(cfg.word_dim, cfg.vis_dim // 2, 3)

    def forward(self, img, word, mask=None, grasp_qua_mask=None, grasp_sin_mask=None,
                grasp_cos_mask=None, grasp_wid_mask=None, grasp_off_mask=None,
                grasp_off_weight=None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''
        # padding mask used in decoder
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

        # vis: C3 / C4 / C5
        # word: b, length, 1024
        # state: b, 1024
        vis = self.backbone.encode_image(img)
        word, state = self.backbone.encode_text(word)

        # b, 512, 26, 26 (C4)
        fq = self.neck(vis, state)
        b, c, h, w = fq.size()
        
        if self.use_contrastive:
            fq = self.decoder(fq, word, pad_mask)
            fq = fq.reshape(b, c, h, w)

        if self.use_grasp_masks:
            
            # b, 1, 104, 104
            pred, grasp_qua_pred, grasp_sin_pred, grasp_cos_pred, grasp_wid_pred, grasp_off_pred = self.proj(fq, state)

            if self.training:
                target_size = pred.shape[-2:]
                mask = F.interpolate(mask, target_size, mode='nearest').detach()
                grasp_qua_mask = F.interpolate(grasp_qua_mask, target_size, mode='nearest').detach()
                grasp_sin_mask = F.interpolate(grasp_sin_mask, target_size, mode='nearest').detach()
                grasp_cos_mask = F.interpolate(grasp_cos_mask, target_size, mode='nearest').detach()
                grasp_wid_mask = F.interpolate(grasp_wid_mask, target_size, mode='nearest').detach()
                grasp_off_mask = F.interpolate(
                    grasp_off_mask, target_size, mode='bilinear', align_corners=False
                ).detach()
                if grasp_off_weight is None:
                    grasp_off_weight = (grasp_qua_mask > 0).float()
                else:
                    grasp_off_weight = F.interpolate(
                        grasp_off_weight, target_size, mode='nearest'
                    ).detach()
                # Ratio Augmentation
                total_area = mask.shape[2] * mask.shape[3]
                coef = 1 - (mask.sum(dim=(2,3)) / total_area)

                # Generate weight
                weight = mask * 0.5 + 1

                loss = F.binary_cross_entropy_with_logits(pred, mask, weight=weight)
                grasp_qua_loss = F.smooth_l1_loss(grasp_qua_pred, grasp_qua_mask)
                grasp_sin_loss = F.smooth_l1_loss(grasp_sin_pred, grasp_sin_mask)
                grasp_cos_loss = F.smooth_l1_loss(grasp_cos_pred, grasp_cos_mask)
                grasp_wid_loss = F.smooth_l1_loss(grasp_wid_pred, grasp_wid_mask)
                off_error = F.smooth_l1_loss(
                    grasp_off_pred, grasp_off_mask, reduction='none'
                )
                off_weight = grasp_off_weight.expand_as(off_error)
                grasp_off_loss = (off_error * off_weight).sum() / off_weight.sum().clamp_min(1.0)

                # @TODO adjust coef of different loss items
                total_loss = (loss + grasp_qua_loss + grasp_sin_loss +
                              grasp_cos_loss + grasp_wid_loss +
                              self.offset_loss_weight * grasp_off_loss)

                loss_dict = {}
                loss_dict["m_ins"] = loss.item()
                loss_dict["m_qua"] = grasp_qua_loss.item()
                loss_dict["m_sin"] = grasp_sin_loss.item()
                loss_dict["m_cos"] = grasp_cos_loss.item()
                loss_dict["m_wid"] = grasp_wid_loss.item()
                loss_dict["m_off"] = grasp_off_loss.item()

                # loss = F.binary_cross_entropy_with_logits(pred, mask, reduction="none").sum(dim=(2,3))
                # loss = torch.dot(coef.squeeze(), loss.squeeze()) / (mask.shape[0] * mask.shape[2] * mask.shape[3])

                return (pred.detach(), grasp_qua_pred.detach(), grasp_sin_pred.detach(), grasp_cos_pred.detach(), grasp_wid_pred.detach(), grasp_off_pred.detach()), (mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask, grasp_off_mask), total_loss, loss_dict
            else:
                return (pred.detach(), grasp_qua_pred.detach(), grasp_sin_pred.detach(), grasp_cos_pred.detach(), grasp_wid_pred.detach(), grasp_off_pred.detach()), (mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask, grasp_off_mask)

        else:
            # b, 1, 104, 104
            pred = self.proj(fq, state)

            if self.training:
                # resize mask
                if pred.shape[-2:] != mask.shape[-2:]:
                    mask = F.interpolate(mask, pred.shape[-2:],
                                        mode='nearest').detach()
                loss = F.binary_cross_entropy_with_logits(pred, mask)
                loss_dict = {}
                loss_dict["m_ins"] = loss.item()
                loss_dict["m_qua"] = 0
                loss_dict["m_sin"] = 0
                loss_dict["m_cos"] = 0
                loss_dict["m_wid"] = 0
                loss_dict["m_off"] = 0
                return (pred.detach(), None, None, None, None), (mask, None, None, None, None), loss, loss_dict
            else:
                return pred.detach(), mask
