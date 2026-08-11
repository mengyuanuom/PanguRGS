import torch
import torch.nn as nn
import torch.nn.functional as F
from .crog_clip import build_model

# 你之前写好的 FiLM 模块，直接复用
class TextVisualFusionFiLM(nn.Module):
    def __init__(self, vis_dim: int, text_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.gamma = nn.Linear(hidden_dim, vis_dim)
        self.beta = nn.Linear(hidden_dim, vis_dim)

    def forward(self, feat, e_txt):
        # feat: (B, C, H, W), e_txt: (B, D)
        B, C, H, W = feat.shape
        h = self.mlp(e_txt)              # (B, hidden_dim)
        gamma = self.gamma(h).view(B, C, 1, 1)
        beta  = self.beta(h).view(B, C, 1, 1)
        return feat * (1.0 + gamma) + beta


class ResidualBlock(nn.Module):
    # 如果你原来是从 inference.models.grasp_model 里 import 的 ResidualBlock，
    # 就继续用原来的；这里写个占位以防报错
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        return F.relu(out)


class GenerativeResnetWithText(nn.Module):
    """
    原来的 GenerativeResnet + 文本条件（FiLM），不继承 GraspModel，只继承 nn.Module

    forward(x_in, e_txt) ->
        pos_output, cos_output, sin_output, width_output
    """

    def __init__(
        self,
        input_channels: int = 1,
        text_dim: int = 512,
        dropout: bool = False,
        prob: float = 0.0,
        channel_size: int = 32,
        fusion_hidden_dim: int = 128,
    ):
        super().__init__()

        # Encoder
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.res1 = ResidualBlock(128, 128)
        self.res2 = ResidualBlock(128, 128)
        self.res3 = ResidualBlock(128, 128)
        self.res4 = ResidualBlock(128, 128)
        self.res5 = ResidualBlock(128, 128)

        # ⭐ 文本-视觉 FiLM：在 128 通道的 bottleneck 上做调制
        self.fusion = TextVisualFusionFiLM(
            vis_dim=128,
            text_dim=text_dim,
            hidden_dim=fusion_hidden_dim
        )

        # Decoder
        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=2, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        self.conv6 = nn.ConvTranspose2d(32, 32, kernel_size=9, stride=1, padding=4)

        self.pos_output = nn.Conv2d(32, 1, kernel_size=2)
        self.cos_output = nn.Conv2d(32, 1, kernel_size=2)
        self.sin_output = nn.Conv2d(32, 1, kernel_size=2)
        self.width_output = nn.Conv2d(32, 1, kernel_size=2)

        self.dropout1 = nn.Dropout(p=prob if dropout else 0.0)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1.0)

    def forward(self, x_in, e_txt):
        """
        x_in:  (B, C, H, W)
        e_txt: (B, text_dim)
        """
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)

        # 文本条件融合
        x = self.fusion(x, e_txt)

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)

        pos_output = self.pos_output(self.dropout1(x))
        cos_output = self.cos_output(self.dropout1(x))
        sin_output = self.sin_output(self.dropout1(x))
        width_output = self.width_output(self.dropout1(x))

        return pos_output, cos_output, sin_output, width_output

    def compute_loss(self, xc, yc, e_txt):
        """
        跟你 GGCNNWithText 的 compute_loss 风格一致

        xc:  (B, C, H, W)
        yc:  (y_pos, y_cos, y_sin, y_width)
        e_txt: (B, text_dim)
        """
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc, e_txt)

        p_loss = F.smooth_l1_loss(pos_pred, y_pos)
        cos_loss = F.smooth_l1_loss(cos_pred, y_cos)
        sin_loss = F.smooth_l1_loss(sin_pred, y_sin)
        width_loss = F.smooth_l1_loss(width_pred, y_width)

        return {
            "loss": p_loss + cos_loss + sin_loss + width_loss,
            "losses": {
                "p_loss": p_loss,
                "cos_loss": cos_loss,
                "sin_loss": sin_loss,
                "width_loss": width_loss,
            },
            "pred": {
                "pos": pos_pred,
                "cos": cos_pred,
                "sin": sin_pred,
                "width": width_pred,
            },
        }

    def predict(self, xc, e_txt):
        pos_pred, cos_pred, sin_pred, width_pred = self(xc, e_txt)
        return {
            "pos": pos_pred,
            "cos": cos_pred,
            "sin": sin_pred,
            "width": width_pred,
        }


class GenerativeResnet_CLIP(nn.Module):
    grasp_size_loss_activation = "clamp"

    def __init__(self, cfg):
        super().__init__()

        self.use_pretrained_clip = cfg.use_pretrained_clip

        # 1) CLIP backbone
        clip_model = torch.jit.load(cfg.clip_pretrain,
                                    map_location="cpu").eval()
        print(f"[GenerativeResnet_CLIP] Load pretrained CLIP: {self.use_pretrained_clip}")
        self.backbone = build_model(
            clip_model.state_dict(),
            cfg.word_len,
            self.use_pretrained_clip
        ).float()

        in_ch = getattr(cfg, "input_channels", 3)
        text_dim = cfg.word_dim
        dropout = getattr(cfg, "dropout", False)
        prob = getattr(cfg, "dropout_prob", 0.0)

        self.grasp_head = GenerativeResnetWithText(
            input_channels=in_ch,
            text_dim=text_dim,
            dropout=dropout,
            prob=prob,
            channel_size=32,
            fusion_hidden_dim=128
        )

    def forward(
        self,
        img,
        word,
        ins_mask=None,
        grasp_qua_mask=None,
        grasp_sin_mask=None,
        grasp_cos_mask=None,
        grasp_wid_mask=None,
        grasp_off_mask=None,
        grasp_off_weight=None,
    ):
        # 文本 encode
        _, state = self.backbone.encode_text(word)   # (B, text_dim)

        # Grasp prediction
        pos_pred, cos_pred, sin_pred, wid_pred = self.grasp_head(img, state)
        ins_pred = pos_pred
        qua_pred = pos_pred

        if self.training:
            if grasp_qua_mask is not None and grasp_qua_mask.shape[-2:] != pos_pred.shape[-2:]:
                if ins_mask is not None:
                    ins_mask = F.interpolate(ins_mask, pos_pred.shape[-2:], mode='nearest').detach()
                grasp_qua_mask = F.interpolate(grasp_qua_mask, pos_pred.shape[-2:], mode='nearest').detach()
                grasp_sin_mask = F.interpolate(grasp_sin_mask, pos_pred.shape[-2:], mode='nearest').detach()
                grasp_cos_mask = F.interpolate(grasp_cos_mask, pos_pred.shape[-2:], mode='nearest').detach()
                grasp_wid_mask = F.interpolate(grasp_wid_mask, pos_pred.shape[-2:], mode='nearest').detach()

            p_loss   = F.smooth_l1_loss(qua_pred, grasp_qua_mask)
            cos_loss = F.smooth_l1_loss(cos_pred, grasp_cos_mask)
            sin_loss = F.smooth_l1_loss(sin_pred, grasp_sin_mask)
            wid_loss = F.smooth_l1_loss(wid_pred, grasp_wid_mask)

            total_loss = p_loss + cos_loss + sin_loss + wid_loss

            # 注意：保持 tensor，方便 DataParallel 聚合
            zero_like = p_loss.detach() * 0.0
            loss_dict = {
                "m_ins": zero_like,
                "m_qua": p_loss.detach(),
                "m_sin": sin_loss.detach(),
                "m_cos": cos_loss.detach(),
                "m_wid": wid_loss.detach(),
            }

            preds = (
                ins_pred,
                qua_pred,
                sin_pred,
                cos_pred,
                wid_pred,
            )
            targets = (
                ins_mask,
                grasp_qua_mask,
                grasp_sin_mask,
                grasp_cos_mask,
                grasp_wid_mask,
            )
            return preds, targets, total_loss, loss_dict

        else:
            preds = (
                ins_pred,
                qua_pred,
                sin_pred,
                cos_pred,
                wid_pred,
            )
            targets = (
                ins_mask,
                grasp_qua_mask,
                grasp_sin_mask,
                grasp_cos_mask,
                grasp_wid_mask,
            )
            return preds, targets
