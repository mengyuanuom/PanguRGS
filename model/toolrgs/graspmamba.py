"""GraspMamba paper reimplementation for the ToolRGS dense-map contract.

The GraspMamba authors have not released their training code or checkpoints.
This module follows the architecture described in the paper: a four-stage
MambaVision backbone, frozen CLIP text features, hierarchical multimodal
fusion, and a top-down grasp decoder.  ToolRGS adds a dedicated instance head
and exposes dense quality/sine/cosine/width maps so all datasets can use the
shared engine.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .crog_clip import build_model as build_clip_model


MAMBAVISION_CHANNELS = {
    "mamba_vision_T": (80, 160, 320, 640),
    "mamba_vision_T2": (80, 160, 320, 640),
    "mamba_vision_S": (96, 192, 384, 768),
    "mamba_vision_B": (128, 256, 512, 1024),
    "mamba_vision_L": (196, 392, 784, 1568),
    "mamba_vision_L2": (196, 392, 784, 1568),
}


class MambaVisionFeatureExtractor(nn.Module):
    """Expose the four pre-downsample feature maps from official MambaVision."""

    def __init__(self, model_name, pretrained=True, checkpoint=None):
        super().__init__()
        if model_name not in MAMBAVISION_CHANNELS:
            available = ", ".join(sorted(MAMBAVISION_CHANNELS))
            raise ValueError(
                f"Unsupported MambaVision model {model_name!r}; available: {available}"
            )
        try:
            from mambavision import create_model
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "GraspMamba requires the optional MambaVision dependency. "
                "Install it with `pip install -r requirement-mamba.txt` after "
                "installing the torch/torch_npu pair matching this Ascend server. "
                "MambaVision NPU support depends on its installed operator build."
            ) from exc

        model_kwargs = {"num_classes": 0}
        if checkpoint:
            checkpoint = Path(checkpoint)
            if pretrained:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
            model_kwargs["model_path"] = str(checkpoint)
        try:
            self.model = create_model(
                model_name,
                pretrained=pretrained,
                **model_kwargs,
            )
        except Exception as exc:
            raise RuntimeError(
                "Unable to construct the MambaVision backbone. If automatic weight "
                "download is unavailable, download the official checkpoint and set "
                "TRAIN.mamba_pretrain to its local path; use "
                "TRAIN.mamba_pretrained=False only for training from scratch."
            ) from exc

        if not hasattr(self.model, "levels") or len(self.model.levels) != 4:
            raise RuntimeError(
                "The installed mambavision package does not expose the expected "
                "four-stage `levels` API; ToolRGS supports mambavision==1.2.0."
            )

        self.channels = MAMBAVISION_CHANNELS[model_name]
        self._captured = {}
        self._feature_hooks = []
        for index in range(3):
            downsample = getattr(self.model.levels[index], "downsample", None)
            if downsample is None:
                raise RuntimeError(
                    f"MambaVision stage {index} has no downsample module to hook"
                )
            self._feature_hooks.append(
                downsample.register_forward_pre_hook(self._capture_input(index))
            )
        self._feature_hooks.append(
            self.model.levels[3].register_forward_hook(self._capture_output(3))
        )

    def _capture_input(self, index):
        def hook(_module, inputs):
            self._captured[index] = inputs[0]

        return hook

    def _capture_output(self, index):
        def hook(_module, _inputs, output):
            self._captured[index] = output

        return hook

    def forward(self, image):
        self._captured = {}
        self.model.forward_features(image)
        missing = [index for index in range(4) if index not in self._captured]
        if missing:
            raise RuntimeError(f"MambaVision failed to expose stages {missing}")
        features = tuple(self._captured[index] for index in range(4))
        for index, (feature, channels) in enumerate(zip(features, self.channels)):
            if feature.ndim != 4 or feature.shape[1] != channels:
                raise RuntimeError(
                    f"Unexpected MambaVision stage {index} shape "
                    f"{tuple(feature.shape)}; expected channel dimension {channels}"
                )
        self._captured = {}
        return features


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class HierarchicalFeatureFusion(nn.Module):
    """Paper equations (5)-(8): multimodal fusion at every visual scale."""

    def __init__(self, visual_channels, text_dim, fusion_dim):
        super().__init__()
        self.visual_projections = nn.ModuleList(
            nn.Conv2d(channels, fusion_dim, kernel_size=1)
            for channels in visual_channels
        )
        # A linear layer is equivalent to the paper's 1x1 convolution before
        # the sentence vector is expanded over the spatial feature map.
        self.text_projections = nn.ModuleList(
            nn.Linear(text_dim, fusion_dim) for _ in visual_channels
        )
        self.fusion_blocks = nn.ModuleList(
            ConvNormAct(2 * fusion_dim, fusion_dim) for _ in visual_channels
        )
        self.top_down = nn.ModuleList(
            ConvNormAct(fusion_dim, fusion_dim)
            for _ in range(len(visual_channels) - 1)
        )

    def forward(self, features, text_state):
        if len(features) != len(self.visual_projections):
            raise ValueError(
                f"Expected {len(self.visual_projections)} visual levels, "
                f"received {len(features)}"
            )

        multimodal = []
        for feature, visual_projection, text_projection, fusion in zip(
            features,
            self.visual_projections,
            self.text_projections,
            self.fusion_blocks,
        ):
            visual = visual_projection(feature)
            text = text_projection(text_state).unsqueeze(-1).unsqueeze(-1)
            text = text.expand(-1, -1, visual.shape[-2], visual.shape[-1])
            multimodal.append(fusion(torch.cat((visual, text), dim=1)))

        output = multimodal[-1]
        for level in range(len(multimodal) - 2, -1, -1):
            output = self.top_down[level](output)
            output = F.interpolate(
                output,
                size=multimodal[level].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            output = output + multimodal[level]
        return output


class GraspMapDecoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.refine_half = ConvNormAct(channels, channels)
        self.refine_full = ConvNormAct(channels, channels)
        self.instance_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.quality_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.sine_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.cosine_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.width_head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, feature, output_size):
        half_size = tuple(max(1, size // 2) for size in output_size)
        feature = F.interpolate(
            feature, size=half_size, mode="bilinear", align_corners=False
        )
        feature = self.refine_half(feature)
        feature = F.interpolate(
            feature, size=output_size, mode="bilinear", align_corners=False
        )
        feature = self.refine_full(feature)
        return (
            self.instance_head(feature),
            self.quality_head(feature),
            self.sine_head(feature),
            self.cosine_head(feature),
            self.width_head(feature),
        )


class GraspMamba(nn.Module):
    """ToolRGS-compatible paper reimplementation of GraspMamba."""

    def __init__(self, cfg):
        super().__init__()
        mamba_model = getattr(cfg, "mamba_model", "mamba_vision_T")
        self.backbone = MambaVisionFeatureExtractor(
            model_name=mamba_model,
            pretrained=getattr(cfg, "mamba_pretrained", True),
            checkpoint=getattr(cfg, "mamba_pretrain", None),
        )

        clip_checkpoint = Path(cfg.clip_pretrain)
        if not clip_checkpoint.is_file():
            raise FileNotFoundError(
                f"CLIP checkpoint not found: {clip_checkpoint}. GraspMamba uses "
                "ToolRGS's local CLIP text encoder."
            )
        clip_model = torch.jit.load(str(clip_checkpoint), map_location="cpu").eval()
        self.txt_backbone = build_clip_model(
            clip_model.state_dict(), cfg.word_len, cfg.use_pretrained_clip
        ).float()
        # The paper freezes the text encoder during grasp training.
        for parameter in self.txt_backbone.parameters():
            parameter.requires_grad = False

        fusion_dim = getattr(cfg, "graspmamba_fusion_dim", 128)
        self.fusion = HierarchicalFeatureFusion(
            visual_channels=self.backbone.channels,
            text_dim=cfg.word_dim,
            fusion_dim=fusion_dim,
        )
        self.decoder = GraspMapDecoder(fusion_dim)
        self.instance_loss_weight = getattr(
            cfg, "graspmamba_instance_loss_weight", 1.0
        )

    def train(self, mode=True):
        super().train(mode)
        # Keep the frozen CLIP tower deterministic even when the rest trains.
        self.txt_backbone.eval()
        return self

    @staticmethod
    def _resize_targets(output_size, targets):
        resized = []
        for target in targets:
            if target is not None and target.shape[-2:] != output_size:
                target = F.interpolate(target, output_size, mode="nearest").detach()
            resized.append(target)
        return tuple(resized)

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
        del grasp_off_mask, grasp_off_weight
        with torch.no_grad():
            _, text_state = self.txt_backbone.encode_text(word)
        features = self.backbone(img)
        fused = self.fusion(features, text_state)
        outputs = self.decoder(fused, img.shape[-2:])
        targets = self._resize_targets(
            outputs[0].shape[-2:],
            (
                ins_mask,
                grasp_qua_mask,
                grasp_sin_mask,
                grasp_cos_mask,
                grasp_wid_mask,
            ),
        )

        if not self.training:
            return tuple(output.detach() for output in outputs), targets

        if any(target is None for target in targets):
            raise ValueError(
                "GraspMamba training requires instance, quality, sine, cosine, "
                "and width supervision"
            )
        instance, quality, sine, cosine, width = outputs
        instance_gt, quality_gt, sine_gt, cosine_gt, width_gt = targets
        instance_loss = F.binary_cross_entropy_with_logits(instance, instance_gt)
        quality_loss = F.binary_cross_entropy_with_logits(quality, quality_gt)
        sine_loss = F.smooth_l1_loss(sine, sine_gt)
        cosine_loss = F.smooth_l1_loss(cosine, cosine_gt)
        width_loss = F.smooth_l1_loss(torch.sigmoid(width), width_gt)
        total_loss = (
            self.instance_loss_weight * instance_loss
            + quality_loss
            + sine_loss
            + cosine_loss
            + width_loss
        )
        loss_dict = {
            "m_ins": instance_loss.detach(),
            "m_qua": quality_loss.detach(),
            "m_sin": sine_loss.detach(),
            "m_cos": cosine_loss.detach(),
            "m_wid": width_loss.detach(),
        }
        return (
            tuple(output.detach() for output in outputs),
            targets,
            total_loss,
            loss_dict,
        )
