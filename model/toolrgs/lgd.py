"""Language-driven grasp diffusion adapted to the ToolRGS dense-map contract.

The diffusion schedule and quality-map denoising follow the public LGD release
(`Fsoft-AIC/LGD`, MIT).  ToolRGS keeps its common five-map output contract, so
the published image/text/timestep conditioning is expressed as a dense grasp
decoder instead of the original repository's standalone training loop.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .crog_clip import build_model as build_clip_model


def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half, 1)
    )
    args = timesteps.float()[:, None] * frequencies[None]
    embedding = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = min(8, channels)
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, value):
        return value + self.block(value)


class LGDCore(nn.Module):
    """Predict clean grasp maps from RGB, text state, timestep, and noise."""

    def __init__(self, word_dim=1024, base_channels=32, time_dim=128):
        super().__init__()
        if base_channels % 8:
            raise ValueError("lgd_base_channels must be divisible by 8")
        hidden = base_channels * 4
        self.time_dim = time_dim

        def down_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(),
            )

        self.image_encoder = nn.Sequential(
            down_block(3, base_channels),
            down_block(base_channels, base_channels * 2),
            down_block(base_channels * 2, hidden),
        )
        self.noise_encoder = nn.Sequential(
            down_block(1, base_channels),
            down_block(base_channels, base_channels * 2),
            down_block(base_channels * 2, hidden),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, 1),
            ResidualBlock(hidden),
            ResidualBlock(hidden),
        )
        self.text_film = nn.Sequential(
            nn.Linear(word_dim, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden * 2),
        )
        self.time_film = nn.Sequential(
            nn.Linear(time_dim, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden * 2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden, base_channels * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels, base_channels, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            ResidualBlock(base_channels),
        )
        self.head = nn.Conv2d(base_channels, 5, 1)

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _apply_film(feature, parameters):
        gamma, beta = parameters.chunk(2, dim=1)
        gamma = torch.tanh(gamma)
        beta = torch.tanh(beta)
        return feature * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]

    def forward(self, image, noisy_quality, timesteps, text_state):
        output_size = image.shape[-2:]
        image_feature = self.image_encoder(image)
        noise_feature = self.noise_encoder(noisy_quality)
        feature = self.fusion(torch.cat((image_feature, noise_feature), dim=1))
        feature = self._apply_film(feature, self.text_film(text_state))
        time_state = timestep_embedding(timesteps, self.time_dim)
        feature = self._apply_film(feature, self.time_film(time_state))
        feature = self.decoder(feature)
        if feature.shape[-2:] != output_size:
            feature = F.interpolate(
                feature, size=output_size, mode="bilinear", align_corners=False
            )
        instance, quality, sine, cosine, width = self.head(feature).chunk(5, dim=1)
        return (
            instance,
            torch.tanh(quality),
            torch.tanh(sine),
            torch.tanh(cosine),
            torch.sigmoid(width),
        )


class CosineDiffusion(nn.Module):
    """Minimal x0-prediction diffusion used by the public LGD implementation."""

    def __init__(self, timesteps=1000, cosine_s=0.008):
        super().__init__()
        if timesteps < 2:
            raise ValueError("lgd_timesteps must be at least 2")
        steps = torch.arange(timesteps + 1, dtype=torch.float64)
        alpha_bar = torch.cos(
            ((steps / timesteps + cosine_s) / (1 + cosine_s)) * math.pi / 2
        ).pow(2)
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        betas = betas.clamp(0.0, 0.999).float()
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.timesteps = int(timesteps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def _extract(self, values, timesteps, reference):
        result = values.gather(0, timesteps)
        return result.reshape(-1, *([1] * (reference.ndim - 1))).to(reference.dtype)

    def q_sample(self, clean, timesteps, noise=None):
        if noise is None:
            noise = torch.randn_like(clean)
        alpha_bar = self._extract(self.alphas_cumprod, timesteps, clean)
        return alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise

    @torch.no_grad()
    def sample(self, denoiser, shape, device, sampling_steps=50):
        """Deterministic DDIM sampling over a subset of the training schedule."""
        sampling_steps = max(2, min(int(sampling_steps), self.timesteps))
        schedule = torch.linspace(
            self.timesteps - 1, 0, sampling_steps, device=device
        ).round().long().unique_consecutive()
        value = torch.randn(shape, device=device)
        prediction = None
        for index, timestep in enumerate(schedule):
            batch_t = torch.full(
                (shape[0],), int(timestep.item()), device=device, dtype=torch.long
            )
            prediction = denoiser(value, batch_t)
            clean = prediction[1].clamp(-1.0, 1.0)
            alpha = self.alphas_cumprod[timestep].to(value.dtype)
            if index + 1 == len(schedule):
                value = clean
                continue
            previous_timestep = schedule[index + 1]
            alpha_previous = self.alphas_cumprod[previous_timestep].to(value.dtype)
            epsilon = (value - alpha.sqrt() * clean) / (1.0 - alpha).sqrt().clamp_min(1e-8)
            value = alpha_previous.sqrt() * clean + (1.0 - alpha_previous).sqrt() * epsilon
        return prediction


class LGD(nn.Module):
    """ToolRGS port of Language-driven Grasp Detection."""

    grasp_size_loss_activation = "clamp"

    def __init__(self, cfg):
        super().__init__()
        clip_model = torch.jit.load(cfg.clip_pretrain, map_location="cpu").eval()
        self.backbone = build_clip_model(
            clip_model.state_dict(), cfg.word_len, cfg.use_pretrained_clip
        ).float()
        # LGDCore owns the visual encoder. Keep only CLIP's text branch trainable
        # so DDP and the optimizer do not carry an unused visual branch.
        for parameter in self.backbone.visual.parameters():
            parameter.requires_grad = False
        self.core = LGDCore(
            word_dim=cfg.word_dim,
            base_channels=getattr(cfg, "lgd_base_channels", 32),
            time_dim=getattr(cfg, "lgd_time_dim", 128),
        )
        self.diffusion = CosineDiffusion(getattr(cfg, "lgd_timesteps", 1000))
        self.sampling_steps = getattr(cfg, "lgd_sampling_steps", 50)
        self.diffusion_weight = getattr(cfg, "lgd_diffusion_weight", 1.0)
        self.contrastive_weight = getattr(cfg, "lgd_contrastive_weight", 0.0)
        self.contrastive_temperature = getattr(cfg, "lgd_contrastive_temperature", 0.1)
        self.geometry_mask_threshold = float(
            getattr(cfg, "lgd_geometry_mask_threshold", 0.05)
        )
        if self.contrastive_temperature <= 0:
            raise ValueError("lgd_contrastive_temperature must be positive")
        if not 0.0 <= self.geometry_mask_threshold <= 1.0:
            raise ValueError("lgd_geometry_mask_threshold must be in [0, 1]")

    @staticmethod
    def _resize_targets(output_size, instance, quality, sine, cosine, width):
        targets = (instance, quality, sine, cosine, width)
        resized = []
        for target in targets:
            if target is not None and target.shape[-2:] != output_size:
                target = F.interpolate(target, output_size, mode="nearest").detach()
            resized.append(target)
        return tuple(resized)

    @staticmethod
    def _quality_logit(clean_quality):
        probability = ((clean_quality + 1.0) * 0.5).clamp(1e-4, 1.0 - 1e-4)
        return torch.logit(probability)

    def _contrastive_loss(self, predicted, target):
        predicted = F.adaptive_avg_pool2d(predicted, (8, 8)).flatten(1)
        target = F.adaptive_avg_pool2d(target, (8, 8)).flatten(1)
        # Dense grasp maps are sparse. The default 1e-12 normalization epsilon
        # can create enormous gradients for an almost-empty pooled prediction,
        # especially on Ascend. Keep this optional ablation finite.
        predicted = F.normalize(predicted, dim=1, eps=1e-4)
        target = F.normalize(target, dim=1, eps=1e-4)
        logits = predicted @ target.t() / self.contrastive_temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, labels)

    def _masked_geometry_loss(self, predicted, target, quality):
        """Regress grasp geometry only where a grasp is supervised."""
        weight = (quality > self.geometry_mask_threshold).to(predicted.dtype)
        error = F.smooth_l1_loss(predicted, target, reduction="none")
        return (error * weight).sum() / weight.sum().clamp_min(1.0)

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
        _, text_state = self.backbone.encode_text(word)
        text_state = F.normalize(text_state.float(), dim=-1)
        targets = self._resize_targets(
            img.shape[-2:],
            ins_mask,
            grasp_qua_mask,
            grasp_sin_mask,
            grasp_cos_mask,
            grasp_wid_mask,
        )
        ins_mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask = targets

        if self.training:
            if grasp_qua_mask is None:
                raise ValueError("LGD training requires grasp quality supervision")
            clean_quality = grasp_qua_mask.clamp(0.0, 1.0).mul(2.0).sub(1.0)
            timesteps = torch.randint(
                0, self.diffusion.timesteps, (img.shape[0],), device=img.device
            )
            noisy_quality = self.diffusion.q_sample(clean_quality, timesteps)
            instance, quality, sine, cosine, width = self.core(
                img, noisy_quality, timesteps, text_state
            )

            instance_loss = F.binary_cross_entropy_with_logits(instance, ins_mask)
            quality_loss = F.smooth_l1_loss(quality, clean_quality)
            sine_loss = self._masked_geometry_loss(
                sine, grasp_sin_mask, grasp_qua_mask
            )
            cosine_loss = self._masked_geometry_loss(
                cosine, grasp_cos_mask, grasp_qua_mask
            )
            width_loss = self._masked_geometry_loss(
                width, grasp_wid_mask, grasp_qua_mask
            )
            # The public LGD training loop does not add a batch-wise
            # contrastive loss to the dense grasp-map objective. Retain it as
            # an explicit ablation only, and do not evaluate it at weight zero:
            # 0 * NaN is still NaN.
            if self.contrastive_weight > 0.0:
                contrastive_loss = self._contrastive_loss(
                    (quality + 1.0) * 0.5, grasp_qua_mask
                )
            else:
                contrastive_loss = quality.new_zeros(())
            total_loss = (
                instance_loss
                + self.diffusion_weight * quality_loss
                + sine_loss
                + cosine_loss
                + width_loss
                + self.contrastive_weight * contrastive_loss
            )
            quality_logit = self._quality_logit(quality)
            predictions = tuple(
                value.detach()
                for value in (instance, quality_logit, sine, cosine, width)
            )
            loss_dict = {
                "m_ins": instance_loss.detach(),
                "m_qua": quality_loss.detach(),
                "m_sin": sine_loss.detach(),
                "m_cos": cosine_loss.detach(),
                "m_wid": width_loss.detach(),
                "m_lgd_contrast": contrastive_loss.detach(),
            }
            return predictions, targets, total_loss, loss_dict

        def denoiser(noisy_quality, timesteps):
            return self.core(img, noisy_quality, timesteps, text_state)

        instance, quality, sine, cosine, width = self.diffusion.sample(
            denoiser,
            (img.shape[0], 1, img.shape[-2], img.shape[-1]),
            img.device,
            self.sampling_steps,
        )
        predictions = tuple(
            value.detach()
            for value in (
                instance,
                self._quality_logit(quality),
                sine,
                cosine,
                width,
            )
        )
        return predictions, targets
