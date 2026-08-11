# VCoT-GraspSet DROG-OFF evaluation - 2026-08-10

This report records the VCoT-GraspSet evaluation of the best DROG-OFF
checkpoint, the inference-decoding ablations used to diagnose it, and a
comparison with the results reported by the official VCoT-Grasp paper and
repository.

## Scope and provenance

- Evaluation code: commit `3d9afef6849ee68ae09def0045742cd842ebf007`.
- Dataset: VCoT-GraspSet compact splits in `datasets/graspanything-vcot`.
- Seen test split: 3,000 referring-grasp samples.
- Unseen test split: 1,487 referring-grasp samples.
- Model: `DROGOFF`, with predicted long and short grasp-rectangle sides.
- Checkpoint: best validation checkpoint selected at epoch 20.

- Hardware: one Ascend 910B3 NPU per evaluation.
- Loader: batch size 32 and 2 workers.
- Prediction protocol: top-1 rotated rectangle; successful when rotated IoU is
  at least 0.25 against one ground-truth grasp and the 180-degree-periodic
  orientation error is at most 30 degrees.

The paper writes the IoU condition as greater than 0.25, whereas this
repository implements greater than or equal to 0.25. Exact boundary hits are
not expected to affect the reported rounded results, but the distinction is
recorded for completeness.

## DROG-OFF results

The matched baseline uses sigmoid grasp-size decoding, size factor 300,
offsets enabled, segmentation-centre filtering disabled, offset geometry
resampling enabled, and inverse image-scale restoration enabled.

The selected calibrated profile changes only two options: offset geometry
resampling and inverse image-scale restoration are disabled. This makes the
old checkpoint's rectangles slightly larger and improves unseen success, at a
small cost on seen success.

| Profile | Split | Segmentation IoU | GraspSR |
| --- | --- | ---: | ---: |
| Matched baseline | Seen | 88.98 | 80.97 |
| Matched baseline | Unseen | 60.96 | 57.30 |
| Calibrated decoding | Seen | 88.98 | 80.77 |
| Calibrated decoding | Unseen | 60.96 | 58.71 |

The paper's `Avg.` column is the harmonic mean of Seen and Unseen rather than
their arithmetic mean. Applying the same definition,

```text
H = 2 * Seen * Unseen / (Seen + Unseen)
```

gives 67.11 for the matched baseline and **68.00** for calibrated decoding.
Calibrated decoding improves the harmonic mean by 0.89 points: it gains 1.41
points on Unseen while losing 0.20 points on Seen.

The complete calibrated precision breakdown is:

| Split | Pr@50 | Pr@60 | Pr@70 | Pr@80 | Pr@90 | GraspSR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Seen | 97.23 | 96.07 | 94.47 | 90.27 | 72.27 | 80.77 |
| Unseen | 65.16 | 63.28 | 60.46 | 55.62 | 42.17 | 58.71 |

## Comparison with VCoT-Grasp

The following paper results are from Tables III and IV of
[VCoT-Grasp](https://arxiv.org/html/2510.05827v1#S4.SS1). Values in the paper's
`Avg.` column are retained as published. The two DROG-OFF averages are computed
from the rounded Seen and Unseen results using the same harmonic-mean formula.

| Method | Seen | Unseen | Avg. / harmonic mean |
| --- | ---: | ---: | ---: |
| LGD (paper) | 38.67 | 13.42 | 19.93 |
| CLIP-Fusion (paper) | 52.40 | 13.51 | 21.48 |
| GG-CNN + CLIP (paper) | 56.33 | 17.89 | 27.16 |
| GR-ConvNet + CLIP (paper) | 70.80 | 33.29 | 45.29 |
| RT-Grasp (paper) | 58.93 | 44.79 | 50.80 |
| VCoT without visual CoT, MLP head (paper) | 67.60 | 49.36 | 57.06 |
| VCoT-Grasp, diffusion head (paper) | 57.50 | 41.29 | 48.07 |
| VCoT-Grasp, MLP head (paper) | 73.37 | 52.25 | 61.03 |
| **DROG-OFF, matched baseline (this repository)** | **80.97** | **57.30** | **67.11** |
| **DROG-OFF, calibrated decoding (this repository)** | **80.77** | **58.71** | **68.00** |
| VCoT-Grasp, LM head with new tokens (paper) | 82.89 | 58.80 | 68.80 |
| VCoT-Grasp, LM head with pretrained tokens (paper) | 83.60 | 58.98 | 69.16 |

The calibrated DROG-OFF result exceeds the VCoT-Grasp MLP head by 7.40 points
on Seen, 6.46 on Unseen, and 6.97 in harmonic mean. Against the paper's best LM
head, it is lower by 2.83 points on Seen, 0.27 on Unseen, and 1.16 in harmonic
mean. It therefore exceeds the MLP variant and all paper baselines, but does
not yet exceed the best LM-head result.

This is a score comparison under the same public split and grasp-success
criterion, not an equal-training-budget comparison. The paper trains its
PaliGemma2-3B models for three epochs at 224-pixel input resolution; this
DROG-OFF checkpoint is selected at epoch 20 and uses 448-pixel inputs.

## Unseen decoding ablations

All rows use the same checkpoint and the complete 1,487-sample Unseen split.
Segmentation IoU remains 60.96 because these options only affect grasp
decoding.

| Decoding change from matched baseline | GraspSR | Difference |
| --- | ---: | ---: |
| None: sigmoid, factor 300, offset on | 57.30 | - |
| Segmentation-centre filter on | 56.62 | -0.68 |
| Offset off | 53.19 | -4.11 |
| Offset geometry resampling off | 57.97 | +0.67 |
| Inverse image-scale restoration off | 58.04 | +0.74 |
| Resampling off and scale restoration off | **58.71** | **+1.41** |
| Filter on and offset off | 54.88 | -2.42 |
| Clamp instead of sigmoid | 0.00 | -57.30 |
| Size factor 100 instead of 300 | 0.61 | -56.69 |

The activation and scale-factor failures confirm that this checkpoint requires
sigmoid decoding and `grasp_size_factor: 300`. Offsets are also beneficial.
The calibrated profile is a checkpoint-specific compensation for modest size
underprediction; disabling mathematically correct inverse scaling is not, by
itself, evidence of a coordinate-system bug.

## Dataset and training diagnosis

The training log reaches roughly 98% training IoU while the best Unseen
GraspSR is 57.30 at epoch 20, indicating a substantial generalization gap.
Dataset inspection provides two strong explanations:

- The training split contains 415 observed object names, the Seen split 76,
  and the Unseen split 21. Name overlap is 74/76 between Train and Seen but
  only 1/21 between Train and Unseen.
- Training examples are imbalanced: `apple` contributes about 29.4%, `spoon`
  13.7%, and `banana` 9.8%; the top three names contribute about 52.9%.

Dense-target sampling also shows that grasp geometry occupies only about
5-6% of pixels. Current width, short-side, sine, and cosine SmoothL1 losses are
averaged over the whole feature map, so background pixels dominate geometry
optimization. In particular, background angle targets are `sin=0, cos=1`,
and the short-side loss becomes numerically close to zero despite sparse
positive supervision. Foreground-masked or foreground-weighted geometry losses
and class-balanced sampling are the highest-priority training improvements.

No long-side/short-side swap, angle-order mismatch, or hard canvas-coordinate
error was found. Original 416-pixel images are uniformly resized to 448 pixels,
and the inverse-transform implementation restores centres and dimensions
consistently.

## Paper real-world results

These results are not directly comparable with the current offline DROG-OFF
evaluation because DROG-OFF has not been tested with the same robot setup.
The paper fine-tunes on its real-world seen-object data and conducts five
physical trials per object.

| Method | Real-world Seen | Real-world Unseen |
| --- | ---: | ---: |
| GR-ConvNet + CLIP | 68% | 55% |
| RT-Grasp | 60% | 53% |
| VCoT-Grasp | 76% | 71% |

The paper's robustness study reports:

| Method | Original | Background change | Distractors |
| --- | ---: | ---: | ---: |
| GR-ConvNet + CLIP | 17/25 | 15/25 | 12/25 |
| RT-Grasp | 14/25 | 14/25 | 13/25 |
| VCoT-Grasp | 19/25 | 21/25 | 16/25 |

See the paper's
[real-world evaluation and robustness study](https://arxiv.org/html/2510.05827v1#S4.SS2)
and the official
[VCoT-Grasp repository](https://github.com/zhanghr2001/VCoT-Grasp#-model-zoo).

## Reproduction

Evaluate calibrated decoding on either split by changing `TEST.test_split`:

```bash
# Activate a compatible CANN/PyTorch NPU environment first.

ASCEND_RT_VISIBLE_DEVICES=0 python test_crog.py \
  --config config/vcot/drogoff.yaml \
  --opts \
  TRAIN.resume <path-to-epoch-20-checkpoint> \
  TEST.test_split unseen \
  TEST.grasp_size_activation sigmoid \
  TEST.use_offset_at_inference True \
  TEST.filter_grasps_by_segmentation False \
  TEST.offset_resample_geometry False \
  TEST.restore_grasp_size_scale False \
  TEST.test_batch_size 32 \
  TEST.test_workers 2
```

Raw logs are retained privately and are not published because they contain
machine-specific metadata.
