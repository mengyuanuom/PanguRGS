# OCID-VLG evaluation - 2026-08-10

This report combines the OCID-VLG test-set evaluations run with current model
code (`91d3c51` through `20c9a6c`) and compatibility evaluations for
pre-August-6 checkpoints using their training-era code (`b09d9bf`). The
intervening current-code commits only add evaluation profiles, tests, logs, and
this report; model forward code is unchanged from `91d3c51`.

## Scope

- Dataset split: `test`, 17,749 referring-expression samples.
- Protocol: `crog_legacy`.
- Hardware: 8 Ascend 910B3 NPUs. Primary tests used four NPUs per model;
  compatibility A/B runs used two NPUs per model.
- Loader: batch size 32 and 2 workers per NPU.
- Visualization: disabled.
- The evaluator shards samples without padding and sums all statistics across
  ranks before reporting metrics.

DROG and DROG-OFF were stored under `exp/OCID-VLG`. Four additional complete
runs were later found under the separately cased `exp/ocid_vlg`: LGD,
GGCNNCLIP, GRConvNetCLIP, and ETRG. The CROG run was stored separately under
`exp/OCID-VLG_multiple_npu` and is included here as well.

Before evaluation, the eight NPUs were occupied by `npu_resource_filler.py`.
That dedicated filler process and its orphaned workers were stopped. No NPU
process remained after all evaluations completed.

## Selected checkpoints

| Model | Epoch | Logged validation IoU | Logged validation J@1 | Logged validation J@5 | Logged width decode |
| --- | ---: | ---: | ---: | ---: | --- |
| DROG-OFF | 30 | 82.11 | 91.24 | 94.12 | sigmoid (matched) |
| DROG | 36 | 82.31 | 85.78 | 93.57 | sigmoid (mismatched) |
| CROG | 17 | 80.34 | 85.80 | 91.33 | sigmoid (mismatched) |
| LGD | 35 | 67.59 | 84.28 | 88.71 | sigmoid (legacy mismatch) |
| GRConvNetCLIP | 36 | 2.31 | 86.27 | 90.94 | sigmoid (legacy mismatch) |
| GGCNNCLIP | 36 | 2.31 | 20.26 | 24.32 | sigmoid (legacy mismatch) |
| ETRG | 34 | 76.66 | 73.55 | 78.77 | sigmoid (legacy mismatch) |

The DROG and DROG-OFF training logs report `world_size: 1` and batch size 32.
Both 36-epoch runs completed without an exception. Training time was 1 day
18:57:23 for DROG-OFF and 1 day 18:05:52 for DROG. The eight-NPU CROG run also
completed all 36 epochs in 1 day 17:22:39; its best J@1 checkpoint was epoch 17.

## Test results

| Model | IoU | Pr@50 | Pr@60 | Pr@70 | Pr@80 | Pr@90 | J@1 | J@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DROG-OFF | 81.23 | 97.02 | 95.66 | 89.19 | 68.54 | 22.33 | **89.30** | 92.94 |
| DROG | **81.38** | **97.39** | **96.40** | **89.67** | 67.82 | 22.03 | 89.10 | **93.58** |
| CROG | 79.34 | 95.68 | 93.21 | 84.53 | 63.48 | 17.01 | 88.17 | 91.36 |
| LGD (training-era code) | 65.94 | 88.38 | 79.14 | 54.71 | 26.96 | 0.02 | 84.94 | 87.27 |
| GRConvNetCLIP | 2.30 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 88.14 | 91.21 |
| GGCNNCLIP (training-era code) | 2.30 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 15.32 | 17.13 |
| ETRG | 74.97 | 90.21 | 87.27 | 78.70 | 53.71 | 12.66 | 73.73 | 76.85 |

DROG-OFF leads test J@1 by only 0.20 points, while DROG leads IoU by 0.15 and
J@5 by 0.64 points. The original DROG validation number is not directly
comparable because it used a mismatched sigmoid width decoder.

Among the additional models, GRConvNetCLIP is closest to DROG/DROG-OFF on
grasp success (88.14 J@1), followed by LGD (84.94). ETRG reaches 73.73, while
GGCNNCLIP is substantially weaker at 15.32. CROG reaches 88.17 J@1 after its
width decoder is aligned with the training loss. The IoU and Pr@50--90 columns
are not meaningful segmentation measures for GRConvNetCLIP or GGCNNCLIP: those
wrappers have no segmentation head and explicitly reuse the grasp-quality map
as `ins_pred`.

## DROG width-decoder A/B

The DROG checkpoint was saved with `grasp_size_activation=None`. The
`DROG.grasp_size_loss_activation = "clamp"` declaration was added after this
training run finished. At training time, the missing YAML field therefore fell
back to the evaluator's legacy sigmoid default even though DROG trained its raw
width output directly against the normalized target.

The same checkpoint was re-evaluated twice on the same 8,669-sample validation
split with current code. Only the width activation changed:

| Width decode | IoU | J@1 | J@5 |
| --- | ---: | ---: | ---: |
| clamp (matches DROG loss) | 82.31 | **89.62** | **94.50** |
| sigmoid (legacy mismatch) | 82.31 | 85.78 | 93.57 |

The sigmoid run exactly reproduces the historical training-log result, while
IoU remains identical. This isolates the apparent test-set gain to width
decoding rather than segmentation or an easier test split. The correct aligned
comparison is validation clamp 89.62/94.50 versus test clamp 89.10/93.58:
the test split is lower by 0.52 J@1 and 0.92 J@5. DROG-OFF remains sigmoid
because its width loss applies sigmoid during training.

## CROG width-decoder A/B

CROG also regresses normalized width directly, so its matching decoder is
clamp. Its checkpoint predates the model metadata declaration and the
training-time evaluator therefore used the legacy sigmoid fallback.

The same epoch-17 checkpoint was evaluated twice on the 8,669-sample
validation split. Only the width activation changed:

| Width decode | IoU | J@1 | J@5 |
| --- | ---: | ---: | ---: |
| clamp (matches CROG loss) | 80.34 | **88.67** | **92.19** |
| sigmoid (legacy mismatch) | 80.34 | 85.80 | 91.33 |

The sigmoid run exactly reproduces the historical training log. Switching to
the matching clamp decoder raises validation J@1 by 2.87 points and J@5 by
0.86, without changing IoU. The aligned comparison is validation clamp
88.67/92.19 versus test clamp 88.17/91.36: the test split is lower by 0.50
J@1 and 0.83 J@5, so there is no unexplained test-set increase.

## Legacy-checkpoint compatibility

All four additional checkpoints contain `grasp_size_activation=None`. Their
training-time evaluator therefore used the legacy sigmoid fallback. The model
losses regressed normalized width directly, so the aligned decoder is clamp.

LGD and GGCNNCLIP also require their training-era forward implementation.
Commit `1427e0a` (August 6, after these checkpoints were trained) added bounded
FiLM parameters and normalized CLIP text states to both models. LGD additionally
changed its internal width output in `91d3c51`. These changes preserve state-dict
shapes, so strict loading succeeds, but they change forward semantics enough to
invalidate direct evaluation of the old weights with current code.

The compatibility worktree at `b09d9bf` reproduces the historical validation
results before running aligned clamp test evaluation:

| Model/run | Split | Decoder | J@1 | J@5 | Interpretation |
| --- | --- | --- | ---: | ---: | --- |
| LGD, current code | test | clamp | 8.66 | 8.89 | incompatible; exclude |
| LGD, training-era code | val | sigmoid | 84.14 | 88.48 | reproduces 84.28/88.71 within diffusion randomness |
| LGD, training-era code | test | clamp | **84.94** | **87.27** | reported result |
| GGCNNCLIP, current code | test | clamp | 6.70 | 8.94 | incompatible; exclude |
| GGCNNCLIP, training-era code | val | sigmoid | 20.26 | 24.32 | exact historical reproduction |
| GGCNNCLIP, training-era code | test | clamp | **15.32** | **17.13** | reported result |

GRConvNetCLIP and ETRG did not receive a post-training forward-semantic change,
so their current-code clamp evaluations are valid. LGD diffusion sampling starts
from random noise; the small validation reproduction difference is expected
because the standalone test entry point does not set a deterministic seed.

## Checkpoint identity

```text
1b18ea348918854eac60ad55294f518e06a6b05613feeb946a202a37de0950e3  best_epoch_030_J1_91.24_J5_94.12.pth
93462598fe0967e256b034a843bc4fa45c64d8f223385777fa6591b01e47dd06  best_epoch_036_J1_85.78_J5_93.57.pth
c592238ba07f4cf82959f6909579688ed11058d82bfdc48fbf90d1457ffd6b53  best_epoch_035_J1_84.28_J5_88.71.pth
f75c4d1142b3fa16187bfeac914548fffd449be3a1c756607654a89919ab4a63  best_epoch_036_J1_86.27_J5_90.94.pth
0575999e856c42384ec8aca33caa22230d0b9a59a4060f661e23530b47cc7660  best_epoch_036_J1_20.26_J5_24.32.pth
e2ef4265a80c6caa9e1f106f5aa49b64fba5b70a2a610433573eed07b55c2a2e  best_epoch_034_J1_73.55_J5_78.77.pth
ba992ca78436ee47b32baa3162f9aeed3fc4e8b05f9223e03c74c1cf0616318b  best_epoch_017_J1_85.80_J5_91.33.pth
```

## Reproduction

Load the matching CANN environment first, then run one model per four-NPU
group. The model YAML files now define test batch size 32, 2 workers, and
metadata-driven grasp-size decoding.

```bash
source /data1/ma00959358/cann_path/cann-8.5.0/set_env.sh

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29611 \
  test_crog.py --config config/OCID-VLG/drogoff.yaml --opts \
  DATA.root_path /data1/ma00959358/pangu/CROG-NPU/datasets/OCID-VLG \
  TRAIN.resume exp/OCID-VLG/drogoff_ocid_vlg_8npu_20260801_001051_484/best_epoch_030_J1_91.24_J5_94.12.pth

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 torchrun \
  --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29613 \
  test_crog.py --config config/OCID-VLG/drog.yaml --opts \
  DATA.root_path /data1/ma00959358/pangu/CROG-NPU/datasets/OCID-VLG \
  TRAIN.resume exp/OCID-VLG/drog_ocid_vlg_8npu_20260801_001106_814/best_epoch_036_J1_85.78_J5_93.57.pth

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29615 \
  test_crog.py --config config/OCID-VLG/crog_multiple_r50.yaml --opts \
  DATA.root_path /data1/ma00959358/pangu/CROG-NPU/datasets/OCID-VLG \
  TRAIN.resume exp/OCID-VLG_multiple_npu/CROG_official_multiple_R50_8npu_20260801_001101_838/best_epoch_017_J1_85.80_J5_91.33.pth \
  TEST.grasp_size_activation clamp
```

The first DROG launch exposed missing TEST keys in `drog.yaml`; it stopped
before model or NPU initialization. The successful run used equivalent
effective settings from the complete DROG-OFF profile with the architecture
overridden to DROG. The configuration fix included with this report makes the
clean DROG command above directly reproducible.

Raw rank-0 logs are stored beside this report. In addition to the four DROG
logs, the valid additional-model logs are `grconvnetclip_test.txt`,
`etrg_test.txt`, `lgd_test_legacy_clamp.txt`, and
`ggcnnclip_test_legacy_clamp.txt`. Validation reproductions are
`lgd_val_legacy_sigmoid.txt` and `ggcnnclip_val_legacy_sigmoid.txt`; the two
explicitly excluded diagnostics are `lgd_current_code_incompatible.txt` and
`ggcnnclip_current_code_incompatible.txt`. CROG logs are
`crog_test_clamp.txt`, `crog_val_clamp.txt`, and `crog_val_sigmoid.txt`.
