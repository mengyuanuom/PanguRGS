# Embedded Grasp-Tools source assets

This directory contains the complete source material required by CROG-NPU's
compositional Grasp-Tools augmentation pipeline:

- `graspall/`: 107 RGB images and their 107 polygon/grasp JSON annotations;
- `backgrounds/`: 42 tool-free background images.

The annotation categories are canonicalized to 22 classes. Two empty object
records in `000000000076.json` are retained in the source annotation for
provenance and are skipped with explicit warnings by the generator.

Do not commit generated `aug_graspall_v2` scenes here. Generate them below the
normal `datasets/grasp-tools/` data root by following
`docs/grasp_tools_generation.md`.
