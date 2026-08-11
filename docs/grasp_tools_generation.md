# Grasp-Tools balanced dataset generation

The generator creates a compositional referring-grasp dataset from labeled
single-object cutouts and unrelated background images. It writes one image per
scene, stores several language queries in the paired JSON file, and expands
those queries through index.jsonl.

## Source layout

Put or link the source files under the repository's asset directory:

~~~text
assets/grasp_tools/
├── graspall/
│   ├── 000000000001.jpg
│   ├── 000000000001.json
│   └── ...
└── backgrounds/
    ├── background_001.jpg
    └── ...
~~~

Every source JSON must contain an objects list. Each valid object needs a
canonical/recognized category, a polygon mask, and at least one grasp. The
generator checks that all 22 categories are present and reports skipped invalid
objects in metadata.json.

## Recommended generation

From the CROG-NPU root:

~~~bash
python tools/dataset_converters/grasp_tools/augment.py --overwrite
~~~

The default inputs are `assets/grasp_tools/graspall` and
`assets/grasp_tools/backgrounds`; the generated dataset is written to
`datasets/grasp-tools/aug_graspall_v2`.

The defaults produce:

| Split | Images | Objects/image | Queries/image | Approx. query samples |
|---|---:|---:|---:|---:|
| train | 6000 | 2–3 | 4 | 24000 |
| val | 500 | 2–3 | 4 | 2000 |
| test | 1000 | 2–3 | 4 | 4000 |
| total | 7500 | 2–3 | — | 30000 |

Use a quick integration run before full generation:

~~~bash
python tools/dataset_converters/grasp_tools/augment.py   --smoke-test   --out-dir datasets/grasp-tools/smoke   --overwrite
~~~

## Balance guarantees

The complete plan for each split is created before rendering. Consequently:

- placements of the 22 categories differ by at most one;
- query-target counts of the 22 categories differ by at most one;
- source instances belonging to the same category are reused equally, with a
  maximum count difference of one;
- scene generation is atomic, so a failed placement retries the whole scene and
  cannot silently alter the planned counts;
- object counts 2 and 3 are used equally for the default difficulty-1 split.

The generator aborts if any guarantee is violated. Exact counts and deltas are
written to metadata.json.

## Language diversity

Each category has four safe surface forms: its canonical name plus aliases or
near-synonyms. Training has 22 command templates, giving 88 category-only
command/term combinations per category. The default difficulty-1 data uses
unique-category targets and shares the same language pool across train, val,
and test. During training, `dynamic_train_prompts: True` selects a reproducibly
shuffled, non-repeating expression for every sample and epoch. Validation and
test keep their generated text fixed. Existing schema-v2.1 JSON files without
the `prompt_cycle` marker remain compatible.

The canonical category is always stored separately in each query, so changing
wording does not change the target label.

## Grasp geometry during evaluation

The source backgrounds are 1280×720 while model inputs are smaller. Grasp
centres and dense maps are inverse-warped to the original canvas for the CROG
Jacquard scorer. The normalized long-side prediction is multiplied by the
inverse resize scale at that point. The short side remains exactly 20 pixels
for models without a dedicated short-side head, matching the generated labels
and the legacy CROG evaluation protocol.
