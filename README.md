# RACE: Rival-Aware Concept Evidence

Code for **RACE: Rival-Aware Concept Evidence for Fine-Grained Visual Recognition**,
by Yingqi Shi and Pengfei Wang, Beijing University of Posts and Telecommunications.

RACE combines instance-conditioned concept selection and local aggregation with
private/shared evidence relative to each candidate's strongest rival. CLIP stays
frozen; adapters, semantic offsets, classifier parameters, and score correction
are trained on cached features.

## Installation

Use Python 3.9 or newer. Install PyTorch and torchvision for your CPU/CUDA setup,
then install the remaining dependencies. Run the following commands from this
directory:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
```

The tests include a CPU training/evaluation run and moving the complete project
to a different directory before re-evaluating its checkpoint. CUDA is used for
the full experiments and the matched baseline trainers. The core RACE trainer
also supports `--device cpu`.

All examples use paths relative to this project. RACE entry points interpret
their path arguments relative to the project root, reject absolute paths, and
write relative paths into run records. Model downloads go to `cache/`. No image
datasets, feature caches, pretrained weights, historical training directories,
machine-specific paths, or credentials are included in this release.

## Directory layout

```text
race/          Public extraction, training, evaluation, and analysis commands
maft/          Model/loss components; internal names preserve checkpoint compatibility
scripts/       Matched baseline ports and shared helpers
configs/       Paper configuration, ablations, vocabularies, and few-shot indices
third_party/   Licensed prompt components and pinned upstream identities
tests/         Model-equation and relocation checks
results/       Reported numerical tables and validation notes
```

Generated data lives in `data/`, `features/`, `runs/`, `cache/`, and `figures/`.

## Data preparation

The expected layout is:

```text
data/
  CUB_200_2011/
    images/
    images.txt
    image_class_labels.txt
    train_test_split.txt
    bounding_boxes.txt          # only needed by pixel interventions
  oxford-iiit-pet/              # torchvision layout
  flowers-102/                 # torchvision layout
  fgvc-aircraft-2013b/         # torchvision layout
  stanford-cars-parquet/
    train-*.parquet
    test-*.parquet
```

Download CUB from its [dataset page](https://www.vision.caltech.edu/datasets/cub_200_2011/).
For Cars, use the uncorrupted original train/test Parquet files from
[tanganke/stanford_cars](https://huggingface.co/datasets/tanganke/stanford_cars),
preserving their shard names. The reader checks 8,144 development images,
8,041 test images, and 196 classes. Do not use corruption subsets.
Obtain datasets under their respective terms; this repository contains no images.

The other three datasets can be downloaded with:

```bash
python -m race.prepare_data --dataset pets
python -m race.prepare_data --dataset flowers
python -m race.prepare_data --dataset aircraft
```

Flowers combines official train+val for development. Pets uses official trainval.
Aircraft uses official trainval, variant labels, and removes its bottom 20-pixel
banner before cropping. Class and concept names are provided in
`configs/vocabularies/`; CUB attribute/part labels are not training inputs.

## Extract matched ten-view features

Each image produces one global view and a row-major 3x3 grid of overlapping
half-width/half-height crops. CLIP preprocessing is applied independently to all
ten views; no random augmentation is used.

```bash
python -m race.extract --dataset cub --baseline-caches
python -m race.extract --dataset pets --baseline-caches
python -m race.extract --dataset flowers --baseline-caches
python -m race.extract --dataset cars --baseline-caches
python -m race.extract --dataset aircraft --baseline-caches
```

The default is OpenAI `ViT-B-32-quickgelu`, FP32 encoding, normalization per view,
and FP16 feature storage. For dataset `cub`, the commands produce:

- `features/spl_fresh32_cub/`: normalized views for RACE and matched baselines.
- `features/spl_preprojection10_cub/`: real scaled preprojection means for ProLIP.
- `features/spl_labo_fresh10_cub/`: unnormalized projected means for LaBo.

Every RACE cache contains `metadata.pt`, `train.pt`, and `test.pt`. Metadata holds
the class/concept text embeddings and names; split files hold `views [N,10,D]`
and zero-based `label [N]`. You may set a different relative dataset `--root`,
cache `--output`, batch size, or number of workers.

The paper's component/design analyses used the original FP32 projected-view
extraction rather than the later matched preprojection pass. Regenerate those
caches explicitly:

```bash
python -m race.extract --dataset cub --profile ablation
python -m race.extract --dataset cars --profile ablation
python -m race.extract --dataset aircraft --profile ablation
```

These go to `features/DATASET_views_grid3/`. Use that directory as `--features`
for the corresponding ablations and CUB evidence analysis. The crop geometry,
class vocabulary, and model remain the same; the encoding pass is recorded.

## Train and evaluate RACE

```bash
python -m race.train --dataset cub --seed 1 --output runs/race_cub_s1
python -m race.evaluate --run runs/race_cub_s1
```

`configs/paper.json` specifies 160 epochs, AdamW, learning rate 0.001, weight
decay 0.01, batch size 128, identity/attack/defense weights 0.3/0.03/0.03,
and concept preservation weight 1. All datasets share these settings.
The development set is split by class with 10% validation, seeded as in the
paper. The earliest highest-validation checkpoint, including epoch 0, is saved.
Training never loads `test.pt`; evaluation loads it separately after selection.
Inference takes views only and does not use labels or test-set statistics.

Run all five datasets with seeds 1--3 after extraction:

```bash
bash scripts/run_race.sh
python -m race.summarize --runs runs/race_cub_s1 runs/race_cub_s2 runs/race_cub_s3
```

A run contains `config.json`, `result.json`, `best.pt`, `split.pt`,
`statistics.pt`, and, after evaluation, `evaluation_full.json`. The latter
contains every test prediction and label. Summaries use sample standard deviation.

The reported three-seed RACE accuracies (%) are CUB **78.79 ± 0.35**, Pets
**91.68 ± 0.42**, Flowers **95.22 ± 0.50**, Cars **82.30 ± 0.43**, and Aircraft
**54.10 ± 0.53**. Complete numerical tables are in `results/paper_tables.json`.
Hardware, encoder precision, library versions, and feature-cache identity should
be recorded when comparing reruns with these measured results.

## Ablations and preservation controls

Variant names and overrides are in `configs/variants.json`:
`base`, `correction_only`, `no_identity`, `no_attack`, `no_defense`,
`no_preservation`, `static_semantics`, `uniform_pooling`, `random_rival`,
`text_rival`, `no_support`, and `no_subtraction`.

```bash
python -m race.train --dataset cub --features features/cub_views_grid3 \
  --variant base --seed 1 --output runs/base_cub_s1
python -m race.evaluate --run runs/base_cub_s1
```

Use seeds 1--5 for the paper's component/design tables. Preservation controls use
seeds 1--3, with `random_preservation`, `random_preservation_gm`,
`rotated_preservation`, and `rotated_preservation_gm`. For rotations, specify
`--rotation-seed 0` through `4` and use distinct output directories, giving 15
runs per rotated setting. The gradient-matching coefficient is fixed after the
first nonzero preservation-gradient measurement, as in the original experiment.

## ViT-B/16 and 16-shot experiments

For each dataset, extract the B/16 BF16 caches with the original projected-view
encoding profile:

```bash
python -m race.extract --dataset cub --model ViT-B-16-quickgelu \
  --profile ablation --precision bf16
python -m race.train --dataset cub --features features/cub_b16_views_grid3 \
  --output runs/race_b16_cub_s1 --seed 1
```

The integer-only files in `configs/fewshot/` preserve the exact class-balanced
16-training/4-validation selections and development/test pools used in the
paper, including the CoOp Flowers test partition (2,463 images). They contain
no image data or filesystem paths. Source label ordering is verified before use.

```bash
python -m race.fewshot --dataset cub
python -m race.train --dataset cub --features features/spl_coop16_cub \
  --split-file features/spl_coop16_cub/fewshot_s1.pt \
  --seed 1 --output runs/fewshot_cub_s1
```

For the paper's few-shot selection, run the Cartesian grid of epochs `{50,160}`
and preservation weights `{1,3}` using separate JSON configs copied from
`configs/paper.json`. Select by validation accuracy before evaluating the chosen
checkpoint on test. Repeat for seeds 1--3, and for `base` where applicable.

## Matched baseline ports

See [BASELINES.md](BASELINES.md) for CoOp, CoCoOp, MaPLe, LaBo,
Tip-Adapter-F, ProKeR, and ProLIP commands. These follow the paper's frozen-backbone,
view, split, and validation budgets; they are adaptations of the named methods
to this protocol. Upstream repositories and revisions are recorded in
`third_party/versions.json`. The main RACE pipeline does not require cloning them.

## Evidence analysis and pixel interventions

Generate crop weights for each seed's CUB model trained on the analysis cache:

```bash
python -m race.evidence --run runs/race_cub_analysis_s1 \
  --output runs/race_cub_analysis_s1/evidence
```

Then run actual pixel interventions, re-encoding all ten views:

```bash
python -m race.pixels --runs runs/race_cub_analysis_s1 runs/race_cub_analysis_s2 \
  runs/race_cub_analysis_s3 --output runs/pixel_analysis
python -m race.plot_evidence --weights runs/race_cub_analysis_s1/evidence/weights.npz \
  --pixels runs/pixel_analysis/summary.json --output figures/evidence
```

The commands use one fixed-RNG CUB test image per class and private/shared/random
crop projections at equal pixel areas. Bboxes are used only by the
background control. Margin changes use the original image's rival as a fixed
measurement reference; inference still selects rivals normally. Ground-truth
labels select explanation branches for this analysis, not prediction inputs.
`--smoke` limits pixel analysis to two fixed images. Plotting produces PDF, SVG,
and 600-dpi PNG without any hard-coded font file location.

## License and citation

The project code is released under the [MIT License](LICENSE). Retained upstream
components keep their original notices; see [THIRD_PARTY.md](THIRD_PARTY.md).
Citation metadata is provided in `CITATION.cff`. A publication DOI and repository
URL can be added when available.
