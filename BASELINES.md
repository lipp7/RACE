# Matched baseline commands

Run all commands from the project root after `race.extract --baseline-caches`.
Repeat with datasets `cub`, `pets`, `flowers`, `cars`, `aircraft` and seeds
1--3, except LaBo, which is reported only for CUB, Flowers, and Aircraft.
All baseline trainers below require CUDA. Run each example's evaluation command
after validation selection. These are the actual protocol ports used in the
paper, rather than copies of published original-protocol scores.

## CoOp and Tip-Adapter-F

```bash
python -m scripts.train_reference --dataset cub --method coop --seed 1 \
  --features features/spl_fresh32_cub --feature-mode viewmean --output runs/coop_cub_s1
python -m scripts.evaluate_reference --run runs/coop_cub_s1

python -m scripts.train_reference --dataset cub --method tip_f --seed 1 \
  --features features/spl_fresh32_cub --feature-mode viewmean --output runs/tip_cub_s1
python -m scripts.evaluate_reference --run runs/tip_cub_s1
```

CoOp uses 200 epochs. Tip-Adapter-F uses 20 epochs with validation selection of
the checkpoint and cache hyperparameters. Image inputs are the normalized mean
of ten normalized view features.

## CoCoOp

```bash
python -m scripts.train_spl_cocoop_fresh --dataset cub --seed 1 --output runs/cocoop_cub_s1
python -m scripts.train_spl_cocoop_fresh --dataset cub --seed 1 --output runs/cocoop_cub_s1 --evaluate
```

The instance-conditioned prompt trainer uses batch one and ten epochs.

## MaPLe

```bash
python -m race.prepare_maple
python -m scripts.train_maple --dataset cub --seed 1 --output runs/maple_cub_s1
```

The trainer re-encodes all ten raw-image views with trainable prompt modules and
frozen OpenAI CLIP weights. It verifies input ordering and original-encoder
agreement against the fresh cache, regenerates the shared split from labels and
seed, and evaluates the validation-selected checkpoint automatically. Settings:
five epochs, batch four, learning rate 0.0035, prompt depth nine, two context
tokens, SGD momentum 0.9, weight decay 0.0005, BF16 autocast with FP32 parameters.
Use `--smoke` for a short runtime check rather than an accuracy reproduction.

The required MIT-licensed prompt/CLIP components are already included; their
small cache-location modification is described in `THIRD_PARTY.md`. No Dassl
installation or historical run registry is needed.

## ProKeR

```bash
python -m scripts.train_spl_proker_fresh --dataset cub --seed 1 --output runs/proker_cub_s1
python -m scripts.train_spl_proker_fresh --dataset cub --seed 1 --output runs/proker_cub_s1 --evaluate
```

The validation grid contains seven kernel scales and five regularizers. The
normalized mean of ten normalized views is used. The cached-feature solver
uses the original FP32 kernel and FP64 system solve/prediction convention.

## ProLIP

```bash
python -m scripts.train_spl_prolip --method prolip --dataset cub --seed 1 --output runs/prolip_cub_s1
python -m scripts.train_spl_prolip --method prolip --dataset cub --seed 1 --output runs/prolip_cub_s1 --evaluate
```

This uses real preprojection features from the extraction pass, not a synthetic
inverse of projected embeddings. There are 49 learning-rate/regularization
settings, each with 300 full-batch updates. Validation chooses the setting.

## LaBo

Fetch the pinned upstream candidate concepts; they are not redistributed here.
Prepare candidates for each of the three supported datasets and then select
concepts from training images separately for each seed:

```bash
python -m race.fetch_labo
python -m scripts.prepare_labo --dataset cub --raw
python -m scripts.prepare_labo --dataset flowers --raw
python -m scripts.prepare_labo --dataset aircraft --raw
python -m scripts.prepare_spl_labo_fresh
python -m scripts.train_spl_labo_fresh --dataset cub --seed 1 --output runs/labo_cub_s1
python -m scripts.train_spl_labo_fresh --dataset cub --seed 1 --output runs/labo_cub_s1 --evaluate
```

Candidate concepts retain the official unnormalized embedding convention. The
final classifier uses the mean of ten unnormalized projected view features.
Training budgets are 5,000 epochs on CUB and 10,000 on Flowers/Aircraft.

## Linear feature controls

The optional mean/concatenation controls from the reproducibility information
use the shared linear solver, for example:

```bash
python -m scripts.train_reference --dataset cub --method linear --seed 1 \
  --features features/spl_fresh32_cub --feature-mode viewconcat --output runs/linear_concat_cub_s1
python -m scripts.evaluate_reference --run runs/linear_concat_cub_s1
```

Use `viewmean` for the mean control. The solver selects among seven regularizers,
with at most 500 L-BFGS iterations per setting.
