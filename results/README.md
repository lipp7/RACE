# Numerical records and release validation

`paper_tables.json` contains the measured main/reproducibility table entries.
Values are percentages and sample standard deviations. They are reference
records and are never read by the model or training pipeline.

Validation performed while preparing this release:

- The released factory loaded a paper CUB checkpoint and matched the original
  selected model on four cached CUB examples: final/base logits, concept weights,
  local attention, private/shared evidence, utility, rivals, and total objective.
  Maximum absolute differences in the compared floating-point outputs were zero.
- CPU unit checks cover forward equations, auxiliary losses, backward gradients,
  concept-response preservation, and the base/no-subtraction variants.
- A two-epoch synthetic CPU run trains without any test cache present, then
  evaluates, relocates the whole project, and evaluates again with identical
  predictions and metrics.

These checks validate the refactoring and portability. The full five-dataset
GPU training matrix, all baseline fits, and image re-encoding experiments were
not rerun as part of release packaging. Reported table values remain the measured
paper results. Record your dependency versions and encoder precision for reruns.

The CPU validation environment used Python 3.9, PyTorch 2.5.1+cpu,
torchvision 0.20.1+cpu, and NumPy 1.26.4.
