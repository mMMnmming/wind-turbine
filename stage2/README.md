# Stage 2: Wind-aware graph pointer-network training

This directory is the project’s second delivery, corresponding to the paper’s
third stage: DRL routing after Stage 1 deployment. It trains from scratch; the
paper provides neither a checkpoint nor complete hyperparameters.

## GPU setup

The training code defaults to `cuda`; PyTorch is therefore mandatory. For a
CUDA 12.1 environment, use the supplied pinned dependency file:

```powershell
pip install -r stage2\requirements-cuda121.txt
```

For a different CUDA version, obtain the matching command from the official
[PyTorch selector](https://pytorch.org/get-started/locally/), then install the
remaining neutral dependencies with `pip install -r stage2\requirements.txt`.
The default device is `cuda` and deliberately fails when CUDA is absent. Use
`--device cpu` only to debug a small run.

## Commands

```powershell
python -m stage2.run prepare_data
python -m stage2.run train
python -m stage2.run train --epochs 1 --batches-per-epoch 1 --batch-size 8 --device cpu
python -m stage2.run resume --checkpoint stage2\outputs\latest.pt --epochs 120
python -m stage2.run evaluate --input turbine87.tsv --checkpoint stage2\outputs\latest.pt
python -m stage2.run evaluate --regions-json another_stage1_regions.json --checkpoint stage2\outputs\latest.pt
```

`turbine102.tsv` supplies the training regions. The 87-turbine evaluation
partition is intentionally not fixed yet.

`config.json` is the single default-training snapshot.  It makes 32 fixed-seed
KDE virtual Walney-102 fields by default, reruns Stage 1 for every field, and
combines their legal local regions with the three real regions. Use
`--virtual-fields 0` for a quick real-data-only smoke test. `resume` requires
`--checkpoint`; its `--epochs` is the total target epoch count, not an
additional count.

## Constraint semantics

The pointer decoder uses both a visited-node mask and a dynamic endurance mask:
at every step a candidate is available only when flying to it, inspecting it,
and returning to the nest remains within the equivalent-energy budget. Each
final route is then independently checked by `WindATSPInstance` before it is
reported. The virtual farm KDE, network depth, seed, and training schedule are
reproduction assumptions, not unpublished paper parameters.
## Evaluation artifacts

Evaluation writes a separate, non-overwriting result directory by default:
`stage2/outputs/evaluations/<dataset>_<checkpoint>/`. It contains
`evaluation.log`, CSV/JSON route audits, a grouped cost comparison, a four-region
model-route overview, and one high-resolution route figure per region.

```powershell
python -m stage2.run evaluate --input turbine87.tsv --checkpoint stage2\outputs\best.pt
python -m stage2.run evaluate --input turbine87.tsv --checkpoint stage2\outputs\best.pt --random-samples 1000 --plot-dpi 300
```

The comparison contains the model greedy route, a directed nearest-neighbour
baseline, and fixed-seed random permutations. Infeasible baselines are retained
and explicitly marked rather than silently repaired.
