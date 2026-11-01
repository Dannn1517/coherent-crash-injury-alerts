# Coherent Crash–Injury Early Warning under Patrol Budgets

Anonymous code and data release prepared for double-blind review. The repository contains no
author names, affiliations, email addresses, or machine-specific paths.

This release supports a study of sparse grid-hour crash forecasting in three cities. The model is
a 0.6M-parameter frequency-severity forecaster: a shared set-transformer encoder, a zero-inflated
crash head, and an injury head whose mean is tied to the expected crash count. The release includes
the exact grid-hour labels and windowed model inputs, the code used for training and evaluation,
and the per-arm result tables behind the reported numbers.

## Repository layout

```text
src/data/                 label builders (raw city records -> grid-hour labels)
src/fusion/               model, calibration, sequential decisions, dispatch, analyses
src/figures/              Figure 2 reproduction script
data/processed/           grid-hour labels and indices for three cities and two NYC resolutions
data/model_inputs/        windowed 12h -> 12h arrays used by the model
data/manifests/           build manifests, checksums, and the Chicago filter audit
experiments/results/      per-arm result CSVs and calibration parameters
figures/                  Figure 2 (regenerated from the released CSVs)
```

## Data at a glance

| City | Grid | Period | Test window | Crash channel | Injury channel |
|---|---|---|---|---|---|
| New York City | 20x20 UTM 18N | 2021–2025 | 2025 | all geocoded crashes | persons injured |
| Chicago | 20x20 UTM 16N | 2021–2025 | 2025 | all geocoded crashes | persons injured |
| Taichung | 20x20 UTM 51N | 2021–2024 | 2024 | A1/A2 (injury/fatal) crashes | persons injured |

The three label sets share one contract: a 20x20 grid, hourly counts, and whole-window splits
(train / decision / locked test). Taichung's public A1/A2 records cover injury and fatal crashes
only; property-damage-only A3 records are not public, so its crash channel approximates injury
crashes. See `DATA_SOURCES.md` for city-level source attribution and `data/model_inputs/README.md`
for array shapes.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Reproduce Figure 2 from the released result CSVs
python src/figures/make_figures_d70.py

# Train one configuration (New York City, free-gate head, seed 6666)
JOINT_FRESH=1 JOINT_SEED=6666 JOINT_ARMS=joint3_zi python src/fusion/t3hour_joint_hier.py
```

The training scripts append to `experiments/results/joint/joint_results.csv` and write per-arm
prediction arrays under the same directory. The shipped CSVs are the reference artifacts used in
the paper; work on a copy if you want to keep them unchanged. See `REPRODUCE.md` for the full
city/arm matrix and the analysis order.

## What is not included

Raw city records and the large per-cell prediction arrays are not redistributed. The raw sources
are public city portals listed in `DATA_SOURCES.md`; the label builders in `src/data/` document how
the released aggregates were produced. Model checkpoints are not included; the training commands
in `REPRODUCE.md` regenerate them.

## License

Code is released under the MIT License (`LICENSE`). The derived aggregate labels are provided for
research reproducibility and remain subject to the terms of the source city portals.
