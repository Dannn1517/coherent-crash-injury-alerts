# Reproduction guide

The repository ships the exact result CSVs used in the paper. The commands below regenerate the
main joint/SetTF results and the analysis chain from the released labels and windowed inputs. Some
baseline CSVs are reference-only artifacts whose generation scripts or intermediate files are not
part of this release; Section 2 lists them. A single 24GB GPU is sufficient; each main training run
takes a few minutes on an RTX 3090.

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins CPU/CUDA-agnostic versions. For CUDA 12.4, install the matching PyTorch
wheel if your platform needs it:

```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

All commands are run from the repository root. The scripts resolve data and result paths relative
to that root; no absolute paths are required.

## 2. Reference artifacts

The shipped files under `experiments/results/` are the paper's artifacts. Training appends to
`joint_results.csv` / `settf_results.csv` and skips arms already listed there. To retrain from
scratch, move the two reference CSVs aside first (or work on a copy of the repository):

```bash
mkdir -p /tmp/reference_results
mv experiments/results/joint/joint_results.csv /tmp/reference_results/
mv experiments/results/settf/settf_results.csv /tmp/reference_results/
```

`JOINT_FRESH=1` and `SETTF_FRESH=1` additionally bypass the completed-arm check.

Calibration note: the configured family set is restricted to gate-preserving refits
(mean-power + dispersion); gate-Platt families are computed for diagnostics but are not
selected, because gate recalibration degraded crash-channel calibration error.

Reference-only artifacts (no generation script in this release): `baselines_t3/*` (GSNet, IGSTGNN,
ZI-Tweedie transplant evaluations), `stllm_transfer/*` (reproduced LLM-augmented baseline),
`fusion_t3hour/gnn_results.csv` (GNN+GRU design grid), and `joint/support_mixture_probe.csv`
(post-hoc support-repair probe). `joint/qa_supplement_newmachine.csv` records the exact Table 1
crash-AUC values (including the Taichung arms), the EB floor AUCs quoted in the Table 1 caption,
and the support margins quoted in Section 5.2.

## 3. Main joint model

The three reported arms are free-gate (`joint3_zi`), compound NB (`joint3_cmp`), and compound ZTNB
(`joint3_cmpzt`). The matched family is NB for New York City and Chicago, and ZTNB for Taichung.
Four seeds are used: `6666`, `1234`, `2025`, `777`.

| City | `JOINT_T3` | `JOINT_DATA_C` | `JOINT_DATA_I` | tag: free-gate / compound NB / compound ZTNB | boundaries |
|---|---|---|---|---|---|
| New York City | default | default | default | `` / `_cmp` / `_zt` | defaults |
| Chicago | `data/processed/T3_region_hour_chicago/labels.npz` | `data/model_inputs/crash_hour_chi` | `data/model_inputs/crash_hour_chi_inj` | `_chi` / `_chi_cmp` / `_chi_zt` | defaults |
| Taichung | `data/processed/T3_region_hour_taichung/labels.npz` | `data/model_inputs/crash_hour_taichung` | `data/model_inputs/crash_hour_taichung_inj` | `_taichung_zi` / `_taichung_cmp` / `_taichung_zt` | `17520` / `26280`, val 2023, test 2024 |

Example, New York City free-gate:

```bash
JOINT_FRESH=1 JOINT_SEED=6666 JOINT_ARMS=joint3_zi \
  python src/fusion/t3hour_joint_hier.py
```

Example, Taichung compound ZTNB:

```bash
JOINT_FRESH=1 JOINT_SEED=6666 JOINT_TAG=_taichung_zt JOINT_ARMS=joint3_cmpzt \
  JOINT_T3=data/processed/T3_region_hour_taichung/labels.npz \
  JOINT_DATA_C=data/model_inputs/crash_hour_taichung \
  JOINT_DATA_I=data/model_inputs/crash_hour_taichung_inj \
  JOINT_B_TRAIN_END=17520 JOINT_B_VAL_END=26280 \
  JOINT_VAL_YEAR=2023 JOINT_TEST_YEAR=2024 \
  python src/fusion/t3hour_joint_hier.py
```

Outputs: `experiments/results/joint/ckpt_<arm><tag>.pt`,
`experiments/results/joint/mu_<arm><tag>_{val,test}.npz`, and appended rows in
`experiments/results/joint/joint_results.csv`.

## 4. Marginal SetTF baseline (calibration and independent control)

| City | channel | `SETTF_TAG` | `SETTF_DATA_DIR` | `SETTF_LABEL_KEY` | boundaries |
|---|---|---|---|---|---|
| New York City | crash | `` | `data/model_inputs/crash_hour` | `region_hour` | defaults |
| New York City | injury | `_inj` | `data/model_inputs/crash_hour_inj` | `region_hour_injured` | defaults |
| Chicago | crash | `_chi` | `data/model_inputs/crash_hour_chi` | `region_hour` | defaults |
| Chicago | injury | `_chi_inj` | `data/model_inputs/crash_hour_chi_inj` | `region_hour_injured` | defaults |
| Taichung | crash | `_taichung` | `data/model_inputs/crash_hour_taichung` | `region_hour` | `17520` / `26280`, val 2023, test 2024 |
| Taichung | injury | `_taichung_inj` | `data/model_inputs/crash_hour_taichung_inj` | `region_hour_injured` | `17520` / `26280`, val 2023, test 2024 |

Taichung boundaries are passed as `SETTF_B_TRAIN_END=17520`, `SETTF_B_VAL_END=26280`,
`SETTF_VAL_YEAR=2023`, and `SETTF_TEST_YEAR=2024`.

Example, Chicago crash marginal:

```bash
SETTF_FRESH=1 SETTF_SEED=6666 SETTF_TAG=_chi SETTF_LABEL_KEY=region_hour \
  SETTF_DATA_DIR=data/model_inputs/crash_hour_chi \
  SETTF_T3=data/processed/T3_region_hour_chicago/labels.npz \
  python src/fusion/t3hour_settf.py
```

## 5. Analysis order

Run the analyses after the corresponding predictions exist. The scripts write the result CSVs
listed below; the released copies are the reference outputs.

```bash
# Calibration (decision-window fit, locked-test reporting)
python src/fusion/t3hour_settf_calibration.py      # calibration_results.csv, calibration_params.json
python src/fusion/t3hour_injury_calibration.py     # calibration_results_injury.csv
python src/fusion/t3hour_tail_calib.py             # tail_calib_real.csv

# Family comparison and paired tests
python src/fusion/t3hour_zt_analysis.py            # zt_analysis.csv
python src/fusion/t3hour_cmp_analysis.py           # cmp_summary.csv
python src/fusion/t3hour_cmp_fixes.py              # cmp_fixes.csv
python src/fusion/t3hour_paired_tests.py           # paired_tests.csv

# Operational layer
python src/fusion/t3hour_sequential_capacity.py    # sequential_capacity*.csv, sequential_capacity_hourly*.csv
python src/fusion/t3hour_e2e_unified.py            # e2e_unified_calib.csv, sequential_capacity_calib*.csv
python src/fusion/t3hour_injury_reliability.py      # injury_top_decile.csv (compound top-decile reliability)
python src/fusion/t3hour_pipeline_e2e.py           # pipeline_e2e_supplement.csv
python src/fusion/t3hour_dispatch_sim.py           # dispatch_sim.csv
python src/fusion/t3hour_independent_control.py    # independent_dual_control.csv

# Figure 2
python src/figures/make_figures_d70.py             # figures/f2_evidence.pdf
```

## 6. Input-delay sensitivity

The sensitivity analysis reuses the trained free-gate weights and shifts only the validation/test
history by 24 or 72 hours. Run it after the free-gate checkpoints exist; `JOINT_MIN_ES=0` forces a
pure re-evaluation without training:

```bash
JOINT_SEED=6666 JOINT_ARMS=joint3_zi JOINT_SHIFT_H=24 JOINT_EVAL_TAG=_sh24 JOINT_MIN_ES=0 \
  python src/fusion/t3hour_joint_hier.py
python src/fusion/t3hour_latency_sensitivity.py    # latency_shift_sensitivity.csv
```

Use the same city/tag arguments as the corresponding main run. Repeat for `JOINT_SHIFT_H=72` and
for seed `1234` to match the released sensitivity table.

## 7. Rebuilding labels from raw records (optional)

The builders in `src/data/` expect the raw city files described in `DATA_SOURCES.md`. They write to
`data/processed/` and use fixed city grid bounding boxes; no external export directory is needed.
The released labels and windowed arrays already make the training and analysis steps runnable
without this step.

```bash
python src/data/build_t3_region_hour.py
python src/data/build_t3_region_hour_chicago.py
python src/data/build_t3_region_hour_taichung.py
```

To rebuild the 12h -> 12h model arrays from labels:

```bash
T3_DIR=data/processed/T3_region_hour LABEL_KEY=region_hour \
  OUT_DIR=data/model_inputs/crash_hour python src/data/build_t3_stllm_format.py
```

For Taichung, set `T3_DIR=data/processed/T3_region_hour_taichung`,
`B_TRAIN_END_H=17520`, `B_VAL_END_H=26280`, and `LABEL_KEY=region_hour` or
`region_hour_injured`. For the New York City resolution variants, set `T3_DIR` to the g10/g40
labels and `OUT_DIR` to the corresponding `data/model_inputs/crash_hour_g*` directory.
