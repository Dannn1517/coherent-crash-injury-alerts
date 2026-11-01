# Result artifacts

These CSVs are the per-arm outputs used for the paper. City codes are `nyc`, `chi`, and
`taichung`. The files are reference artifacts: running the training or analysis scripts appends to
or overwrites some of them, so keep a copy if you want to compare.

| Paper item | File |
|---|---|
| Table 1, forecast quality | `joint/joint_results.csv`; crash AUC and EB floor AUC: `joint/qa_supplement_newmachine.csv` |
| Table 2, coherence price | `joint/zt_analysis.csv`, `joint/cmp_summary.csv`, `joint/paired_tests.csv` |
| Table 5, baselines | `baselines_t3/*`, `fusion_t3hour/gnn_results.csv`, `stllm_transfer/transfer_results.csv`, `settf/settf_results.csv` |
| Table 3, sequential capacity | `joint/sequential_capacity.csv`, `joint/sequential_capacity_ci.csv`, `joint/sequential_capacity_hourly*.csv`, `settf/independent_dual_control.csv` |
| Table 4, end-to-end pipeline | ECE: `joint/e2e_unified_calib.csv`; EB diagnostics and calibrated cost: `joint/pipeline_e2e_supplement.csv`; `joint/sequential_capacity_calib.csv`; `settf/calibration_params.json` |
| Figure 2a, coherence price | `joint/zt_analysis.csv` |
| Figure 2b, sequential gain | `joint/sequential_capacity_ci.csv` |
| Compound injury top-decile reliability | `joint/injury_top_decile.csv` |
| Calibration diagnostics | `settf/calibration_results.csv`, `settf/calibration_results_injury.csv`, `settf/calibration_perhorizon_test.csv`, `settf/reliability_settf_zinb_calib.csv`, `settf/tail_calib_real.csv` |
| Input-delay sensitivity | `joint/latency_shift_sensitivity.csv` |
| Dispatch simulation | `joint/dispatch_sim.csv` |

## File groups

- `joint/`: joint frequency-severity runs, sequential decisions, calibration re-evaluation,
  dispatch simulation, and sensitivity analyses.
- `settf/`: marginal SetTF-ZINB baseline, calibration parameters, independent dual-head control.
- `baselines_t3/`: GSNet, IGSTGNN, and ZI-Tweedie transplant evaluations.
- `stllm_transfer/`: reproduced LLM-augmented baseline.
- `fusion_t3hour/`: GNN+GRU count-likelihood grid.

The `arm` columns encode the model family, city tag, and seed. The seed suffix is empty for 6666
and `_s1234`, `_s2025`, `_s777` otherwise. `mu_*.npz` and `ckpt_*.pt` are not included; the
training scripts regenerate them.
