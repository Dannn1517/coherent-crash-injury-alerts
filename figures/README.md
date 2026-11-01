# Figure 2

`f2_evidence.pdf` is regenerated from `experiments/results/joint/zt_analysis.csv` and
`experiments/results/joint/sequential_capacity_ci.csv`:

```bash
python src/figures/make_figures_d70.py
```

Panel (a) shows the coherence-price gradient by conditional family; panel (b) shows the sequential
per-cycle recall gain over the empirical-Bayes baseline with 7-day-block paired 95% intervals.
