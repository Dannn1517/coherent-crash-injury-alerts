# Windowed model inputs

Each directory follows the same contract: 12 hours of history, 12 hours ahead, stride 3.

```text
x: (windows, 12, cells, 3)  float32
   x[..., 0] = crash or injury count (raw, standardized inside the training script)
   x[..., 1] = time of day
   x[..., 2] = day of week
y: (windows, 12, cells, 1)  float32
   future counts for the same channel
```

| Directory | City / channel | Cells | Train / val / test windows |
|---|---|---:|---|
| `crash_hour` | New York City, crash | 400 | 8,753 / 2,921 / 2,913 |
| `crash_hour_inj` | New York City, injury | 400 | 8,753 / 2,921 / 2,913 |
| `crash_hour_chi` | Chicago, crash | 400 | 8,753 / 2,921 / 2,913 |
| `crash_hour_chi_inj` | Chicago, injury | 400 | 8,753 / 2,921 / 2,913 |
| `crash_hour_taichung` | Taichung, crash | 400 | 5,833 / 2,913 / 2,921 |
| `crash_hour_taichung_inj` | Taichung, injury | 400 | 5,833 / 2,913 / 2,921 |
| `crash_hour_g10` | New York City, 10x10 crash | 100 | 8,753 / 2,921 / 2,913 |
| `crash_hour_g10_inj` | New York City, 10x10 injury | 100 | 8,753 / 2,921 / 2,913 |
| `crash_hour_g40` | New York City, 40x40 crash | 1,600 | 8,753 / 2,921 / 2,913 |
| `crash_hour_g40_inj` | New York City, 40x40 injury | 1,600 | 8,753 / 2,921 / 2,913 |

The arrays are the exact inputs used for the reported runs. All cities and resolution variants use
the midpoint time-of-day encoding `(hour+0.5)/24`; rebuilding from labels with
`src/data/build_t3_stllm_format.py` reproduces the same encoding. Use the shipped arrays when
comparing against the released result tables.

`adj_mx.pkl` files from the original build are not needed by the set-transformer models and are not
included.
