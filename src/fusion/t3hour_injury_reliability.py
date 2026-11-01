#!/usr/bin/env python3
"""Top-decile injury-occurrence reliability of the configured compound pipeline.

For each city and seed: fit the crash-channel mean-power calibrator on the decision
window, re-derive the compound injury occurrence from the calibrated crash parameters
(conditional severity fixed), then compare the mean predicted occurrence with the
realized injured-event rate in the top 10% of issued forecasts (window space).

Output: experiments/results/joint/injury_top_decile.csv
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import nbinom

WS = __import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__), '../..'))
JT = f'{WS}/experiments/results/joint'
SEEDS = {'6666': '', '1234': '_s1234', '2025': '_s2025', '777': '_s777'}
CITY = dict(
    nyc=dict(t3='T3_region_hour', h0=24 * 1461, btr=24 * 1095,
             arm='joint3_cmp_cmp', vwin='val2024'),
    chi=dict(t3='T3_region_hour_chicago', h0=24 * 1461, btr=24 * 1095,
             arm='joint3_cmp_chi_cmp', vwin='val2024'),
    taichung=dict(t3='T3_region_hour_taichung', h0=26280, btr=17520,
                  arm='joint3_cmpzt_taichung_zt', vwin='val2023'),
)


def zinb_nll(y, mu, pi, a):
    n = 1 / a
    return -np.logaddexp(np.log(pi) + np.where(y == 0, 0.0, -30.0),
                         np.log1p(-pi) + nbinom.logpmf(y, n, n / (n + np.maximum(mu, 1e-6))))


def main():
    jr = pd.read_csv(f'{JT}/joint_results.csv')
    zta = pd.read_csv(f'{JT}/zt_analysis.csv')
    rng = np.random.default_rng(3)
    rows = []
    for c, cfg in CITY.items():
        L = np.load(f'{WS}/data/processed/{cfg["t3"]}/labels.npz')
        yc = L['region_hour'].astype(np.float64)
        yi = L['region_hour_injured'].astype(np.float64)
        for sd, sfx in SEEDS.items():
            dv = np.load(f'{JT}/mu_{cfg["arm"]}{sfx}_val.npz')
            dt = np.load(f'{JT}/mu_{cfg["arm"]}{sfx}_test.npz')
            a0 = float(jr[(jr.arm == cfg['arm'] + sfx) & (jr.window == cfg['vwin'])]['alpha_c'].iloc[0])
            nv = dv['mu_c'].shape[0]
            tv = (cfg['btr'] - 12 + 3 * np.arange(nv))[:, None] + 12 + np.arange(12)[None, :]
            yv = np.transpose(yc[:, tv], (1, 0, 2))
            mu = np.maximum(dv['mu_c'].astype(np.float64), 1e-6)
            pi = np.clip(dv['pi_c'].astype(np.float64), 1e-7, 1 - 1e-7)
            si = rng.choice(yv.size, size=min(2_000_000, yv.size), replace=False)
            yf, mf, pf = yv.ravel()[si], mu.ravel()[si], pi.ravel()[si]
            g = minimize_scalar(lambda x: zinb_nll(yf, mf ** x, pf, a0).mean(),
                                bounds=(0.5, 1.5), method='bounded').x
            a = float(np.exp(minimize_scalar(
                lambda la: zinb_nll(yf, mf ** g, pf, np.exp(la)).mean(),
                bounds=(-6, 3), method='bounded').x))
            ns = dt['mu_c'].shape[0]
            tt = (cfg['h0'] + 3 * np.arange(ns))[:, None] + 12 + np.arange(12)[None, :]
            yit = np.transpose(yi[:, tt], (1, 0, 2))
            mut = np.maximum(dt['mu_c'].astype(np.float64), 1e-6)
            pit = np.clip(dt['pi_c'].astype(np.float64), 1e-7, 1 - 1e-7)
            nc = 1 / a
            pc = nc / (nc + mut ** g)
            if c == 'taichung':
                pocc = 1 - (pit + (1 - pit) * pc ** nc)
            else:
                r2 = dt['r']
                a_s = float(zta[(zta.city == c) & (zta.seed == int(sd))]['cmp_alpha_s'].iloc[0])
                nss = 1 / a_s
                ps = nss / (nss + np.clip(r2, 1e-6, 50))
                t_ = ps ** nss
                pocc = 1 - (pit + (1 - pit) * (pc / (1 - (1 - pc) * t_)) ** nc)
            flat = pocc.ravel()
            real = (yit.ravel() > 0).astype(np.float64)
            sel = flat >= np.quantile(flat, 0.90)
            rows.append(dict(city=c, seed=int(sd), g3=round(float(g), 5), alpha=round(a, 5),
                             pred_occurrence=round(float(flat[sel].mean()), 5),
                             realized_occurrence=round(float(real[sel].mean()), 5),
                             realized_persons=round(float(yit.ravel()[sel].mean()), 5),
                             fraction=round(float(sel.mean()), 5)))
    D = pd.DataFrame(rows)
    D.to_csv(f'{JT}/injury_top_decile.csv', index=False)
    print(D.to_string(index=False))


if __name__ == '__main__':
    main()
