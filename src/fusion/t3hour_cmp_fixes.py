#  补算 (方法学审查 M2/M3/M5/M7 + minor 1/2/3/4/9):
#  (a) cmp_analysis 的 dyn_vs_obs 末seed泄漏修复 → 4-seed 均值;
#  (b) 支撑质量/截断尾质量/不一致上界 全 seed 重算并落 csv;
#  (c) 臂间逐点配对双簇 bootstrap: cpl-vs-zi, cmp-vs-zi(分城), joint-vs-marginal(crash/inj);
#  (d) 机制: 拟合 P(y_i=0) vs 观测伤情零率 (分城, cmp/zi);
#  (e) α_s 逐 seed; (f) 边际伤情 exact-NB AUC/R20 4-seed 登记.
import numpy as np, pandas as pd, os, json
from scipy.stats import nbinom
WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
R = f'{WS}/data/model_inputs'; JT = f'{WS}/experiments/results/joint'
SET = f'{WS}/experiments/results/settf'; OUT = JT
jr = pd.read_csv(f'{JT}/joint_results.csv')
SEEDS = ['', '_s1234', '_s2025', '_s777']
CITY = {'nyc': ('crash_hour', 'crash_hour_inj'), 'chi': ('crash_hour_chi', 'crash_hour_chi_inj')}
rows = []

def labels(city, ch):
    tag = CITY[city][0 if ch == 'c' else 1]
    return np.transpose(np.load(f'{R}/{tag}/test.npz')['y'][:, :, :, 0], (0, 2, 1)).astype(np.float64)

def load(arm, city):
    d = np.load(f'{JT}/mu_{arm}_test.npz')
    return d

def boot_pair(diff_windows, days):
    S = len(diff_windows)
    w = diff_windows[np.random.default_rng(7).integers(0, S, (500, S))].mean(1)
    ud = np.unique(days); dm = np.array([diff_windows[days == d].mean() for d in ud])
    d_ = dm[np.random.default_rng(8).integers(0, len(ud), (500, len(ud)))].mean(1)
    return (float(np.quantile(w, .025)), float(np.quantile(w, .975))), (float(np.quantile(d_, .025)), float(np.quantile(d_, .975)))

n_h = None
for city in ['nyc', 'chi']:
    t3p = f'{WS}/data/processed/T3_region_hour/labels.npz' if city == 'nyc' else f'{WS}/data/processed/T3_region_hour_chicago/labels.npz'
    t3 = np.load(t3p); YC = t3['region_hour']; n_h = YC.shape[1]
    starts = [i for i in range(0, n_h - 23, 3) if i >= 24 * 1461]
    days = np.array(starts) // 24
    yc, yi = labels(city, 'c'), labels(city, 'i')
    zero_obs_i = float((yi == 0).mean()); zero_obs_c = float((yc == 0).mean())
    for fam, base in [('zi', 'joint3_zi' if city == 'nyc' else 'joint3_zi_chi'), ('cmp', 'joint3_cmp_cmp' if city == 'nyc' else 'joint3_cmp_chi_cmp'),
                      ('cpl', 'joint3_zi_cpl' if city == 'nyc' else 'joint3_zi_chi_cpl')]:
        dv_obs, dv_sta, masses, tails, alphas, p0fit = [], [], [], [], [], []
        for s in SEEDS:
            arm = base + s
            if arm not in set(jr.arm):
                continue
            d = load(arm, city)
            mu = d['mu_c'].astype(np.float64); pic = np.clip(d['pi_c'].astype(np.float64), 1e-7, 1 - 1e-7)
            r = d['r'].astype(np.float64)
            row = jr[(jr.arm == arm) & (jr.window == 'test2025')].iloc[0]
            ac = float(row['alpha_c'])
            if row['rate_diag']:
                dg = json.loads(row['rate_diag']); dv_obs.append(dg['dyn_vs_obs']); dv_sta.append(dg['dyn_vs_static'])
            n_c = 1 / ac; p_c = n_c / (n_c + np.maximum(mu, 1e-6))
            if fam == 'cmp':
                a_s = float(np.asarray(d['alpha_s']).mean()); alphas.append(a_s)
                n_s = 1 / a_s; p_s = n_s / (n_s + np.clip(r, 1e-6, 50))
                t = p_s ** n_s
                Pi0 = pic + (1 - pic) * (p_c / (1 - (1 - p_c) * t)) ** n_c
                # 截断尾质量 P(y_c>24)
                tail = 1 - nbinom.cdf(24, n_c, p_c)
                tails.append(float(tail.max()))
            else:
                pii = np.clip(d['pi_i'].astype(np.float64), 1e-7, 1 - 1e-7)
                ai = float(row['alpha_i']); n_i = 1 / ai
                mui = r * ((1 - pic) * mu) + 1e-9
                Pi0 = pii + (1 - pii) * (n_i / (n_i + mui)) ** n_i
            Pc0 = pic + (1 - pic) * p_c ** n_c
            masses.append(float(((1 - Pi0) * Pc0).mean()) if fam != 'cmp' else 0.0)
            p0fit.append(float(Pi0.mean()))
        if dv_obs:
            rows.append(dict(city=city, fam=fam, n=len(dv_obs),
                             dyn_vs_obs=round(float(np.mean(dv_obs)), 4), dyn_vs_static=round(float(np.mean(dv_sta)), 4),
                             support_mass=round(float(np.mean(masses)), 5), max_tail_Pyc_gt_24=float(f'{max(tails):.2e}') if tails else np.nan,
                             alpha_s_mean=round(float(np.mean(alphas)), 4) if alphas else np.nan,
                             alpha_s_sd=round(float(np.std(alphas, ddof=1)), 4) if len(alphas) > 1 else np.nan,
                             fitted_Pyi0=round(float(np.mean(p0fit)), 5), observed_zero_inj=round(zero_obs_i, 5)))
    # paired arm-vs-arm (per-point, dual cluster)
    def pts(arm, ch):
        d = load(arm, city); row = jr[(jr.arm == arm) & (jr.window == 'test2025')].iloc[0]
        mu = d['mu_c'].astype(np.float64); pic = np.clip(d['pi_c'].astype(np.float64), 1e-7, 1 - 1e-7)
        r = d['r'].astype(np.float64); ac = float(row['alpha_c']); n_c = 1 / ac; p_c = n_c / (n_c + np.maximum(mu, 1e-6))
        y = yc if ch == 'c' else yi
        if ch == 'c':
            return -np.logaddexp(np.log(pic) + np.where(y == 0, 0.0, -30.0),
                                 np.log1p(-pic) + nbinom.logpmf(y, n_c, p_c))
        if 'cmp' in arm:
            a_s = float(np.asarray(d['alpha_s']).mean()); n_s = 1 / a_s; p_s = n_s / (n_s + np.clip(r, 1e-6, 50))
            acc = np.where(y == 0, pic + (1 - pic) * p_c ** n_c, 0.0)
            for k in range(1, 25):
                acc = acc + (1 - pic) * nbinom.pmf(k, n_c, p_c) * np.exp(nbinom.logpmf(y, k * n_s, p_s))
            return -np.log(np.maximum(acc, 1e-300))
        pii = np.clip(d['pi_i'].astype(np.float64), 1e-7, 1 - 1e-7); ai = float(row['alpha_i']); n_i = 1 / ai
        mui = r * ((1 - pic) * mu) + 1e-9
        return -np.logaddexp(np.log(pii) + np.where(y == 0, 0.0, -30.0),
                             np.log1p(-pii) + nbinom.logpmf(y, n_i, n_i / (n_i + mui)))
    zb = 'joint3_zi' if city == 'nyc' else 'joint3_zi_chi'
    cb = 'joint3_cmp_cmp' if city == 'nyc' else 'joint3_cmp_chi_cb'
    cb = 'joint3_cmp_cmp' if city == 'nyc' else 'joint3_cmp_chi_cmp'
    pb = 'joint3_zi_cpl' if city == 'nyc' else 'joint3_zi_chi_cpl'
    for ch in ['c', 'i']:
        pz = pts(zb, ch)
        for other, tag in [(cb, 'cmp_vs_zi'), (pb, 'cpl_vs_zi')]:
            if other in set(jr.arm):
                po = pts(other, ch)
                dw = (po - pz).reshape(len(starts), -1).mean(1)
                w, dd = boot_pair(dw, days)
                rows.append(dict(city=city, fam=tag + '_' + ch, n=4,
                                 dNLL_pct=round(100 * float(dw.mean()) / float(pz.mean()), 3),
                                 win_ci=f'{w[0]:.6f},{w[1]:.6f}', day_ci=f'{dd[0]:.6f},{dd[1]:.6f}',
                                 sig_better=bool(w[1] < 0 and dd[1] < 0), sig_worse=bool(w[0] > 0 and dd[0] > 0)))
    # joint vs marginal (settf) crash & injury
    for ch, marg in [('c', 'settf_zinb' if city == 'nyc' else 'settf_zinb_chi'),
                     ('i', 'settf_zinb_inj' if city == 'nyc' else 'settf_zinb_chi_inj')]:
        dm_ = np.load(f'{SET}/mu_{marg}_test.npz')
        mu = dm_['mu'].astype(np.float64); pi = np.clip(dm_['pi'].astype(np.float64), 1e-7, 1 - 1e-7)
        sr = pd.read_csv(f'{SET}/settf_results.csv')
        rowm = sr[(sr.arm == marg) & (sr.window == 'test2025')].iloc[0]
        a = float(rowm.model_alpha); n = 1 / a
        y = yc if ch == 'c' else yi
        pm = -np.logaddexp(np.log(pi) + np.where(y == 0, 0.0, -30.0), np.log1p(-pi) + nbinom.logpmf(y, n, n / (n + mu)))
        pz = pts(zb, ch)
        dw = (pz - pm).reshape(len(starts), -1).mean(1)
        w, dd = boot_pair(dw, days)
        rows.append(dict(city=city, fam='zi_vs_marginal_' + ch, n=1,
                         dNLL_pct=round(100 * float(dw.mean()) / float(pm.mean()), 3),
                         win_ci=f'{w[0]:.6f},{w[1]:.6f}', day_ci=f'{dd[0]:.6f},{dd[1]:.6f}',
                         sig_better=bool(w[1] < 0 and dd[1] < 0), sig_worse=bool(w[0] > 0 and dd[0] > 0)))
D = pd.DataFrame(rows)
D.to_csv(f'{OUT}/cmp_fixes.csv', index=False)
pd.set_option('display.width', 250)
print(D.to_string(index=False))
