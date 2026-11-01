# internal reviewB2补算 (): 梯度定律6点+CHI边际臂的日簇bootstrap (内审发现settf脚本只算窗簇)
# 臂: settf_zinb_{g10,g10_inj,g40,g40_inj,chi,chi_inj}; 口径=floor_correction同款(EB_mom地板/float64/rng7+8)
# 输出: experiments/results/settf/gradient_bootstrap.csv
import numpy as np, pandas as pd, os
from scipy.stats import nbinom

WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
R = f'{WS}/data/model_inputs'
SET = f'{WS}/experiments/results/settf'
IN_LEN, OUT_LEN, STRIDE = 12, 12, 3
B_TRAIN_END, B_VAL_END, WARM = 24 * 1095, 24 * 1461, 168

ARMS = [('settf_zinb_g10', 'T3_region_hour_g10', 'crash_hour_g10', 'region_hour'),
        ('settf_zinb_g10_inj', 'T3_region_hour_g10', 'crash_hour_g10_inj', 'region_hour_injured'),
        ('settf_zinb_g40', 'T3_region_hour_g40', 'crash_hour_g40', 'region_hour'),
        ('settf_zinb_g40_inj', 'T3_region_hour_g40', 'crash_hour_g40_inj', 'region_hour_injured'),
        ('settf_zinb_chi', 'T3_region_hour_chicago', 'crash_hour_chi', 'region_hour'),
        ('settf_zinb_chi_inj', 'T3_region_hour_chicago', 'crash_hour_chi_inj', 'region_hour_injured')]

rows = []
for arm, t3dir, datadir, key in ARMS:
    t3 = np.load(f'{WS}/data/processed/{t3dir}/labels.npz')
    C = t3[key].astype(np.float64)
    n_reg, n_hours = C.shape
    how = np.arange(n_hours) % 168
    starts = {'train': [], 'val': [], 'test': []}
    for i in range(0, n_hours - 23, STRIDE):
        o = i + 24
        if o <= B_TRAIN_END: starts['train'].append(i)
        elif i >= B_TRAIN_END and o <= B_VAL_END: starts['val'].append(i)
        elif i >= B_VAL_END: starts['test'].append(i)
    starts = {k: np.array(v) for k, v in starts.items()}
    tr_h = np.arange(WARM, B_TRAIN_END)
    g = C[:, tr_h].mean()
    clim = np.zeros((n_reg, 168))
    for w in range(168):
        m = how[tr_h] == w
        clim[:, w] = (C[:, tr_h][:, m].sum(1) + 0.5) / (m.sum() + 0.5 / max(g, 1e-9))
    y_tr = np.transpose(np.load(f'{R}/{datadir}/train.npz')['y'][:, :, :, 0], (0, 2, 1)).astype(np.float64)
    tt_tr = starts['train'][:len(y_tr), None] + IN_LEN + np.arange(OUT_LEN)[None, :]
    mf_tr = clim[:, how[tt_tr.ravel()]].T.reshape(len(y_tr), OUT_LEN, n_reg).transpose(0, 2, 1)
    a_f = float(max((((y_tr - mf_tr) ** 2 - mf_tr).mean()) / max((mf_tr ** 2).mean(), 1e-12), 1e-4))
    csv = pd.read_csv(f'{SET}/settf_results.csv')
    a_m = float(csv[csv.arm == arm]['model_alpha'].iloc[0])
    for sp in ['val', 'test']:
        S = len(starts[sp])
        y = np.transpose(np.load(f'{R}/{datadir}/{sp}.npz')['y'][:, :, :, 0], (0, 2, 1)).astype(np.float64)[:S]
        d = np.load(f'{SET}/mu_{arm}_{sp}.npz')
        mu = d['mu'][:S].astype(np.float64)
        pi = d['pi'][:S].astype(np.float64) if 'pi' in d.files else None
        n = 1.0 / max(a_m, 1e-8)
        mun = np.maximum(mu, 1e-6)
        lnb = nbinom.logpmf(y, n, n / (n + mun))
        if pi is not None:
            pic = np.clip(pi, 1e-7, 1 - 1e-7)
            pts_m = -np.logaddexp(np.log(pic) + np.where(y == 0, 0.0, -30.0), np.log1p(-pic) + lnb)
        else:
            pts_m = -lnb
        tt = starts[sp][:S, None] + IN_LEN + np.arange(OUT_LEN)[None, :]
        mf = clim[:, how[tt.ravel()]].T.reshape(S, OUT_LEN, n_reg).transpose(0, 2, 1)
        nf = 1.0 / max(a_f, 1e-8)
        pts_f = -nbinom.logpmf(y, nf, nf / (nf + np.maximum(mf, 1e-6)))
        d_w = (pts_m - pts_f).reshape(S, -1).mean(1)
        days = starts[sp][:S] // 24
        rb = np.random.default_rng(7)
        bsw = d_w[rb.integers(0, S, size=(500, S))].mean(1)
        ud = np.unique(days)
        dm = np.array([d_w[days == dd].mean() for dd in ud])
        bsd = dm[np.random.default_rng(8).integers(0, len(ud), size=(500, len(ud)))].mean(1)
        rows.append(dict(arm=arm, window=f'{sp}2024' if sp == 'val' else f'{sp}2025',
                         dNLL_pct=round(100 * float(d_w.mean()) / float(pts_f.mean()), 3),
                         win_ci=f'{np.quantile(bsw, .025):.6f},{np.quantile(bsw, .975):.6f}',
                         win_sig=bool(np.quantile(bsw, .975) < 0),
                         day_ci=f'{np.quantile(bsd, .025):.6f},{np.quantile(bsd, .975):.6f}',
                         day_sig=bool(np.quantile(bsd, .975) < 0),
                         day_sig_worse=bool(np.quantile(bsd, .025) > 0)))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f'{SET}/gradient_bootstrap.csv', index=False)
print('DONE ->', f'{SET}/gradient_bootstrap.csv')
