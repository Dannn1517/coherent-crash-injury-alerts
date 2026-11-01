# Decompose the price of exact coherence by conditional family.
# Compares free-gate, compound NB, and compound ZTNB injury heads on locked test windows using the
# same EB floor and evaluation code as the joint model.
# Output: experiments/results/joint/zt_analysis.csv

import numpy as np, pandas as pd, os
from scipy.stats import nbinom
WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
R = f'{WS}/data/model_inputs'; JT = f'{WS}/experiments/results/joint'
IN_LEN, OUT_LEN, STRIDE, WARM = 12, 12, 3, 168
SEEDS = {'6666': '', '1234': '_s1234', '2025': '_s2025', '777': '_s777'}
CITY = {
    'nyc': dict(t3='T3_region_hour', di='crash_hour_inj', dc='crash_hour',
                btr=24*1095, bva=24*1461, test_win='test2025',
                arms=dict(zi='joint3_zi', cmp='joint3_cmp_cmp', zt='joint3_cmpzt_zt')),
    'chi': dict(t3='T3_region_hour_chicago', di='crash_hour_chi_inj', dc='crash_hour_chi',
                btr=24*1095, bva=24*1461, test_win='test2025',
                arms=dict(zi='joint3_zi_chi', cmp='joint3_cmp_chi_cmp', zt='joint3_cmpzt_chi_zt')),
    'taichung':  dict(t3='T3_region_hour_taichung', di='crash_hour_taichung_inj', dc='crash_hour_taichung',
                btr=24*730, bva=24*1095, test_win='test2024',
                arms=dict(zi='joint3_zi_taichung_zi', cmp='joint3_cmp_taichung_cmp', zt='joint3_cmpzt_taichung_zt')),
}
jr = pd.read_csv(f'{JT}/joint_results.csv')
rows = []

def clim_eb(C, tr_h):
    g = C[:, tr_h].mean(); cl = np.zeros((C.shape[0], 168))
    how = np.arange(C.shape[1]) % 168
    for w in range(168):
        m = how[tr_h] == w
        cl[:, w] = (C[:, tr_h][:, m].sum(1) + 0.5) / (m.sum() + 0.5 / max(g, 1e-9))
    return cl

def fit_alpha(y, mu):
    mu = np.clip(mu.astype(np.float64), 1e-6, None); y = y.astype(np.float64)
    return float(max((((y - mu)**2 - mu).mean()) / max((mu**2).mean(), 1e-12), 1e-4))

def boot2(d_w, days):
    S = len(d_w); rb = np.random.default_rng(7)
    bs = d_w[rb.integers(0, S, size=(500, S))].mean(1)
    win = (float(np.quantile(bs, .025)), float(np.quantile(bs, .975)))
    ud = np.unique(days)
    dmean = np.array([d_w[days == d].mean() for d in ud])
    bs2 = dmean[np.random.default_rng(8).integers(0, len(ud), size=(500, len(ud)))].mean(1)
    day = (float(np.quantile(bs2, .025)), float(np.quantile(bs2, .975)))
    return win, day

def zt_mixture(y, pic, mu_c, r, a_s, ac):
    """ZTNB compound 边际 (与 eval_arm cmpzt 分支同式): P(0)=P(y_c=0); y≥1 加 ZT 归一"""
    n_c = 1.0/ac; p_c = n_c/(n_c + np.maximum(mu_c, 1e-6))
    n_s = 1.0/max(a_s, 1e-6); p_s = n_s/(n_s + np.clip(r, 1e-6, 50.0))
    Pc0 = pic + (1-pic)*p_c**n_c
    lsp = np.log(np.clip(p_s, 1e-7, 1-1e-7))
    pos = y > 0
    acc = np.where(pos, 0.0, Pc0)
    for k in range(1, 25):
        logt = np.minimum(k*n_s*lsp, -1e-7)
        log1mt = np.where(logt > -0.6931, np.log(-np.expm1(logt)), np.log1p(-np.exp(logt)))
        term = (1-pic)*nbinom.pmf(k, n_c, p_c)*np.exp(nbinom.logpmf(y, k*n_s, p_s) - log1mt)
        acc = acc + np.where(pos, term, 0.0)
    return -np.log(np.maximum(acc, 1e-300)), Pc0

def cmp_mixture(y, pic, mu_c, r, a_s, ac):
    n_c = 1.0/ac; p_c = n_c/(n_c + np.maximum(mu_c, 1e-6))
    n_s = 1.0/max(a_s, 1e-6); p_s = n_s/(n_s + np.clip(r, 1e-6, 50.0))
    Pc0 = pic + (1-pic)*p_c**n_c
    acc = np.where(y == 0, Pc0, 0.0)
    for k in range(1, 25):
        acc = acc + (1-pic)*nbinom.pmf(k, n_c, p_c)*np.exp(nbinom.logpmf(y, k*n_s, p_s))
    G = Pc0 + (1-pic)*sum(nbinom.pmf(k, n_c, p_c)*(p_s**n_s)**k for k in range(1, 25))
    return -np.log(np.maximum(acc, 1e-300)), G

for city, cfg in CITY.items():
    t3 = np.load(f'{WS}/data/processed/{cfg["t3"]}/labels.npz')
    YI_L = t3['region_hour_injured'].astype(np.float64); YC_L = t3['region_hour'].astype(np.float64)
    n_reg, n_hours = YI_L.shape
    how = np.arange(n_hours) % 168
    starts = {'train': [], 'test': []}
    for i in range(0, n_hours-23, STRIDE):
        o = i+24
        if o <= cfg['btr']: starts['train'].append(i)
        elif i >= cfg['bva']: starts['test'].append(i)
    starts = {k: np.array(v) for k, v in starts.items()}
    tr_h = np.arange(WARM, cfg['btr'])
    CLIM_I = clim_eb(YI_L, tr_h)
    S = len(starts['test'])
    tt = starts['test'][:S, None] + IN_LEN + np.arange(OUT_LEN)[None, :]
    mf = CLIM_I[:, how[tt.ravel()]].T.reshape(S, OUT_LEN, n_reg).transpose(0, 2, 1)
    ytr = np.transpose(np.load(f'{R}/{cfg["di"]}/train.npz')['y'][:, :, :, 0], (0, 2, 1))
    tt_tr = starts['train'][:len(ytr), None] + IN_LEN + np.arange(OUT_LEN)[None, :]
    mf_tr = CLIM_I[:, how[tt_tr.ravel()]].T.reshape(len(ytr), OUT_LEN, n_reg).transpose(0, 2, 1)
    A_F = fit_alpha(ytr, mf_tr)
    yi = np.transpose(np.load(f'{R}/{cfg["di"]}/test.npz')['y'][:, :, :, 0], (0, 2, 1)).astype(np.float64)[:S]
    yc = np.transpose(np.load(f'{R}/{cfg["dc"]}/test.npz')['y'][:, :, :, 0], (0, 2, 1)).astype(np.float64)[:S]
    n = 1.0/A_F
    pts_f = -nbinom.logpmf(yi, n, n/(n + np.maximum(mf, 1e-6)))
    days = starts['test'][:S]//24
    zero_obs = float((yi == 0).mean())
    p0_given_crash = float(((yi == 0) & (yc > 0)).sum() / max((yc > 0).sum(), 1))

    per_seed = {}
    for sd, sfx in SEEDS.items():
        arm = {f: cfg['arms'][f]+sfx for f in ['zi', 'cmp', 'zt']}
        if not all(os.path.exists(f'{JT}/mu_{arm[f]}_test.npz') for f in ['zi', 'cmp', 'zt']):
            continue
        if not all(len(jr[(jr.arm == arm[f]) & (jr.window == cfg['test_win'])]) for f in ['zi', 'cmp', 'zt']):
            continue
        d = {f: np.load(f'{JT}/mu_{arm[f]}_test.npz') for f in ['zi', 'cmp', 'zt']}
        a = {f: jr[(jr.arm == arm[f]) & (jr.window == cfg['test_win'])].iloc[0] for f in ['zi', 'cmp', 'zt']}
        pts = {}; p0fit = {}; alpha_s = {}
        # zi: ZINB 边际
        pic_z = np.clip(d['zi']['pi_c'].astype(float), 1e-7, 1-1e-7)
        muz = np.maximum(d['zi']['mu_c'].astype(float), 1e-6); rz = d['zi']['r'].astype(float)
        pii = np.clip(d['zi']['pi_i'].astype(float), 1e-7, 1-1e-7)
        mui = rz*(1-pic_z)*muz + 1e-9; ai = float(a['zi']['alpha_i']); ni = 1/ai
        lnb = nbinom.logpmf(yi, ni, ni/(ni + np.maximum(mui, 1e-6)))
        pts['zi'] = -np.logaddexp(np.log(pii) + np.where(yi == 0, 0.0, -30.0), np.log1p(-pii) + lnb)
        p0fit['zi'] = pii + (1-pii)*(ni/(ni + np.maximum(mui, 1e-6)))**ni
        # cmp / zt
        for f, fn in [('cmp', cmp_mixture), ('zt', zt_mixture)]:
            pic = np.clip(d[f]['pi_c'].astype(float), 1e-7, 1-1e-7)
            mu_ = np.maximum(d[f]['mu_c'].astype(float), 1e-6); r_ = d[f]['r'].astype(float)
            a_s = float(np.asarray(d[f]['alpha_s']).mean())
            pts[f], p0fit[f] = fn(yi, pic, mu_, r_, a_s, float(a[f]['alpha_c']))
            alpha_s[f] = a_s
        rec = dict(city=city, seed=sd, zero_obs=round(zero_obs, 6), p0_given_crash=round(p0_given_crash, 6))
        for f in ['zi', 'cmp', 'zt']:
            rec[f'{f}_dNLL'] = round(100*float((pts[f]-pts_f).mean())/float(pts_f.mean()), 3)
            rec[f'{f}_overshoot_pp'] = round(100*float(p0fit[f].mean() - zero_obs), 3)
        rec['cmp_alpha_s'] = round(alpha_s['cmp'], 4); rec['zt_alpha_s'] = round(alpha_s['zt'], 4)
        for tag, (pa, pb) in {'zt_vs_zi': ('zt', 'zi'), 'zt_vs_cmp': ('zt', 'cmp')}.items():
            d_w = (pts[pa]-pts[pb]).reshape(S, -1).mean(1)
            win, day = boot2(d_w, days)
            rec[tag] = round(100*float(d_w.mean())/float(pts[pb].mean()), 3)
            rec[f'{tag}_win_ci'] = f'{win[0]:.6f},{win[1]:.6f}'
            rec[f'{tag}_sig_better'] = bool(win[1] < 0 and day[1] < 0)
            rec[f'{tag}_sig_worse'] = bool(win[0] > 0 and day[0] > 0)
        per_seed[sd] = rec
        rows.append(rec)
        print(f'[{city}/{sd}] ' + ' '.join(f'{f}={rec[f"{f}_dNLL"]:+.3f}%' for f in ['zi', 'cmp', 'zt'])
              + f' | zt-zi={rec["zt_vs_zi"]:+.3f}% ({"sig" if rec["zt_vs_zi_sig_better"] or rec["zt_vs_zi_sig_worse"] else "ns"})'
              + f' overshoot zi/cmp/zt={rec["zi_overshoot_pp"]}/{rec["cmp_overshoot_pp"]}/{rec["zt_overshoot_pp"]}pp'
              + f' P(y_i=0|y_c>0)={rec["p0_given_crash"]:.4f}', flush=True)
    if per_seed:
        m = pd.DataFrame(per_seed.values())
        print(f'[{city}] {len(m)}-seed mean±SD: ' + ' '.join(f'{f}={m[f"{f}_dNLL"].mean():+.3f}±{m[f"{f}_dNLL"].std():.3f}' for f in ['zi', 'cmp', 'zt'])
              + f' | zt−zi={m.zt_vs_zi.mean():+.3f}±{m.zt_vs_zi.std():.3f} zt−cmp={m.zt_vs_cmp.mean():+.3f}±{m.zt_vs_cmp.std():.3f}', flush=True)

D = pd.DataFrame(rows)
D.to_csv(f'{JT}/zt_analysis.csv', index=False)
print('SAVED', f'{JT}/zt_analysis.csv', len(D), 'rows')
