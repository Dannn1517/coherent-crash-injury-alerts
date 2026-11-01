#  (H7): 臂间配对 bootstrap —— selected vs zero-training hybrid, compound vs free-gate.
# 只用持久化预测 + mvp0 hybrid 数字, CPU; 窗簇 rng7 / 日簇 rng8 与主线同纪律.
import numpy as np, pandas as pd, os
from scipy.stats import nbinom
WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
R = f'{WS}/data/model_inputs'
JT = f'{WS}/experiments/results/joint'
SET = f'{WS}/experiments/results/settf'
OUT = f'{WS}/experiments/results/joint'
IN_LEN = OUT_LEN = 12; STRIDE = 3; B_TR = 24 * 1095; B_VA = 24 * 1461; WARM = 168; N = 400
t3 = np.load(f'{WS}/data/processed/T3_region_hour/labels.npz')
YC = t3['region_hour'].astype(np.float64); YI = t3['region_hour_injured'].astype(np.float64)
n_h = YC.shape[1]; how = np.arange(n_h) % 168
starts = []
for i in range(0, n_h - 23, STRIDE):
    o = i + 24
    if i >= B_VA:
        starts.append(i)
starts = np.array(starts); days = starts // 24; S = len(starts)
tr_h = np.arange(WARM, B_TR)

def clim(C):
    g = C[:, tr_h].mean(); cl = np.zeros((N, 168))
    for w in range(168):
        m = how[tr_h] == w
        cl[:, w] = (C[:, tr_h][:, m].sum(1) + 0.5) / (m.sum() + 0.5 / max(g, 1e-9))
    return cl
def floor_mu(cl):
    tt = starts[:S, None] + IN_LEN + np.arange(OUT_LEN)[None, :]
    return cl[:, how[tt.ravel()]].T.reshape(S, OUT_LEN, N).transpose(0, 2, 1)
CL_I = clim(YI); A_FI = 14.110347
yi = np.transpose(np.load(f'{R}/crash_hour_inj/test.npz')['y'][:, :, :, 0], (0, 2, 1)).astype(np.float64)
mf_i = floor_mu(CL_I)
pts_fi = -nbinom.logpmf(yi, 1 / A_FI, (1 / A_FI) / ((1 / A_FI) + np.maximum(mf_i, 1e-6)))

def nb_pts(y, mu, a):
    n = 1 / a
    return -nbinom.logpmf(y, n, n / (n + np.maximum(mu, 1e-6)))

def zinb_pts(y, mu, pi, a):
    n = 1 / a; pi = np.clip(pi, 1e-7, 1 - 1e-7); mu = np.maximum(mu, 1e-6)
    lnb = nbinom.logpmf(y, n, n / (n + mu))
    return -np.logaddexp(np.log(pi) + np.where(y == 0, 0.0, -30.0), np.log1p(-pi) + lnb)

def boot(d_w):
    Sw = len(d_w)
    win = np.quantile(d_w[np.random.default_rng(7).integers(0, Sw, (500, Sw))].mean(1), [.025, .975])
    ud = np.unique(days); dm = np.array([d_w[days == d].mean() for d in ud])
    day = np.quantile(dm[np.random.default_rng(8).integers(0, len(ud), (500, len(ud)))].mean(1), [.025, .975])
    return win, day

rows = []
# --- selected (joint3_zi) vs zero-training hybrid, injury channel, per seed + pooled ---
# Hybrid NLL from the zero-training hybrid arm (mvp0_results.csv, not shipped); a fixed
# reference constant keeps the shipped pair table reproducible from the package alone.
hyb_nll = 0.05181
for seed, arm in [('6666', 'joint3_zi'), ('1234', 'joint3_zi_s1234'), ('2025', 'joint3_zi_s2025'), ('777', 'joint3_zi_s777')]:
    d = np.load(f'{JT}/mu_{arm}_test.npz')
    mu_c = d['mu_c'].astype(float); pi_c = np.clip(d['pi_c'].astype(float), 1e-7, 1 - 1e-7)
    r = d['r'].astype(float); pi_i = np.clip(d['pi_i'].astype(float), 1e-7, 1 - 1e-7)
    jr = pd.read_csv(f'{JT}/joint_results.csv')
    row = jr[(jr.arm == arm) & (jr.window == 'test2025')].iloc[0]
    ai = float(row['alpha_i'])
    Ec = (1 - pi_c) * mu_c; mu_i = r * Ec + 1e-9
    pts = zinb_pts(yi, mu_i, pi_i, ai)
    d_w = (pts - pts_fi).reshape(S, -1).mean(1)          # vs floor
    w1, d1 = boot(d_w)
    rows.append(dict(pair=f'zi{seed}_vs_floor', dNLL_pct=round(100 * d_w.mean() / pts_fi.mean(), 3),
                     win_ci=f'{w1[0]:.6f},{w1[1]:.6f}', day_ci=f'{d1[0]:.6f},{d1[1]:.6f}'))
    hw = pts.reshape(S, -1).mean(1) - hyb_nll             # hybrid 为单点 NLL: 对 zi 的窗均值做 bootstrap
    w2, d2 = boot(hw)
    rows.append(dict(pair=f'zi{seed}_vs_hybrid', gap_pct=round(100 * hw.mean() / hyb_nll, 3),
                     win_ci=f'{w2[0]:.6f},{w2[1]:.6f}', day_ci=f'{d2[0]:.6f},{d2[1]:.6f}',
                     sig_worse_than_hybrid=bool(w2[0] > 0 and d2[0] > 0),
                     sig_better_than_hybrid=bool(w2[1] < 0 and d2[1] < 0)))
# --- compound vs free-gate, injury channel, per seed (paired on same windows) ---
for seed, a_cmp, a_zi in [('6666', 'joint3_cmp_cmp', 'joint3_zi'), ('1234', 'joint3_cmp_cmp_s1234', 'joint3_zi_s1234'),
                          ('2025', 'joint3_cmp_cmp_s2025', 'joint3_zi_s2025'), ('777', 'joint3_cmp_cmp_s777', 'joint3_zi_s777'),
                          ('6666chi', 'joint3_cmp_chi_cmp', 'joint3_zi_chi'), ('1234chi', 'joint3_cmp_chi_cmp_s1234', 'joint3_zi_chi_s1234'),
                          ('2025chi', 'joint3_cmp_chi_cmp_s2025', 'joint3_zi_chi_s2025'), ('777chi', 'joint3_cmp_chi_cmp_s777', 'joint3_zi_chi_s777')]:
    if 'chi' in seed:
        t3c = np.load(f'{WS}/data/processed/T3_region_hour_chicago/labels.npz')
        YIc = t3c['region_hour_injured'].astype(np.float64)
        yic = np.transpose(np.load(f'{R}/crash_hour_chi_inj/test.npz')['y'][:, :, :, 0], (0, 2, 1)).astype(np.float64)
        CL_Ic = clim(YIc)
        tt = starts[:S, None] + IN_LEN + np.arange(OUT_LEN)[None, :]
        mf_ic = CL_Ic[:, how[tt.ravel()]].T.reshape(S, OUT_LEN, N).transpose(0, 2, 1)
        jr = pd.read_csv(f'{JT}/joint_results.csv')
        ai_c = float(jr[(jr.arm == a_zi) & (jr.window == 'test2025')].alpha_i.iloc[0])
        pts_f = -nbinom.logpmf(yic, 1 / ai_c, (1 / ai_c) / ((1 / ai_c) + np.maximum(mf_ic, 1e-6)))
        y_use, floor_use = yic, pts_f
        dcmp = np.load(f'{JT}/mu_{a_cmp}_test.npz'); dzi = np.load(f'{JT}/mu_{a_zi}_test.npz')
        ai_cmp = ai_c  # compound 边际混合不需要 α_i; 用 zi 的地板 α 仅做地板参照
    else:
        y_use, floor_use = yi, pts_fi
        dcmp = np.load(f'{JT}/mu_{a_cmp}_test.npz'); dzi = np.load(f'{JT}/mu_{a_zi}_test.npz')
        jr = pd.read_csv(f'{JT}/joint_results.csv')
        ai_c = float(jr[(jr.arm == a_zi) & (jr.window == 'test2025')].alpha_i.iloc[0])
    # compound 边际混合
    mu_c = dcmp['mu_c'].astype(float); pic = np.clip(dcmp['pi_c'].astype(float), 1e-7, 1 - 1e-7)
    r = dcmp['r'].astype(float); a_s = float(np.asarray(dcmp['alpha_s']).mean()) if dcmp['alpha_s'].ndim == 0 else None
    jrrow = pd.read_csv(f'{JT}/joint_results.csv')
    ac = float(jrrow[(jrrow.arm == a_cmp) & (jrrow.window == 'test2025')].alpha_c.iloc[0])
    n_c = 1 / ac; p_c = n_c / (n_c + np.maximum(mu_c, 1e-6))
    n_s = 1 / max(a_s if a_s else float(dcmp['alpha_s'].mean()), 1e-8); p_s = n_s / (n_s + np.clip(r, 1e-6, 50))
    acc = np.where(y_use == 0, pic + (1 - pic) * p_c ** n_c, 0.0)
    for k in range(1, 25):
        acc = acc + (1 - pic) * nbinom.pmf(k, n_c, p_c) * np.exp(nbinom.logpmf(y_use, k * n_s, p_s))
    pts_cmp = -np.log(np.maximum(acc, 1e-300))
    # zi 边际
    muzi = dzi['mu_c'].astype(float); pizi = np.clip(dzi['pi_c'].astype(float), 1e-7, 1 - 1e-7)
    rzi = dzi['r'].astype(float); pizi_i = np.clip(dzi['pi_i'].astype(float), 1e-7, 1 - 1e-7)
    Eczi = (1 - pizi) * muzi
    pts_zi = zinb_pts(y_use, rzi * Eczi + 1e-9, pizi_i, ai_c)
    d_w = (pts_cmp - pts_zi).reshape(S, -1).mean(1)
    win, day = boot(d_w)
    rows.append(dict(pair=f'cmp_vs_zi_{seed}', dNLL_pct=round(100 * d_w.mean() / pts_zi.mean(), 3),
                     win_ci=f'{win[0]:.6f},{win[1]:.6f}', day_ci=f'{day[0]:.6f},{day[1]:.6f}',
                     sig_better=bool(win[1] < 0 and day[1] < 0), sig_worse=bool(win[0] > 0 and day[0] > 0)))
D = pd.DataFrame(rows)
D.to_csv(f'{OUT}/paired_tests.csv', index=False)
print(D.to_string(index=False))
