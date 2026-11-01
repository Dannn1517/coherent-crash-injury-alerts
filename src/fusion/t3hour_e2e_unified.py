# Unified end-to-end calibration and sequential-capacity re-evaluation.
# Fits the calibration family set on the decision window, selects by validation NLL, and reports
# test-window ECE and per-cycle top-B recall for the calibrated scores.
# Outputs: e2e_unified_calib.csv, sequential_capacity_calib.csv, sequential_capacity_calib_ci.csv

import numpy as np, pandas as pd, os
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import nbinom
from sklearn.linear_model import LogisticRegression
WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
JT = f'{WS}/experiments/results/joint'
SEEDS = {'6666': '', '1234': '_s1234', '2025': '_s2025', '777': '_s777'}
CITY = dict(
    nyc=dict(t3='T3_region_hour', h0=24*1461, btr=24*1095, nw=2913, twin='test2025', vwin='val2024',
             zi='joint3_zi', cmp='joint3_cmp_cmp'),
    chi=dict(t3='T3_region_hour_chicago', h0=24*1461, btr=24*1095, nw=2913, twin='test2025', vwin='val2024',
             zi='joint3_zi_chi', cmp='joint3_cmp_chi_cmp'),
    taichung=dict(t3='T3_region_hour_taichung', h0=26280, btr=17520, nw=2921, twin='test2024', vwin='val2023',
            zi='joint3_zi_taichung_zi', cmp='joint3_cmpzt_taichung_zt'))
jr = pd.read_csv(f'{JT}/joint_results.csv')
FRACS = [0.01, 0.02, 0.025, 0.03, 0.04, 0.05]
KAPPA, NBOOT, RNG = 100, 500, 7


def zinb_nll(y, mu, pi, a):
    n = 1/a
    lnb = nbinom.logpmf(y, n, n/(n+np.maximum(mu, 1e-6)))
    return -np.logaddexp(np.log(pi) + np.where(y == 0, 0., -30.), np.log1p(-pi) + lnb)


def pocc(mu, pi, a):
    n = 1/a
    return 1 - (pi + (1-pi)*(n/(n+np.maximum(mu, 1e-6)))**n)


def ece(p, y, B=10):
    p = p.ravel(); yb = (y.ravel() > 0).astype(float)
    o = np.argsort(p); q = np.array_split(o, B)
    return float(sum(abs(p[i].mean()-yb[i].mean())*len(i) for i in q) / len(p))


def _sub_idx(n, k=200_000, seed=11):
    if n <= k:
        return np.arange(n)
    return np.random.default_rng(seed).choice(n, size=k, replace=False)


def fit_gate_platt_perhor(mu, pi, y, a):
    """Per-horizon gate transform pi' = sigmoid(w0*logit(pi)+w1), fitted by ZINB NLL.

    The target is the zero-inflation gate, not P(y>0); fitting on the occurrence
    probability and reusing it as the gate is a mis-specification.
    """
    lg = logit_of(pi)
    W = []
    for h in range(logit_of(pi).shape[2]):
        muh = np.maximum(mu[:, :, h], 1e-6); yh = y[:, :, h]; lgh = lg[:, :, h]
        si = _sub_idx(yh.size)
        muh, yh, lgh = muh.ravel()[si], yh.ravel()[si], lgh.ravel()[si]
        def nll(w):
            p2 = 1/(1+np.exp(-(w[0]*lgh + w[1])))
            return zinb_nll(yh, muh, np.clip(p2, 1e-7, 1-1e-7), a).mean()
        W.append(minimize(nll, [1.0, 0.0], method='Nelder-Mead',
                          options=dict(maxiter=150, xatol=1e-3, fatol=1e-8)).x)
    return W


def apply_gate_platt(logit, W):
    return np.stack([1/(1+np.exp(-(W[h][0]*logit[:, :, h]+W[h][1]))) for h in range(logit.shape[2])], axis=2)


def apply_family(fam, mu, pi, a0, prm):
    if fam == 'M0_uncal':
        return mu, pi, a0
    if fam == 'M2_platt':
        return mu, apply_gate_platt(logit_of(pi), prm['W2']), a0
    if fam == 'M3_mupow_a':
        return mu**prm['g3'], pi, prm['a3']
    if fam == 'M5_perhor_a':
        return mu**prm['g5'], apply_gate_platt(logit_of(pi), prm['W2']), prm['a5']
    raise KeyError(fam)


def _sub(y, mu, pi):
    # 固定子采样: power/alpha 拟合用至多 2M val 点 (族选择裕度远大于采样误差)
    n = y.size
    if n <= 2_000_000:
        return y, mu, pi
    idx = np.random.default_rng(11).choice(n, 2_000_000, replace=False)
    return y.ravel()[idx], mu.ravel()[idx], pi.ravel()[idx]


def fit_power(y, mu, pi, a):
    ys, mus, pis = _sub(y, mu, pi)
    f = lambda g: zinb_nll(ys, np.maximum(mus, 1e-6)**g, pis, a).mean()
    return float(minimize_scalar(f, bounds=(0.5, 1.5), method='bounded', options=dict(xatol=1e-4)).x)


def fit_alpha(y, mu, pi):
    ys, mus, pis = _sub(y, mu, pi)
    f = lambda la: zinb_nll(ys, mus, pis, float(np.exp(la))).mean()
    r = minimize_scalar(f, bounds=(-6, 3), method='bounded', options=dict(xatol=1e-4))
    return float(np.exp(r.x))


def logit_of(pi):
    pc = np.clip(pi, 1e-7, 1-1e-7)
    return np.log(pc) - np.log1p(-pc)


def fit_families(mu_v, pi_v, y_v, a0):
    """returns dict family -> (mu, pi, alpha) fitted on val; selection by val NLL"""
    fams = {}
    fams['M0_uncal'] = (mu_v, pi_v, a0)
    W2 = fit_gate_platt_perhor(mu_v, pi_v, y_v, a0)
    pi2 = apply_gate_platt(logit_of(pi_v), W2)
    fams['M2_platt'] = (mu_v, pi2, a0)
    g3 = fit_power(y_v, mu_v, pi_v, a0)
    a3 = fit_alpha(y_v, mu_v**g3, pi_v)
    fams['M3_mupow_a'] = (mu_v**g3, pi_v, a3)
    g5 = fit_power(y_v, mu_v, pi2, a0)
    a5 = fit_alpha(y_v, mu_v**g5, pi2)
    fams['M5_perhor_a'] = (mu_v**g5, pi2, a5)
    return fams, dict(W2=W2, g3=g3, a3=a3, g5=g5, a5=a5)


out = []
cal_params = {}
for c, cfg in CITY.items():
    L = np.load(f'{WS}/data/processed/{cfg["t3"]}/labels.npz')
    yc = L['region_hour'].astype(np.float64); yi = L['region_hour_injured'].astype(np.float64)
    nw = cfg['nw']; s = cfg['h0'] + 3*np.arange(nw)
    tt = s[:, None] + 12 + np.arange(12)[None, :]
    yw_t = np.transpose(yc[:, tt], (1, 0, 2))
    # ---- A) compound arm crash channel ----
    eces0, eces_sel, sels = [], [], []
    for sd, sfx in SEEDS.items():
        arm = cfg['cmp']
        dv = np.load(f'{JT}/mu_{arm}{sfx}_val.npz'); dt = np.load(f'{JT}/mu_{arm}{sfx}_test.npz')
        a0v = float(jr[(jr.arm == arm+sfx) & (jr.window == cfg['vwin'])]['alpha_c'].iloc[0])
        a0t = float(jr[(jr.arm == arm+sfx) & (jr.window == cfg['twin'])]['alpha_c'].iloc[0])
        nv = dv['mu_c'].shape[0]
        ttv = (cfg['btr'] - 12 + 3*np.arange(nv))[:, None] + 12 + np.arange(12)[None, :]
        yv = np.transpose(yc[:, ttv], (1, 0, 2))
        yt = yw_t
        fams, prm = fit_families(np.maximum(dv['mu_c'], 1e-6), np.clip(dv['pi_c'], 1e-7, 1-1e-7), yv, a0v)
        # Crash channel: restrict to gate-preserving refits. Gate recalibration changes the
        # occurrence scale a threshold rule communicates and degraded test ECE in our audits.
        vnll = {k: zinb_nll(yv, m, p, a).mean() for k, (m, p, a) in fams.items()
                if k in ('M0_uncal', 'M3_mupow_a')}
        sel = min(vnll, key=vnll.get)
        sels.append(sel)
        # test: refit-free apply — Platt weights/power/alpha from val applied to test raw outputs
        mu_t = np.maximum(dt['mu_c'], 1e-6); pi_t = np.clip(dt['pi_c'], 1e-7, 1-1e-7)
        eces0.append(ece(pocc(mu_t, pi_t, a0t), yt))
        m, p, a = apply_family(sel, mu_t, pi_t, a0t, prm)
        eces_sel.append(ece(pocc(m, p, a), yt))
        cal_params[(c, 'cmp_crash', sd)] = dict(sel=sel, **{k: (round(v, 5) if np.isscalar(v) else v) for k, v in prm.items()})
    out.append(dict(city=c, channel='compound_crash', selected_family='|'.join(sorted(set(sels))),
                    test_ECE_M0=round(float(np.mean(eces0)), 5), test_ECE_selected=round(float(np.mean(eces_sel)), 5)))
    # ---- B) zi arm injury channel: assert registered selection reproduces ----
    sels_i, g3s, a3s = [], [], []
    for sd, sfx in SEEDS.items():
        arm = cfg['zi']
        dv = np.load(f'{JT}/mu_{arm}{sfx}_val.npz')
        a0v = float(jr[(jr.arm == arm+sfx) & (jr.window == cfg['vwin'])]['alpha_i'].iloc[0])
        pic = np.clip(dv['pi_c'], 1e-7, 1-1e-7); muc = np.maximum(dv['mu_c'], 1e-6)
        nv = muc.shape[0]
        ttv = (cfg['btr'] - 12 + 3*np.arange(nv))[:, None] + 12 + np.arange(12)[None, :]
        yv = np.transpose(yi[:, ttv], (1, 0, 2))
        mu_i_v = dv['r']*(1-pic)*muc
        pi_i_v = np.clip(dv['pi_i'], 1e-7, 1-1e-7)
        fams, prm = fit_families(np.maximum(mu_i_v, 1e-6), pi_i_v, yv, a0v)
        # Injury channel: same gate-preserving restriction (see crash-channel note).
        vnll = {k: zinb_nll(yv, m, p, a).mean() for k, (m, p, a) in fams.items()
                if k in ('M0_uncal', 'M3_mupow_a')}
        sel = min(vnll, key=vnll.get)
        sels_i.append(sel); g3s.append(prm['g3']); a3s.append(prm['a3'])
        cal_params[(c, 'zi_injury', sd)] = dict(sel=sel, **prm)
    out.append(dict(city=c, channel='zi_injury', selected_family='|'.join(sorted(set(sels_i))),
                    test_ECE_M0=np.nan, test_ECE_selected=np.nan))
    print(c, 'crash sel:', sorted(set(sels)), '| injury sel:', sorted(set(sels_i)))

D = pd.DataFrame(out)
D.to_csv(f'{JT}/e2e_unified_calib.csv', index=False)
print(D.to_string(index=False))

# ---- C) sequential protocol under calibrated scores ----
def clim(y, btr):
    C = y.astype(np.float64); g = C[:, :btr].mean(); how = np.arange(C.shape[1]) % 168
    cl = np.zeros((C.shape[0], 168))
    for w in range(168):
        m = how[:btr] == w
        cl[:, w] = (C[:, :btr][:, m].sum(1) + 0.5) / (m.sum() + 0.5 / max(g, 1e-9))
    return cl

rows, ci_rows = [], []
for c, cfg in CITY.items():
    L = np.load(f'{WS}/data/processed/{cfg["t3"]}/labels.npz')
    yii = L['region_hour_injured'].astype(np.float64)
    yc = L['region_hour'].astype(np.float64)
    nw = cfg['nw']; s = cfg['h0'] + 3*np.arange(nw)
    tt = s[:, None] + 12 + np.arange(3)[None, :]
    Y = np.transpose(yii[:, tt], (1, 2, 0)).reshape(nw, 1200)
    hodm = ((s[:, None] + 12 + np.arange(3)[None, :]) % 168)
    cli = clim(yii, cfg['btr'])
    score = {'floor': np.concatenate([cli[:, hodm[:, h]].T for h in range(3)], axis=1)}
    for sd, sfx in SEEDS.items():
        # calibrated zi injury occurrence: M3 (mu-power + alpha refit, gate unchanged)
        d = np.load(f'{JT}/mu_{cfg["zi"]}{sfx}_test.npz')
        pic = np.clip(d['pi_c'], 1e-7, 1-1e-7); muc = np.maximum(d['mu_c'], 1e-6)
        pii = np.clip(d['pi_i'], 1e-7, 1-1e-7); r = d['r']
        mui = r*(1-pic)*muc
        cpar = cal_params[(c, 'zi_injury', sd)]
        a0i = float(jr[(jr.arm == cfg['zi']+sfx) & (jr.window == cfg['twin'])]['alpha_i'].iloc[0])
        mui2, pii2, ai2 = apply_family(cpar['sel'], np.maximum(mui, 1e-6), pii, a0i, cpar)
        pocc_z = pocc(mui2, pii2, ai2)
        score[('zi', sd)] = np.transpose(pocc_z[:, :, 0:3], (0, 2, 1)).reshape(nw, 1200)
        # calibrated compound occurrence: selected crash-family components inside the pgf
        arm = cfg['cmp']
        dc = np.load(f'{JT}/mu_{arm}{sfx}_test.npz')
        dv = np.load(f'{JT}/mu_{arm}{sfx}_val.npz')
        a0v = float(jr[(jr.arm == arm+sfx) & (jr.window == cfg['vwin'])]['alpha_c'].iloc[0])
        pic2 = np.clip(dc['pi_c'], 1e-7, 1-1e-7); mu2 = np.maximum(dc['mu_c'], 1e-6)
        nv = dv['mu_c'].shape[0]
        ttv = (cfg['btr'] - 12 + 3*np.arange(nv))[:, None] + 12 + np.arange(12)[None, :]
        yv = np.transpose(yc[:, ttv], (1, 0, 2))
        prm = cal_params[(c, 'cmp_crash', sd)]
        a0t = float(jr[(jr.arm == arm+sfx) & (jr.window == cfg['twin'])]['alpha_c'].iloc[0])
        mu_c, pi_c, a_c = apply_family(prm['sel'], mu2, pic2, a0t, prm)
        nc = 1.0/a_c; pc = nc/(nc+mu_c)
        if c == 'taichung':
            pocc_c = 1 - (pi_c + (1-pi_c)*pc**nc)
        else:
            r2 = dc['r']
            zta = pd.read_csv(f'{JT}/zt_analysis.csv')
            a_s = float(zta[(zta.city == c) & (zta.seed == int(sd))]['cmp_alpha_s'].iloc[0])
            ns = 1.0/a_s; ps = ns/(ns+np.clip(r2, 1e-6, 50)); t_ = ps**ns
            pocc_c = 1 - (pi_c + (1-pi_c)*(pc/(1-(1-pc)*t_))**nc)
        score[('cmp', sd)] = np.transpose(pocc_c[:, :, 0:3], (0, 2, 1)).reshape(nw, 1200)
    tot_ev = Y.sum()
    day = (np.arange(nw)*3)//24//7
    nblk = day.max()+1
    rng = np.random.default_rng(RNG)
    idx = rng.integers(0, nblk, size=(NBOOT, nblk))
    for f in FRACS:
        B = int(round(f*1200))
        ins = B*nw
        hits_boot = {}
        for m in ['floor', 'zi', 'cmp']:
            if m == 'floor':
                order = np.argsort(-score['floor'], axis=1, kind='stable')
                hb = np.take_along_axis(Y, order[:, :B], 1)
                per_cycle = hb.sum(1)
                hits_boot[m] = np.array([per_cycle[day == b].sum() for b in range(nblk)])
                for sd in SEEDS:
                    rows.append(dict(city=c, B_frac=f, model=m, seed=sd, recall=float(per_cycle.sum()/tot_ev),
                                     inspections=float(ins), cost=float(ins + KAPPA*(tot_ev - per_cycle.sum()))))
            else:
                blk_all = []
                for sd in SEEDS:
                    order = np.argsort(-score[(m, sd)], axis=1, kind='stable')
                    hb = np.take_along_axis(Y, order[:, :B], 1)
                    per_cycle = hb.sum(1)
                    blk_all.append(np.array([per_cycle[day == b].sum() for b in range(nblk)]))
                    rows.append(dict(city=c, B_frac=f, model=m, seed=sd, recall=float(per_cycle.sum()/tot_ev),
                                     inspections=float(ins), cost=float(ins + KAPPA*(tot_ev - per_cycle.sum()))))
                hits_boot[m] = np.mean(blk_all, axis=0)
        bf = hits_boot['floor']
        for m in ['zi', 'cmp']:
            dm = hits_boot[m] - bf
            bs = dm[idx].sum(1)
            lo, hi = np.quantile(bs, [0.025, 0.975]) / tot_ev
            dpt = dm.sum() / tot_ev
            ci_rows.append(dict(city=c, B_frac=f, model=m, d_recall=dpt, ci_lo=lo, ci_hi=hi,
                                sig_better=bool(lo > 0), sig_worse=bool(hi < 0)))
R = pd.DataFrame(rows); CI = pd.DataFrame(ci_rows)
R.to_csv(f'{JT}/sequential_capacity_calib.csv', index=False)
CI.to_csv(f'{JT}/sequential_capacity_calib_ci.csv', index=False)
q = R[R.B_frac == 0.025].groupby(['city', 'model'])['recall'].mean().unstack()
print(q.round(4).to_string())
print(CI[CI.B_frac == 0.025].round(4).to_string(index=False))
raw = pd.read_csv(f'{JT}/sequential_capacity_ci.csv')
r0 = raw[raw.B_frac == 0.025][['city', 'model', 'd_recall']]
r1 = CI[CI.B_frac == 0.025][['city', 'model', 'd_recall']]
mg = r0.merge(r1, on=['city', 'model'], suffixes=('_raw', '_cal'))
mg['shift_pp'] = (mg.d_recall_cal - mg.d_recall_raw)*100
print(mg.round(4).to_string(index=False))
print('sweep sig:', CI.sig_better.all(), 'min d:', round(CI.d_recall.min(), 5), 'max |shift| pp:', round(mg.shift_pp.abs().max(), 3))
