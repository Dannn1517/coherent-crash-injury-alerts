# Dispatch simulation on deduplicated real cell-hours.
# Each real (cell, hour) is scored once at its shortest lead (h in {0,1,2}), matching a causal
# cycle where the operator commits patrols from the latest issued forecast. Cost is inspections
# plus kappa times missed events; budgets can be fixed per cycle or matched to decision-window recall.
# Output: experiments/results/joint/dispatch_sim.csv

import numpy as np, pandas as pd, os, json, sys
WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
R = f'{WS}/data/model_inputs'; JT = f'{WS}/experiments/results/joint'
IN_LEN, OUT_LEN, STRIDE, WARM = 12, 12, 3, 168
SEEDS = ['', '_s1234', '_s2025', '_s777']
KAPPAS = [1, 10, 50, 100, 500]
EFFS = [0.25, 0.5, 0.75, 1.0]
BUDGET_FRACS = [0.005, 0.01, 0.02, 0.025, 0.03, 0.05, 0.08]
RECALL_TARGETS = [0.10, 0.15, 0.20, 0.25, 0.30]
TAUS = [0.02, 0.05, 0.10, 0.13, 0.20, 0.30]

CITY = {
    'nyc': dict(t3='T3_region_hour', dc='crash_hour', di='crash_hour_inj',
                btr=24*1095, bva=24*1461, test_win='test2025', val_win='val2024',
                zi='joint3_zi', cmp='joint3_cmp_cmp', zt='joint3_cmpzt_zt'),
    'chi': dict(t3='T3_region_hour_chicago', dc='crash_hour_chi', di='crash_hour_chi_inj',
                btr=24*1095, bva=24*1461, test_win='test2025', val_win='val2024',
                zi='joint3_zi_chi', cmp='joint3_cmp_chi_cmp', zt='joint3_cmpzt_chi_zt'),
    'taichung':  dict(t3='T3_region_hour_taichung', dc='crash_hour_taichung', di='crash_hour_taichung_inj',
                btr=24*730, bva=24*1095, test_win='test2024', val_win='val2023',
                zi='joint3_zi_taichung_zi', cmp='joint3_cmp_taichung_cmp', zt='joint3_cmpzt_taichung_zt'),
}
jr = pd.read_csv(f'{JT}/joint_results.csv')
rows = []

def platz_fit(p, y):
    # 逐时距 Platt: logit(p_cal) = a*logit(p) + b, 12×2 参数; z/eta 截断+ridge 防完全分离发散
    pc = np.clip(p, 1e-7, 1-1e-7); z = np.clip(np.log(pc/(1-pc)), -20, 20)
    out = []
    for h in range(p.shape[2]):
        zh, yh = z[:, :, h].ravel(), y[:, :, h].ravel().astype(float)
        X = np.stack([zh, np.ones_like(zh)], 1); w = np.array([1.0, 0.0])
        for _ in range(60):
            eta = np.clip(X @ w, -30, 30); mu = 1/(1+np.exp(-eta)); Wd = np.maximum(mu*(1-mu), 1e-10)
            g = X.T @ (yh-mu) - 1e-2*w; H = X.T @ (X*Wd[:, None]) + 1e-2*np.eye(2)
            step = np.linalg.solve(H, g); w = w + np.clip(step, -5, 5)
            if np.abs(step).max() < 1e-8: break
        out.append(w)
    return np.stack(out)  # (12,2)

def platz_apply(p, w):
    # w: (12,2) 逐时距 [slope, intercept]; p: (S,N,12) → 广播 (1,1,12)
    pc = np.clip(p, 1e-7, 1-1e-7); z = np.clip(np.log(pc/(1-pc)), -20, 20)
    eta = np.clip(z*w[:, 0][None, None, :] + w[:, 1][None, None, :], -40, 40)
    return 1/(1+np.exp(-eta))

def scores_joint(d, alpha_c, alpha_i, ch, fam):
    mu = d['mu_c'].astype(np.float64); pic = np.clip(d['pi_c'].astype(np.float64), 1e-7, 1-1e-7)
    n_c = 1/alpha_c; p_c = n_c/(n_c+np.maximum(mu, 1e-6))
    Pc0 = pic + (1-pic)*p_c**n_c
    if ch == 'c':
        return 1-Pc0
    r = d['r'].astype(np.float64)
    if fam == 'cmp':
        a_s = float(np.asarray(d['alpha_s']).mean())
        n_s = 1/a_s; p_s = n_s/(n_s+np.clip(r, 1e-6, 50)); t = p_s**n_s
        Pi0 = pic + (1-pic)*(p_c/(1-(1-p_c)*t))**n_c
    elif fam == 'zt':
        Pi0 = Pc0          # ZTNB: P(y_i=0)=P(y_c=0) 精确 → occurrence 排序=crash 排序 (结构推论)
    else:
        pii = np.clip(d['pi_i'].astype(np.float64), 1e-7, 1-1e-7)
        n_i = 1/alpha_i; mui = r*(1-pic)*mu + 1e-9
        Pi0 = pii + (1-pii)*(n_i/(n_i+mui))**n_i
    return 1-Pi0

def topb_stats(score, y, budget_fracs, fixed=None):
    # 容量型: 一次全排序, 前缀累计 → 所有预算点; fixed: {key: B} 绝对预算 (val 定, test 兑现)
    flat_s = score.ravel(); flat_y = (y.ravel() > 0).astype(np.int64)
    N = flat_s.size; P = int(flat_y.sum())
    order = np.argsort(-flat_s, kind='stable')
    cum = np.cumsum(flat_y[order])
    out = {}
    def at(B):
        B = int(min(max(B, 1), N)); h = int(cum[B-1])
        return dict(alerts=B, hits=h, recall=h/max(P, 1), precision=h/B, missed=P-h)
    for b in budget_fracs:
        out[str(b)] = at(max(int(round(b*N)), 1))
    for rho in RECALL_TARGETS:
        out[f'rho{rho}_posthoc'] = at(min_budget(cum, P, rho, N))
    for k, B in (fixed or {}).items():
        out[k] = at(B)
    return out, cum, N, P

def min_budget(cum, P, rho, N):
    return int(min(np.searchsorted(cum, max(int(np.ceil(rho*P)), 1)) + 1, N))

def emit(city, model, ch, seed, space, policy, st, effs=(1.0,)):
    for k, v in st.items():
        for e in effs:
            for kap in KAPPAS:
                rows.append(dict(city=city, model=model, ch=ch, seed=seed, space=space, policy=policy,
                                 key=str(k), kappa=kap, eff=e, **{kk: vv for kk, vv in v.items()},
                                 cost=v['alerts']+kap*(v['missed']+(1-e)*v['hits'])))

for city, cfg in CITY.items():
    if not os.path.exists(f'{JT}/mu_{cfg["zi"]}_test.npz'):
        print(f'[{city}] 预测未就绪 (stage35 未完?), 跳过'); continue
    t3 = np.load(f'{WS}/data/processed/{cfg["t3"]}/labels.npz')
    YC_L = t3['region_hour'].astype(np.float64); YI_L = t3['region_hour_injured'].astype(np.float64)
    n_reg, n_hours = YC_L.shape
    how = np.arange(n_hours) % 168
    starts = {'train': [], 'val': [], 'test': []}
    for i in range(0, n_hours-23, STRIDE):
        o = i+24
        if o <= cfg['btr']: starts['train'].append(i)
        elif i >= cfg['btr'] and o <= cfg['bva']: starts['val'].append(i)
        elif i >= cfg['bva']: starts['test'].append(i)
    starts = {k: np.array(v) for k, v in starts.items()}
    tr_h = np.arange(WARM, cfg['btr'])

    def clim_eb(C):
        g = C[:, tr_h].mean(); cl = np.zeros((n_reg, 168))
        for w in range(168):
            m = how[tr_h] == w
            cl[:, w] = (C[:, tr_h][:, m].sum(1)+0.5)/(m.sum()+0.5/max(g, 1e-9))
        return cl
    def floor_mu(cl, sp, S):
        tt = starts[sp][:S, None]+IN_LEN+np.arange(OUT_LEN)[None, :]
        return cl[:, how[tt.ravel()]].T.reshape(S, OUT_LEN, n_reg).transpose(0, 2, 1)
    def fit_alpha(y, mu):
        mu = np.clip(mu.astype(np.float64), 1e-6, None); y = y.astype(np.float64)
        return float(max((((y-mu)**2-mu).mean())/max((mu**2).mean(), 1e-12), 1e-4))
    CLIM_C, CLIM_I = clim_eb(YC_L), clim_eb(YI_L)
    ytr_c = np.transpose(np.load(f'{R}/{cfg["dc"]}/train.npz')['y'][:, :, :, 0], (0, 2, 1))
    ytr_i = np.transpose(np.load(f'{R}/{cfg["di"]}/train.npz')['y'][:, :, :, 0], (0, 2, 1))
    A_FC = fit_alpha(ytr_c, floor_mu(CLIM_C, 'train', len(ytr_c)))
    A_FI = fit_alpha(ytr_i, floor_mu(CLIM_I, 'train', len(ytr_i)))

    lab = {sp: {ch: np.transpose(np.load(f'{R}/{cfg["dc" if ch=="c" else "di"]}/{sp}.npz')['y'][:, :, :, 0], (0, 2, 1)).astype(np.float64)
                for ch in ['c', 'i']} for sp in ['val', 'test']}
    S_te = len(starts['test']); S_va = len(starts['val'])
    def nb_pocc(mu, A):   # NB(n=1/α, p=n/(n+μ)) 的 1−P(0), 与 harness pocc 口径一致
        n = 1.0 / A
        return 1.0 - (n / (n + np.maximum(mu, 1e-6))) ** n
    fl_pocc = {sp: {ch: nb_pocc(floor_mu(CLIM_C if ch == 'c' else CLIM_I, sp, S_te if sp == 'test' else S_va),
                                A_FC if ch == 'c' else A_FI) for ch in ['c', 'i']} for sp in ['val', 'test']}

    for ch in ['i', 'c']:
        y_te_all = lab['test'][ch]; y_va_all = lab['val'][ch][:S_va]
        y_te = y_te_all[:, :, 0:3]; y_va = y_va_all[:, :, 0:3]      # real 空间: h∈{0,1,2} 平铺
        # ---- 地板 ----
        for space, (sc_te, sc_va, y_t, y_v) in {'real': (fl_pocc['test'][ch][:, :, 0:3], fl_pocc['val'][ch][:, :, 0:3], y_te, y_va),
                                                'pred': (fl_pocc['test'][ch], fl_pocc['val'][ch], y_te_all, y_va_all)}.items():
            stv, cumv, Nva, Pva = topb_stats(sc_va, y_v, [])
            fixed = {f'valrho{rho}': min_budget(cumv, Pva, rho, Nva) for rho in RECALL_TARGETS} if space == 'real' else None
            st, *_ = topb_stats(sc_te, y_t, BUDGET_FRACS, fixed=fixed)
            emit(city, 'floor', ch, '-', space, 'capacity', st, effs=EFFS if space == 'real' else (1.0,))
        # ---- 模型臂 (逐 seed) ----
        fams = [('zi', cfg['zi']), ('cmp', cfg['cmp'])] + ([('zt', cfg['zt'])] if cfg.get('zt') else [])
        for fam, base in fams:
            per_seed = {}
            for sfx in SEEDS:
                arm = base+sfx
                if arm not in set(jr.arm): continue
                fpth = f'{JT}/mu_{arm}_test.npz'; vpth = f'{JT}/mu_{arm}_val.npz'
                if not (os.path.exists(fpth) and os.path.exists(vpth)): continue
                row = jr[(jr.arm == arm) & (jr.window == cfg['test_win'])]
                if len(row) == 0: continue
                row = row.iloc[0]
                d = np.load(fpth); dv = np.load(vpth)
                sc = scores_joint(d, float(row['alpha_c']), float(row['alpha_i']), ch, fam)
                sc_v = scores_joint(dv, float(row['alpha_c']), float(row['alpha_i']), ch, fam)
                seed_lbl = sfx or 's6666'
                # real 空间: 容量分数 + val 定预算
                stv, cumv, Nva, Pva = topb_stats(sc_v[:, :, 0:3], y_va, [])
                fixed = {f'valrho{rho}': min_budget(cumv, Pva, rho, Nva) for rho in RECALL_TARGETS}
                st, *_ = topb_stats(sc[:, :, 0:3], y_te, BUDGET_FRACS, fixed=fixed)
                emit(city, fam, ch, seed_lbl, 'real', 'capacity', st, effs=EFFS)
                per_seed[sfx] = st
                # pred 空间诊断 (v1 复现)
                stp, *_ = topb_stats(sc, y_te_all, BUDGET_FRACS)
                emit(city, fam, ch, seed_lbl, 'pred', 'capacity', stp)
                # ---- 阈值型: 原始 vs 校准 (val 全窗拟合 Platt, test 判决) ----
                w = platz_fit(sc_v, (y_va_all > 0).astype(np.float64))
                for tag, s_te in [('raw', sc), ('cal', platz_apply(sc, w))]:
                    for space, (s_use, y_use) in {'real': (s_te[:, :, 0:3], y_te), 'pred': (s_te, y_te_all)}.items():
                        for tau in TAUS:
                            m = s_use >= tau
                            al = int(m.sum()); hits = int((y_use[m] > 0).sum()); P = int((y_use > 0).sum())
                            for kap in KAPPAS:
                                rows.append(dict(city=city, model=fam, ch=ch, seed=seed_lbl, space=space,
                                                 policy=f'thresh_{tag}', key=f'tau{tau}', kappa=kap, eff=1.0,
                                                 alerts=al, hits=hits, recall=hits/max(P, 1),
                                                 precision=hits/max(al, 1), missed=P-hits,
                                                 cost=al+kap*(P-hits)))
            if per_seed:
                agg = {k: {m: float(np.mean([ps[k][m] for ps in per_seed.values()])) for m in ['alerts', 'hits', 'recall', 'precision', 'missed']}
                       for k in list(per_seed.values())[0]}
                print(f'[{city}/{ch}/{fam}] n_seeds={len(per_seed)}', json.dumps({k: {m: round(v, 4) for m, v in d.items()} for k, d in list(agg.items())[:3]}))
    print(f'[{city}] done', flush=True)

df = pd.DataFrame(rows)
df.to_csv(f'{JT}/dispatch_sim.csv', index=False)
print('SAVED', f'{JT}/dispatch_sim.csv', len(df), 'rows')

# ---- summary (real 空间 = 论文口径; injury 通道) ----
sub = df[(df.space == 'real') & (df.ch == 'i') & (df.policy == 'capacity') & (df.eff == 1.0) & (df.kappa == 100)]
for city in df.city.unique():
    s = sub[sub.city == city]
    fl = s[s.model == 'floor'].set_index('key')
    for fam in ['zi', 'cmp', 'zt']:
        mo = s[s.model == fam].groupby('key').mean(numeric_only=True)
        if not len(mo): continue
        if '0.025' in fl.index and '0.025' in mo.index:
            r_fl, r_mo = fl.loc['0.025', 'recall'], mo.loc['0.025', 'recall']
            c_fl, c_mo = fl.loc['0.025', 'cost'], mo.loc['0.025', 'cost']
            m_fl, m_mo = fl.loc['0.025', 'missed'], mo.loc['0.025', 'missed']
            print(f'[{city}/{fam}] REAL matched-budget(2.5%): recall floor={r_fl:.4f} model={r_mo:.4f} | missed {m_fl:.0f}->{m_mo:.0f} | cost(κ=100) {c_fl:.0f}->{c_mo:.0f} ({100*(c_mo-c_fl)/c_fl:+.1f}%)')
        k2 = 'valrho0.2'
        if k2 in fl.index and k2 in mo.index:
            a_fl, a_mo = fl.loc[k2, 'alerts'], mo.loc[k2, 'alerts']
            rr_fl, rr_mo = fl.loc[k2, 'recall'], mo.loc[k2, 'recall']
            print(f'[{city}/{fam}] REAL val-fixed(ρ=0.2): alerts floor={a_fl:.0f} model={a_mo:.0f} ({100*(a_mo-a_fl)/a_fl:+.1f}%) | test recall floor={rr_fl:.4f} model={rr_mo:.4f} (transfer gap {rr_mo-0.2:+.4f}/{rr_fl-0.2:+.4f})')
        k2p = 'rho0.2_posthoc'
        if k2p in fl.index and k2p in mo.index:
            print(f'[{city}/{fam}] PRED-space-style posthoc ρ=0.2 (diagnostic, real space): alerts floor={fl.loc[k2p,"alerts"]:.0f} model={mo.loc[k2p,"alerts"]:.0f} ({100*(mo.loc[k2p,"alerts"]-fl.loc[k2p,"alerts"])/fl.loc[k2p,"alerts"]:+.1f}%)')
# ---- P1-4 有效率敏感性 (real/valrho0.2, κ=100): cost(e) = alerts + κ(P − e·hits) ----
sens = df[(df.space == 'real') & (df.ch == 'i') & (df.policy == 'capacity') & (df.key == 'valrho0.2') & (df.kappa == 100)]
for city in sens.city.unique():
    s = sens[sens.city == city]
    flc = s[s.model == 'floor'].groupby('eff')['cost'].mean()
    for fam in ['zi', 'cmp', 'zt']:
        moc = s[s.model == fam].groupby('eff')['cost'].mean()
        if not len(moc): continue
        line = ' '.join(f'e={e}:{100*(moc[e]-flc[e])/flc[e]:+.1f}%' for e in sorted(flc.index))
        print(f'[{city}/{fam}] eff-sensitivity κ=100 valrho0.2 cost vs floor: {line}')
# ---- P0-2: 阈值例 (NYC crash τ=0.20, real 空间): 报全 alerts/hits/recall/precision ----
th = df[(df.space == 'real') & (df.city == 'nyc') & (df.ch == 'c') & (df.policy.isin(['thresh_raw', 'thresh_cal'])) & (df.key == 'tau0.2') & (df.kappa == 100)]
for pol in ['thresh_raw', 'thresh_cal']:
    t = th[th.policy == pol].groupby('model')[['alerts', 'hits', 'recall', 'precision', 'cost']].mean()
    print(f'[NYC/c τ=0.20 REAL] {pol}:\n{t.round(4).to_string()}')
