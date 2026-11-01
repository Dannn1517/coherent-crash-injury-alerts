# : 独立 ZINB 双头对照 (marginal SetTF-ZINB per channel, 无联合建模/无耦合)
# 指标: (a) 顺序逐周期协议 injury recall @ 六档容量 + 7天块 paired CI vs EB floor
#       (b) 独立性代价: E-crossings P(E_i>E_c) 与不可能质量 mean(pocc_i * (1-pocc_c))
#       (c) injury dNLL% vs EB 直接读 settf_results.csv (test 窗, 4 seeds)
# 资产: experiments/results/settf/mu_settf_zinb{*}_inj*_test.npz (keys mu, pi) + 同城 crash 边际 npz
# 输出: experiments/results/settf/independent_dual_control.csv
import numpy as np, pandas as pd, os
WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
ST = f'{WS}/experiments/results/settf'
SEEDS = {'6666': '', '1234': '_s1234', '2025': '_s2025', '777': '_s777'}
CITY = dict(
    nyc=dict(t3='T3_region_hour', h0=24*1461, nw=2913, twin='test2025', ci='settf_zinb', ii='settf_zinb_inj'),
    chi=dict(t3='T3_region_hour_chicago', h0=24*1461, nw=2913, twin='test2025', ci='settf_zinb_chi', ii='settf_zinb_chi_inj'),
    taichung=dict(t3='T3_region_hour_taichung', h0=26280, nw=2921, twin='test2024', ci='settf_zinb_taichung', ii='settf_zinb_taichung_inj'))
sr = pd.read_csv(f'{ST}/settf_results.csv')
FRACS = [0.01, 0.02, 0.025, 0.03, 0.04, 0.05]
NBOOT, RNG = 500, 7


def clim(y, btr):
    C = y.astype(np.float64); g = C[:, :btr].mean(); how = np.arange(C.shape[1]) % 168
    cl = np.zeros((C.shape[0], 168))
    for w in range(168):
        m = how[:btr] == w
        cl[:, w] = (C[:, :btr][:, m].sum(1) + 0.5) / (m.sum() + 0.5 / max(g, 1e-9))
    return cl


def pocc(mu, pi, a):
    n = 1/a
    return 1 - (pi + (1-pi)*(n/(n+np.maximum(mu, 1e-6)))**n)


rows = []
for c, cfg in CITY.items():
    need = [f'{ST}/mu_{cfg["ii"]}{sfx}_test.npz' for sfx in SEEDS.values()]
    if not all(os.path.exists(p) for p in need):
        print(c, 'SKIP: missing independent injury npz'); continue
    L = np.load(f'{WS}/data/processed/{cfg["t3"]}/labels.npz')
    yi = L['region_hour_injured'].astype(np.float64)
    btr = 17520 if c == 'taichung' else 24*1095
    nw = cfg['nw']; s = cfg['h0'] + 3*np.arange(nw)
    tt = s[:, None] + 12 + np.arange(3)[None, :]
    Y = np.transpose(yi[:, tt], (1, 2, 0)).reshape(nw, 1200)
    hodm = ((s[:, None] + 12 + np.arange(3)[None, :]) % 168)
    cli = clim(yi, btr)
    floor = np.concatenate([cli[:, hodm[:, h]].T for h in range(3)], axis=1)
    tot_ev = Y.sum()
    day = (np.arange(nw)*3)//24//7
    nblk = day.max()+1
    rng = np.random.default_rng(RNG)
    idx = rng.integers(0, nblk, size=(NBOOT, nblk))
    blk_floor = np.array([np.take_along_axis(Y, np.argsort(-floor, axis=1, kind='stable')[:, :30], 1).sum(1)[day == b].sum() for b in range(nblk)])
    # coherence: independent pair (crash marginal x injury marginal), seed 6666 representative + mean over seeds
    ec_list, imp_list = [], []
    for sd, sfx in SEEDS.items():
        di = np.load(f'{ST}/mu_{cfg["ii"]}{sfx}_test.npz')
        if not os.path.exists(f'{ST}/mu_{cfg["ci"]}{sfx}_test.npz'):
            continue
        dc = np.load(f'{ST}/mu_{cfg["ci"]}{sfx}_test.npz')
        ai = float(sr[(sr.arm == cfg['ii']+sfx) & (sr.window == cfg['twin'])]['model_alpha'].iloc[0])
        ac = float(sr[(sr.arm == cfg['ci']+sfx) & (sr.window == cfg['twin'])]['model_alpha'].iloc[0])
        mu_i, pi_i = np.maximum(di['mu'], 1e-6), np.clip(di['pi'], 1e-7, 1-1e-7)
        mu_c, pi_c = np.maximum(dc['mu'], 1e-6), np.clip(dc['pi'], 1e-7, 1-1e-7)
        E_i = (1-pi_i)*mu_i; E_c = (1-pi_c)*mu_c
        ec_list.append(float((E_i > E_c).mean()*100))
        imp_list.append(float((pocc(mu_i, pi_i, ai)*(1-pocc(mu_c, pi_c, ac))).mean()*100))
    for f in FRACS:
        B = int(round(f*1200))
        blk_ind_all = []
        for sd, sfx in SEEDS.items():
            di = np.load(f'{ST}/mu_{cfg["ii"]}{sfx}_test.npz')
            ai = float(sr[(sr.arm == cfg['ii']+sfx) & (sr.window == cfg['twin'])]['model_alpha'].iloc[0])
            sc = pocc(np.maximum(di['mu'], 1e-6), np.clip(di['pi'], 1e-7, 1-1e-7), ai)[:, :, 0:3]
            sc = np.transpose(sc, (0, 2, 1)).reshape(nw, 1200)
            order = np.argsort(-sc, axis=1, kind='stable')
            pc = np.take_along_axis(Y, order[:, :B], 1).sum(1)
            blk_ind_all.append(np.array([pc[day == b].sum() for b in range(nblk)]))
            rows.append(dict(city=c, B_frac=f, model='ind', seed=sd, recall=float(pc.sum()/tot_ev)))
        blk_ind = np.mean(blk_ind_all, axis=0)
        dm = blk_ind - blk_floor
        bs = dm[idx].sum(1)
        lo, hi = np.quantile(bs, [0.025, 0.975]) / tot_ev
        rows.append(dict(city=c, B_frac=f, model='ind_ci', seed='mean', recall=float(dm.sum()/tot_ev),
                         ci_lo=lo, ci_hi=hi, sig_better=bool(lo > 0), sig_worse=bool(hi < 0)))
    dn = sr[(sr.arm.str.startswith(cfg['ii'])) & (sr.window == cfg['twin'])]['dNLL_pct']
    rows.append(dict(city=c, B_frac=-1, model='coh_ecross', seed='mean', recall=float(np.mean(ec_list)),
                     ci_lo=float(np.min(ec_list)), ci_hi=float(np.max(ec_list))))
    rows.append(dict(city=c, B_frac=-1, model='coh_imposs', seed='mean', recall=float(np.mean(imp_list)),
                     ci_lo=float(np.min(imp_list)), ci_hi=float(np.max(imp_list))))
    rows.append(dict(city=c, B_frac=-1, model='inj_dnll', seed='mean', recall=float(dn.mean()),
                     ci_lo=float(dn.std()), ci_hi=float(len(dn))))
    print(c, '| inj dNLL% mean/sd:', round(dn.mean(), 3), round(dn.std(), 3),
          '| E-cross%:', round(np.mean(ec_list), 2), '| imposs%:', round(np.mean(imp_list), 3))
D = pd.DataFrame(rows)
D.to_csv(f'{ST}/independent_dual_control.csv', index=False)
print(D[(D.B_frac == 0.025)].round(4).to_string(index=False))
