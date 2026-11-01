# Sequential per-cycle patrol decisions under a fixed capacity.
# Each 3h cycle ranks its own 1,200 judged cell-hours from issued forecasts only and inspects the
# top B; capacity cannot move across cycles. Reports injury recall, inspections, and simulated cost,
# with 7-day-block paired bootstrap intervals.
# Outputs: sequential_capacity.csv, sequential_capacity_ci.csv

import numpy as np, pandas as pd, os
from scipy.stats import nbinom
WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
JT = f'{WS}/experiments/results/joint'
KAPPA, NBOOT, RNG = 100, 500, 7
SEEDS = {'6666': '', '1234': '_s1234', '2025': '_s2025', '777': '_s777'}
FRACS = [0.01, 0.02, 0.025, 0.03, 0.04, 0.05]
CITY = dict(
    nyc=dict(t3='T3_region_hour', h0=24*1461, btr=24*1095, nw=2913, twin='test2025',
             zi='joint3_zi', cmp='joint3_cmp_cmp'),
    chi=dict(t3='T3_region_hour_chicago', h0=24*1461, btr=24*1095, nw=2913, twin='test2025',
             zi='joint3_zi_chi', cmp='joint3_cmp_chi_cmp'),
    taichung=dict(t3='T3_region_hour_taichung', h0=26280, btr=17520, nw=2921, twin='test2024',
            zi='joint3_zi_taichung_zi', cmp='joint3_cmp_taichung_cmp', zt='joint3_cmpzt_taichung_zt'))
jr = pd.read_csv(f'{JT}/joint_results.csv')
zta = pd.read_csv(f'{JT}/zt_analysis.csv')

def clim(y, btr):
    C = y.astype(np.float64); g = C[:, :btr].mean()
    how = np.arange(C.shape[1]) % 168
    cl = np.zeros((C.shape[0], 168))
    for w in range(168):
        m = how[:btr] == w
        cl[:, w] = (C[:, :btr][:, m].sum(1) + 0.5) / (m.sum() + 0.5 / max(g, 1e-9))
    return cl

rows, ci_rows, rows_h, ci_h = [], [], [], []
for c, cfg in CITY.items():
    L = np.load(f'{WS}/data/processed/{cfg["t3"]}/labels.npz')
    yi = L['region_hour_injured'].astype(np.float64)
    nw = cfg['nw']; s = cfg['h0'] + 3 * np.arange(nw)
    tt = s[:, None] + 12 + np.arange(3)[None, :]          # (nw,3) 被判真实小时
    Y = np.transpose(yi[:, tt], (1, 2, 0)).reshape(nw, 1200)   # (nw,1200) j=h*400+cell
    hod = (tt.ravel() % 168)
    cli = clim(yi, cfg['btr'])
    score = {}
    hodm = ((s[:, None] + 12 + np.arange(3)[None, :]) % 168)  # (nw,3)
    score['floor'] = np.concatenate([cli[:, hodm[:, h]].T for h in range(3)], axis=1)  # (nw,1200), j=h*400+cell
    for sd, sfx in SEEDS.items():
        d = np.load(f'{JT}/mu_{cfg["zi"]}{sfx}_test.npz')
        pic = np.clip(d['pi_c'], 1e-7, 1-1e-7); muc = np.maximum(d['mu_c'], 1e-6)
        pii = np.clip(d['pi_i'], 1e-7, 1-1e-7); r = d['r']
        ai = float(jr[(jr.arm == cfg['zi']+sfx) & (jr.window == cfg['twin'])]['alpha_i'].iloc[0])
        ni = 1.0/ai; mui = r*(1-pic)*muc
        pocc = 1 - (pii + (1-pii)*(ni/(ni+np.maximum(mui, 1e-6)))**ni)   # (nw,400,12)
        pocc = pocc[:, :, 0:3]
        score[('zi', sd)] = np.transpose(pocc, (0, 2, 1)).reshape(nw, 1200)
        arm = cfg.get('zt') if c == 'taichung' else cfg['cmp']
        dc = np.load(f'{JT}/mu_{arm}{sfx}_test.npz')
        pic2 = np.clip(dc['pi_c'], 1e-7, 1-1e-7); mu2 = np.maximum(dc['mu_c'], 1e-6); r2 = dc['r']
        ac = float(jr[(jr.arm == arm+sfx) & (jr.window == cfg['twin'])]['alpha_c'].iloc[0])
        nc = 1.0/ac; pc = nc/(nc+mu2)
        if c == 'taichung':
            pocc_c = 1 - (pic2 + (1-pic2)*pc**nc)                            # ZTNB: P(y_i=0)=P(y_c=0)
        else:
            a_s = float(zta[(zta.city == c) & (zta.seed == int(sd))]['cmp_alpha_s'].iloc[0])
            ns = 1.0/a_s; ps = ns/(ns+np.clip(r2, 1e-6, 50)); t_ = ps**ns
            pocc_c = 1 - (pic2 + (1-pic2)*(pc/(1-(1-pc)*t_))**nc)             # NB compound pgf 零质量
        score[('cmp', sd)] = np.transpose(pocc_c[:, :, 0:3], (0, 2, 1)).reshape(nw, 1200)
    tot_ev = Y.sum()
    day = (np.arange(nw) * 3) // 24 // 7                                      # 7 天块 id (周期→天→周块)
    nblk = day.max() + 1
    blk_of_cycle = day
    for f in FRACS:
        B = int(round(f * 1200))
        ins = B * nw
        for m in ['floor', 'zi', 'cmp']:
            if m == 'floor':
                S = score['floor']
                order = np.argsort(-S, axis=1, kind='stable')
                hits_per_cycle = np.take_along_axis(Y, order[:, :B], 1).sum(1)
                for sd in SEEDS:
                    rows.append(dict(city=c, model=m, seed=sd, B_frac=f, B_per_cycle=B,
                                     recall=float(hits_per_cycle.sum()/tot_ev), inspections=ins,
                                     cost=float(ins + KAPPA*(tot_ev - hits_per_cycle.sum()))))
                hm = hits_per_cycle  # 单值 (floor 无 seed 维)
                recs = {sd: hm for sd in SEEDS}
            else:
                recs = {}
                for sd in SEEDS:
                    S = score[(m, sd)]
                    order = np.argsort(-S, axis=1, kind='stable')
                    recs[sd] = np.take_along_axis(Y, order[:, :B], 1).sum(1)
                hm = np.mean([recs[sd] for sd in SEEDS], axis=0)
                for sd in SEEDS:
                    h = recs[sd].sum()
                    rows.append(dict(city=c, model=m, seed=sd, B_frac=f, B_per_cycle=B,
                                     recall=float(h/tot_ev), inspections=ins,
                                     cost=float(ins + KAPPA*(tot_ev - h))))
            # 4-seed 平均逐周期命中 vs floor 的 7 天块 paired bootstrap
            hf = recs['6666'] * 0  # placeholder replaced below
            hits_floor = None
            Sf = score['floor']
            of = np.argsort(-Sf, axis=1, kind='stable')
            hits_floor = np.take_along_axis(Y, of[:, :B], 1).sum(1)
            hm = np.mean([recs[sd] for sd in SEEDS], axis=0) if m != 'floor' else hits_floor
            d_hit = (hm - hits_floor) if m != 'floor' else np.zeros(nw)
            d_cost = (KAPPA * -d_hit)  # inspections 相同 → 成本差 = −κ·Δhits
            blk_sum_d = np.array([d_hit[blk_of_cycle == b].sum() for b in range(nblk)])
            blk_sum_f = np.array([hits_floor[blk_of_cycle == b].sum() for b in range(nblk)])
            rng = np.random.default_rng(RNG)
            idx = rng.integers(0, nblk, size=(NBOOT, nblk))
            bs_d = blk_sum_d[idx].sum(1) / tot_ev
            bs_f = blk_sum_f[idx].sum(1) / tot_ev
            bs_m = (blk_sum_f[idx] + blk_sum_d[idx]).sum(1) / tot_ev
            ci = (float(np.quantile(bs_d, .025)), float(np.quantile(bs_d, .975)))
            ci_rows.append(dict(city=c, model=m, B_frac=f, B_per_cycle=B,
                                eb_recall=float(hits_floor.sum()/tot_ev),
                                model_recall=float(hm.sum()/tot_ev),
                                d_recall=float(d_hit.sum()/tot_ev),
                                d_recall_ci_lo=ci[0], d_recall_ci_hi=ci[1],
                                sig_better=bool(ci[0] > 0), sig_worse=bool(ci[1] < 0)))
    # ---- optional: fixed hourly capacity (same total per 3-h cycle) ----
    hidx = [np.arange(hh*400, (hh+1)*400) for hh in range(3)]
    for f in FRACS:
        B = int(round(f*1200)); cap = int(np.ceil(B/3)); ins = B*nw
        hits_floor = None
        for m in ['floor', 'zi', 'cmp']:
            if m == 'floor':
                chosen = np.concatenate([hh[np.argsort(-score['floor'][:, hh], axis=1, kind='stable')[:, :cap]]
                                         for hh in hidx], axis=1)
                hits_floor = np.take_along_axis(Y, chosen, 1).sum(1)
                for sd in SEEDS:
                    rows_h.append(dict(city=c, model=m, seed=sd, B_frac=f, cap_per_hour=cap,
                                       recall=float(hits_floor.sum()/tot_ev), inspections=ins,
                                       cost=float(ins + KAPPA*(tot_ev - hits_floor.sum()))))
                recs = {sd: hits_floor for sd in SEEDS}
            else:
                recs = {}
                for sd in SEEDS:
                    S = score[(m, sd)]
                    chosen = np.concatenate([hh[np.argsort(-S[:, hh], axis=1, kind='stable')[:, :cap]]
                                             for hh in hidx], axis=1)
                    recs[sd] = np.take_along_axis(Y, chosen, 1).sum(1)
                    h = recs[sd].sum()
                    rows_h.append(dict(city=c, model=m, seed=sd, B_frac=f, cap_per_hour=cap,
                                       recall=float(h/tot_ev), inspections=ins,
                                       cost=float(ins + KAPPA*(tot_ev - h))))
            hmh = np.mean([recs[sd] for sd in SEEDS], axis=0)
            d_hit = hmh - hits_floor
            blk_d = np.array([d_hit[blk_of_cycle == b].sum() for b in range(nblk)])
            bs = blk_d[idx].sum(1) / tot_ev
            ci_h.append(dict(city=c, model=m, B_frac=f, cap_per_hour=cap,
                             eb_recall=float(hits_floor.sum()/tot_ev),
                             model_recall=float(hmh.sum()/tot_ev),
                             d_recall=float(d_hit.sum()/tot_ev),
                             d_recall_ci_lo=float(np.quantile(bs, .025)),
                             d_recall_ci_hi=float(np.quantile(bs, .975)),
                             sig_better=bool(np.quantile(bs, .025) > 0)))
pd.DataFrame(rows_h).to_csv(f'{JT}/sequential_capacity_hourly.csv', index=False)
pd.DataFrame(ci_h).to_csv(f'{JT}/sequential_capacity_hourly_ci.csv', index=False)

pd.DataFrame(rows).to_csv(f'{JT}/sequential_capacity.csv', index=False)
pd.DataFrame(ci_rows).to_csv(f'{JT}/sequential_capacity_ci.csv', index=False)
ci = pd.DataFrame(ci_rows)
print(ci[ci.B_frac == 0.025].to_string(index=False))
print('--- recall by B (4-seed mean, model vs floor) ---')
print(ci.pivot_table(index=['city', 'model'], columns='B_frac', values='d_recall').round(4).to_string())
