# : compound severity head (C1) 结果聚合与支撑精确性验证 (CPU, 只读产物)
import numpy as np, pandas as pd, os, glob, json
WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
JT = f'{WS}/experiments/results/joint'
jr = pd.read_csv(f'{JT}/joint_results.csv')

ARMS = {
 ('zi','nyc'): ['joint3_zi','joint3_zi_s1234','joint3_zi_s2025','joint3_zi_s777'],
 ('zi','chi'): ['joint3_zi_chi','joint3_zi_chi_s1234','joint3_zi_chi_s2025','joint3_zi_chi_s777'],
 ('cmp','nyc'): ['joint3_cmp_cmp','joint3_cmp_cmp_s1234','joint3_cmp_cmp_s2025','joint3_cmp_cmp_s777'],
 ('cmp','chi'): ['joint3_cmp_chi_cmp','joint3_cmp_chi_cmp_s1234','joint3_cmp_chi_cmp_s2025','joint3_cmp_chi_cmp_s777'],
 ('cmp2','nyc'): ['joint3_cmp2_cmp2','joint3_cmp2_cmp2_s1234','joint3_cmp2_cmp2_s2025','joint3_cmp2_cmp2_s777'],
 ('cmp2','chi'): ['joint3_cmp2_chi_cmp2','joint3_cmp2_chi_cmp2_s1234','joint3_cmp2_chi_cmp2_s2025','joint3_cmp2_chi_cmp2_s777'],
 ('cmp3','nyc'): ['joint3_cmp3_cmp3','joint3_cmp3_cmp3_s1234','joint3_cmp3_cmp3_s2025','joint3_cmp3_cmp3_s777'],
 ('cmp3','chi'): ['joint3_cmp3_chi_cmp3','joint3_cmp3_chi_cmp3_s1234','joint3_cmp3_chi_cmp3_s2025','joint3_cmp3_chi_cmp3_s777'],
 ('cmp4','nyc'): ['joint3_cmp4_cmp4','joint3_cmp4_cmp4_s1234','joint3_cmp4_cmp4_s2025','joint3_cmp4_cmp4_s777'],
 ('cmp4','chi'): ['joint3_cmp4_chi_cmp4','joint3_cmp4_chi_cmp4_s1234','joint3_cmp4_chi_cmp4_s2025','joint3_cmp4_chi_cmp4_s777'],
 ('cmp5','nyc'): ['joint3_cmp5_cmp5','joint3_cmp5_cmp5_s1234','joint3_cmp5_cmp5_s2025','joint3_cmp5_cmp5_s777'],
 ('cmp5','chi'): ['joint3_cmp5_chi_cmp5','joint3_cmp5_chi_cmp5_s1234','joint3_cmp5_chi_cmp5_s2025','joint3_cmp5_chi_cmp5_s777'],
 ('cmp6','nyc'): ['joint3_cmp6_cmp6','joint3_cmp6_cmp6_s1234','joint3_cmp6_cmp6_s2025','joint3_cmp6_cmp6_s777'],
 ('cmp6','chi'): ['joint3_cmp6_chi_cmp6','joint3_cmp6_chi_cmp6_s1234','joint3_cmp6_chi_cmp6_s2025','joint3_cmp6_chi_cmp6_s777'],
}
def agg(prefix, city):
    arms = [a for a in ARMS[(prefix, city)] if a in set(jr.arm)]
    return jr[(jr.arm.isin(arms)) & (jr.window == 'test2025')]

def support_check(arm):
    """compound 臂: 验证混合零质量 >= crash 零质量 (支撑精确的数值证据)"""
    d = np.load(f'{JT}/mu_{arm}_test.npz')
    mu = d['mu_c'].astype(np.float64); pic = np.clip(d['pi_c'].astype(np.float64), 1e-7, 1 - 1e-7)
    r = d['r'].astype(np.float64); a_s = d['alpha_s'].astype(np.float64)
    ac = float(jr[(jr.arm == arm) & (jr.window == 'test2025')].alpha_c.iloc[0])
    n_c = 1 / ac; p_c = n_c / (n_c + np.maximum(mu, 1e-6))
    n_s = 1 / np.maximum(a_s, 1e-6); p_s = n_s / (n_s + np.clip(r, 1e-6, 50))
    t = p_s ** n_s
    Pi0 = pic + (1 - pic) * (p_c / (1 - (1 - p_c) * t)) ** n_c     # P(y_i=0)
    Pc0 = pic + (1 - pic) * p_c ** n_c                              # P(y_c=0)
    return float((Pi0 >= Pc0 - 1e-12).mean()), float((Pc0 - Pi0).max())

rows = []
for city in ['nyc', 'chi']:
    for fam, pre in [('zi', 'joint3_zi'), ('cmp', 'joint3_cmp_cmp'), ('cmp2', 'joint3_cmp2_cmp2'), ('cmp3', 'joint3_cmp3_cmp3'), ('cmp4', 'joint3_cmp4_cmp4'), ('cmp5', 'joint3_cmp5_cmp5'), ('cmp6', 'joint3_cmp6_cmp6')]:
        sub = agg(fam, city)
        if len(sub) == 0:
            continue
        rows.append(dict(city=city, family=fam, n=len(sub),
                         c_dNLL=round(sub.c_dNLL_pct.mean(), 3), c_sd=round(sub.c_dNLL_pct.std(ddof=1), 3),
                         i_dNLL=round(sub.i_dNLL_pct.mean(), 3), i_sd=round(sub.i_dNLL_pct.std(ddof=1), 3),
                         i_AUC=round(sub.i_AUC.mean(), 5), i_R20=round(sub.i_Recall20.mean(), 4)))
        if fam != 'zi':
            oks, gaps = [], []
            for arm0 in ARMS[(fam, city)]:
                if arm0 in set(jr.arm) and os.path.exists(f'{JT}/mu_{arm0}_test.npz'):
                    ok, gap = support_check(arm0); oks.append(ok); gaps.append(gap)
            if oks:
                rows[-1]['support_frac_ok'] = round(float(np.mean(oks)), 4)
                rows[-1]['max_Pc0_minus_Pi0'] = float(f'{max(gaps):.2e}')
            dgs = [json.loads(x) for x in sub.rate_diag if isinstance(x, str) and x]
            dg = {k: round(float(np.mean([d[k] for d in dgs])), 4) for k in ['dyn_vs_static', 'dyn_vs_obs']} if dgs else {}
            rows[-1]['dyn_vs_obs'] = dg.get('dyn_vs_obs')
            rows[-1]['dyn_vs_static'] = dg.get('dyn_vs_static')
D = pd.DataFrame(rows)
D.to_csv(f'{JT}/cmp_summary.csv', index=False)
print(D.to_string(index=False))
