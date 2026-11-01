# 实验3 (): 端到端 pipeline 统一协议表 (Table 4) 的补算量
# pipelines: EB baseline / free-gate (zi+M5 calibrator) / compound (regime-matched family + 同型 M5 calibrator)
# 补算: EB crash ECE (test, window space), EB impossible-event mass (独立通道乘积), compound crash ECE
#   (per-horizon Platt(sklearn LR), decision-window 拟合 test 终判, 4-seed 均值; Taichung 用 matched ZTNB 臂),
#   顺序容量成本比 (引 calibrated sequential_capacity_calib.csv)
# 输出: experiments/results/joint/pipeline_e2e_supplement.csv
import numpy as np, pandas as pd, os
from sklearn.linear_model import LogisticRegression
WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
JT = f'{WS}/experiments/results/joint'
SEEDS = {'6666': '', '1234': '_s1234', '2025': '_s2025', '777': '_s777'}
CITY = dict(
    nyc=dict(t3='T3_region_hour', h0=24*1461, btr=24*1095, nw=2913, twin='test2025', vwin='val2024', cmp='joint3_cmp_cmp'),
    chi=dict(t3='T3_region_hour_chicago', h0=24*1461, btr=24*1095, nw=2913, twin='test2025', vwin='val2024', cmp='joint3_cmp_chi_cmp'),
    taichung=dict(t3='T3_region_hour_taichung', h0=26280, btr=17520, nw=2921, twin='test2024', vwin='val2023', cmp='joint3_cmp_taichung_cmp', zt='joint3_cmpzt_taichung_zt'))
jr = pd.read_csv(f'{JT}/joint_results.csv')

def clim(y, btr):
    C = y.astype(np.float64); g = C[:, :btr].mean(); how = np.arange(C.shape[1]) % 168
    cl = np.zeros((C.shape[0], 168))
    for w in range(168):
        m = how[:btr] == w
        cl[:, w] = (C[:, :btr][:, m].sum(1) + 0.5) / (m.sum() + 0.5 / max(g, 1e-9))
    return cl

def ece(p, y, B=10):
    p = p.ravel(); yb = (y.ravel() > 0).astype(float)
    o = np.argsort(p); q = np.array_split(o, B)
    return float(sum(abs(p[i].mean()-yb[i].mean())*len(i) for i in q) / len(p))

out = []
for c, cfg in CITY.items():
    L = np.load(f'{WS}/data/processed/{cfg["t3"]}/labels.npz')
    yc = L['region_hour'].astype(np.float64); yii = L['region_hour_injured'].astype(np.float64)
    nw = cfg['nw']; s = cfg['h0'] + 3*np.arange(nw); tt = s[:, None] + 12 + np.arange(12)[None, :]
    yw = np.transpose(yc[:, tt], (1, 0, 2)); hod = tt.ravel() % 168
    cli = np.transpose(clim(yc, cfg['btr'])[:, hod].reshape(400, nw, 12), (1, 0, 2))
    g = yc[:, :cfg['btr']].mean(); var = yc[:, :cfg['btr']].var()
    a_eb = float(max((var-g)/max(g*g, 1e-12), 1e-4)); n = 1/a_eb
    P0eb = (n/(n+cli))**n
    out.append(dict(city=c, metric='EB_crash_ECE_test', value=round(ece(1-P0eb, yw), 5)))
    cli_i = np.transpose(clim(yii, cfg['btr'])[:, hod].reshape(400, nw, 12), (1, 0, 2))
    gi = yii[:, :cfg['btr']].mean(); vari = yii[:, :cfg['btr']].var()
    a_i = float(max((vari-gi)/max(gi*gi, 1e-12), 1e-4)); ni = 1/a_i
    out.append(dict(city=c, metric='EB_impossible_mass_pct',
                    value=round(float((P0eb*(1-(ni/(ni+cli_i))**ni)).mean()*100), 3)))
    arm = cfg.get('zt', cfg['cmp']); eces = []
    for sd, sfx in SEEDS.items():
        dv = np.load(f'{JT}/mu_{arm}{sfx}_val.npz'); dt = np.load(f'{JT}/mu_{arm}{sfx}_test.npz')
        av = float(jr[(jr.arm == arm+sfx) & (jr.window == cfg['vwin'])]['alpha_c'].iloc[0])
        def pocc(d, a):
            pic = np.clip(d['pi_c'], 1e-7, 1-1e-7); mu = np.maximum(d['mu_c'], 1e-6); nn = 1/a
            return 1-(pic+(1-pic)*(nn/(nn+mu))**nn), pic, mu
        _, picv, _ = pocc(dv, av)
        nv = picv.shape[0]
        ttv = (cfg['btr'] - 12 + 3*np.arange(nv))[:, None] + 12 + np.arange(12)[None, :]
        yv = np.transpose(yc[:, ttv], (1, 0, 2))
        yt = yw
        logitv = np.log(picv) - np.log1p(-picv); ys = (yv > 0).astype(float)
        W = []
        for h in range(12):
            lr = LogisticRegression(penalty=None, max_iter=200).fit(
                logitv[:, :, h].ravel()[:, None], ys[:, :, h].ravel().astype(int))
            W.append((float(lr.coef_[0][0]), float(lr.intercept_[0])))
        def apply_pl(logit):
            return np.stack([1/(1+np.exp(-(W[h][0]*logit[:, :, h]+W[h][1]))) for h in range(12)], axis=2)
        pv_cal = apply_pl(logitv)
        logitt = np.log(np.clip(dt['pi_c'], 1e-7, 1-1e-7)) - np.log1p(-np.clip(dt['pi_c'], 1e-7, 1-1e-7))
        pt_cal = apply_pl(logitt)
        eces.append(ece(pt_cal, yt))
    out.append(dict(city=c, metric='compound_crash_ECE_test', value=round(float(np.mean(eces)), 5)))
    sq = pd.read_csv(f'{JT}/sequential_capacity_calib.csv'); q = sq[(sq.B_frac == 0.025) & (sq.city == c)]
    cf = q[q.model == 'floor'].cost.mean()
    for m in ['floor', 'zi', 'cmp']:
        out.append(dict(city=c, metric=f'seq_cost_ratio_{m}',
                        value=round(float(q[q.model == m].cost.mean()/cf), 4)))
D = pd.DataFrame(out)
D.to_csv(f'{JT}/pipeline_e2e_supplement.csv', index=False)
print(D.to_string(index=False))
