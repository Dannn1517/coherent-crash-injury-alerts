# : 输入上报延迟敏感性 (读 t3hour_joint_hier 的 JOINT_SHIFT_H 产物 mu_*_sh{K}_test.npz)
# 指标 (zi 臂, test 窗, window 空间): dNLL% crash/injury vs EB 矩匹配地板 (shift K vs K=0),
#   顺序逐周期协议 injury recall @2.5% (shifted score vs 未移 floor), 与未移 zi 的 Δrecall 对照
# 输出: experiments/results/joint/latency_shift_sensitivity.csv
import numpy as np, pandas as pd, os
from scipy.stats import nbinom
WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
JT = f'{WS}/experiments/results/joint'
SEEDS = {'6666': '', '1234': '_s1234'}
CITY = dict(
    nyc=dict(t3='T3_region_hour', h0=24*1461, nw=2913, btr=24*1095, twin='test2025', tag=''),
    chi=dict(t3='T3_region_hour_chicago', h0=24*1461, nw=2913, btr=24*1095, twin='test2025', tag='_chi'),
    taichung=dict(t3='T3_region_hour_taichung', h0=26280, nw=2921, btr=17520, twin='test2024', tag='_taichung_zi'))
jr = pd.read_csv(f'{JT}/joint_results.csv')
KS = [24, 72]


def clim(y, btr):
    C = y.astype(np.float64); g = C[:, :btr].mean(); how = np.arange(C.shape[1]) % 168
    cl = np.zeros((C.shape[0], 168))
    for w in range(168):
        m = how[:btr] == w
        cl[:, w] = (C[:, :btr][:, m].sum(1) + 0.5) / (m.sum() + 0.5 / max(g, 1e-9))
    return cl


def zinb_nll(y, mu, pi, a):
    n = 1/a
    lnb = nbinom.logpmf(y, n, n/(n+np.maximum(mu, 1e-6)))
    return -np.logaddexp(np.log(pi) + np.where(y == 0, 0., -30.), np.log1p(-pi) + lnb)


rows = []
for c, cfg in CITY.items():
    L = np.load(f'{WS}/data/processed/{cfg["t3"]}/labels.npz')
    yc = L['region_hour'].astype(np.float64); yi = L['region_hour_injured'].astype(np.float64)
    nw = cfg['nw']; s = cfg['h0'] + 3*np.arange(nw)
    tt = s[:, None] + 12 + np.arange(12)[None, :]
    YC = np.transpose(yc[:, tt], (1, 0, 2)); YI = np.transpose(yi[:, tt], (1, 0, 2))
    # EB 地板 (train 矩匹配, 与 pipeline_e2e 同式)
    def eb_nll(Y, chan):
        g = chan[:, :cfg['btr']].mean(); var = chan[:, :cfg['btr']].var()
        a = float(max((var-g)/max(g*g, 1e-12), 1e-4)); n = 1/a
        hod = tt.ravel() % 168
        cl = np.transpose(clim(chan, cfg['btr'])[:, hod].reshape(400, nw, 12), (1, 0, 2))
        return float(zinb_nll(Y, cl, np.zeros_like(cl), a).mean())
    eb_c, eb_i = eb_nll(YC, yc), eb_nll(YI, yi)
    tt3 = s[:, None] + 12 + np.arange(3)[None, :]
    Y3 = np.transpose(yi[:, tt3], (1, 2, 0)).reshape(nw, 1200)
    hodm = (tt3 % 168)
    floor = np.concatenate([clim(yi, cfg['btr'])[:, hodm[:, h]].T for h in range(3)], axis=1)
    tot_ev = Y3.sum()
    order_f = np.argsort(-floor, axis=1, kind='stable')
    B = 30
    rec_floor = float(np.take_along_axis(Y3, order_f[:, :B], 1).sum()/tot_ev)
    for sd, sfx in SEEDS.items():
        for K in [0] + KS:
            tagf = f'_sh{K}' if K else ''
            fp = f'{JT}/mu_joint3_zi{cfg["tag"]}{sfx}{tagf}_test.npz'
            if not os.path.exists(fp):
                print('missing', fp); continue
            d = np.load(fp)
            pic = np.clip(d['pi_c'], 1e-7, 1-1e-7); muc = np.maximum(d['mu_c'], 1e-6)
            pii = np.clip(d['pi_i'], 1e-7, 1-1e-7); r = d['r']
            ac = float(jr[(jr.arm == 'joint3_zi'+cfg['tag']+sfx) & (jr.window == cfg['twin'])]['alpha_c'].iloc[0])
            ai = float(jr[(jr.arm == 'joint3_zi'+cfg['tag']+sfx) & (jr.window == cfg['twin'])]['alpha_i'].iloc[0])
            nll_c = float(zinb_nll(YC, muc, pic, ac).mean())
            mui = r*(1-pic)*muc
            nll_i = float(zinb_nll(YI, mui, pii, ai).mean())
            pocc = 1 - (pii + (1-pii)*((1/ai)/((1/ai)+np.maximum(mui, 1e-6)))**(1/ai))
            sc = np.transpose(pocc[:, :, 0:3], (0, 2, 1)).reshape(nw, 1200)
            rec = float(np.take_along_axis(Y3, np.argsort(-sc, axis=1, kind='stable')[:, :B], 1).sum()/tot_ev)
            rows.append(dict(city=c, seed=sd, shift_h=K, dnll_crash=100*(nll_c-eb_c)/eb_c,
                             dnll_injury=100*(nll_i-eb_i)/eb_i, recall_25=rec, recall_floor=rec_floor))
D = pd.DataFrame(rows)
D.to_csv(f'{JT}/latency_shift_sensitivity.csv', index=False)
piv = D.pivot_table(index=['city', 'seed'], columns='shift_h', values=['dnll_crash', 'dnll_injury', 'recall_25'])
print(piv.round(4).to_string())
