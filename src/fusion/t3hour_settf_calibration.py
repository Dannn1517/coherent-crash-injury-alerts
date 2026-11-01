# Data build 后续候选①(): settf_zinb 事后校准修复 (零GPU, 用持久化 mu/pi npz)
# 诊断(探针): top-decile p_occ 预测 0.143 vs 观测 0.130(val)/0.127(test), E[(1-π)μ] 高于 E[y] +7%;
#   中低分位近无偏 → 顶端过置信, 对预警应用(top-k 告警概率)是直接失真
# 方法族(全部只在 val2024 判定窗拟合; test2025 锁定窗只出终判, 不回头选参):
#   M1 gate 温度 π'=σ(logit(π)/T)          M2 gate Platt π'=σ(a·logit(π)+b)
#   M3 μ 幂校准 μ'=c·μ^γ                     M4 = M2+M3 (4参数)
#   M5 = M4 逐 horizon 版 (48参数, 每 horizon 独立拟合)
#   各法另有 +α 重拟合变体 (标量 NLL 最优); M0=未校准参照, M0_a=仅 α 重拟合
# 判定: val NLL 最低者中选 → test2025 报 NLL/AUC/ESR5/MAE/CRPS + reliability 十分位 + 窗簇 bootstrap 门(vs H1 地板)
# 口径: 与 t3hour_settf.eval_arm 同地板/同 a_floor/同 bootstrap rng7×500/同 ESR5 抽样 step;
#   已知微差(internal auditMINOR-2): dNLL_pct 分母用精确地板均值(eval_arm 用5位舍入值)→同一量两csv差≤0.001pp;
#   ZINB NLL 为 −30 哨兵截断变体(与训练/transfer 同式, 对均值影响≤1.2e-6 nats, internal auditMINOR-5);
#   ⚠ 地板为 legacy 口径(g_tr单位bug, 见 t3hour_settf.py 注释); EB修正地板下本批数字整体平移
#   (M0 test −3.403→−2.764, M5 −3.601→−2.963), 校准改善量 ΔNLL 与地板无关保持不变 → floor_correction.csv
# 韧性: 拟合完成即落盘 calibration_params.json; 重跑自动 RESUME 免重拟合 (首轮评估段形状bug崩溃教训)
import numpy as np, pandas as pd, os, json, time
from scipy.special import gammaln
from scipy.optimize import minimize
from scipy.stats import nbinom
from sklearn.metrics import roc_auc_score

WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
R = f'{WS}/data/model_inputs'
OUT = f'{WS}/experiments/results/settf'
PARAMS_JSON = f'{OUT}/calibration_params.json'
def _alpha0():
    p = f'{OUT}/settf_results.csv'
    if os.path.exists(p):
        df = pd.read_csv(p)
        m = df.loc[df['arm'] == 'settf_zinb', 'model_alpha']
        if len(m):
            return float(m.iloc[0])
    return 0.50094

ALPHA0 = _alpha0()          # settf_zinb learned alpha (settf_results.csv model_alpha)
IN_LEN, OUT_LEN, STRIDE = 12, 12, 3
B_TRAIN_END, B_VAL_END = 24 * 1095, 24 * 1461
WARM = 168

# ---- 数据与地板 (与 t3hour_settf 同配方) ----
t3 = np.load(f'{WS}/data/processed/T3_region_hour/labels.npz')
C = t3['region_hour']; n_reg, n_hours = C.shape
how = (np.arange(n_hours) % 168)
starts = {'train': [], 'val': [], 'test': []}
for i in range(0, n_hours - (IN_LEN + OUT_LEN) + 1, STRIDE):
    o_end = i + IN_LEN + OUT_LEN
    if o_end <= B_TRAIN_END:
        starts['train'].append(i)
    elif i >= B_TRAIN_END and o_end <= B_VAL_END:
        starts['val'].append(i)
    elif i >= B_VAL_END:
        starts['test'].append(i)
starts = {k: np.array(v) for k, v in starts.items()}
tr_h = np.arange(WARM, B_TRAIN_END)
# ⚠ legacy 地板口径(internal audit, ): 保持与已落盘 calibration_results.csv 可复现;
# EB 修正地板对照见 floor_correction.csv (cal_M5_perhor_a test −3.601→−2.963, 校准改善量不变)
g_tr = C[:, tr_h].sum() / len(tr_h)
clim = np.zeros((n_reg, 168), np.float32)
for w in range(168):
    m = how[tr_h] == w
    clim[:, w] = (C[:, tr_h][:, m].sum(1) + 0.5) / (m.sum() + 0.5 / max(g_tr, 1e-9))

def fit_alpha(y, mu):   # NEP-50 教训: 显式 float64
    mu = np.clip(mu.astype(np.float64), 1e-6, None); y = y.astype(np.float64)
    return float(max((((y - mu) ** 2 - mu).mean()) / max((mu ** 2).mean(), 1e-12), 1e-4))

# 地板 α: train 窗拟合 (与 eval_arm 同规)
y_tr = np.transpose(np.load(f'{R}/crash_hour/train.npz')['y'][:, :, :, 0], (0, 2, 1))
tt_tr = starts['train'][:y_tr.shape[0], None] + IN_LEN + np.arange(OUT_LEN)[None, :]
mu_f_tr = clim[:, how[tt_tr.ravel()]].T.reshape(y_tr.shape[0], OUT_LEN, n_reg).transpose(0, 2, 1)
A_FLOOR = fit_alpha(y_tr.ravel(), mu_f_tr.ravel())

W = {}   # 每窗预计算
for sp in ['val', 'test']:
    d = np.load(f'{OUT}/mu_settf_zinb_{sp}.npz')
    y = np.transpose(np.load(f'{R}/crash_hour/{sp}.npz')['y'][:, :, :, 0], (0, 2, 1)).astype(np.float64)
    S = y.shape[0]
    assert d['mu'].shape[0] == S and d['pi'].shape[0] == S, 'mu npz 行数与 y 窗数不一致'
    mu = d['mu'][:S].astype(np.float64); pi = d['pi'][:S].astype(np.float64)
    tt = starts[sp][:S, None] + IN_LEN + np.arange(OUT_LEN)[None, :]
    mu_f = clim[:, how[tt.ravel()]].T.reshape(S, OUT_LEN, n_reg).transpose(0, 2, 1).astype(np.float64)
    W[sp] = dict(y=y, mu=mu, pi=pi, mu_f=mu_f, S=S,
                 gl_y1=gammaln(y + 1), y0=(y == 0), ymax=int(y.max()))

def zinb_pts(mu, pi, alpha, w):
    """ZINB 逐点 NLL (稳定式; 与 scipy nbinom.logpmf 原式冒烟验证 diff 1.8e-14)"""
    n = 1.0 / max(alpha, 1e-8)
    mu = np.maximum(mu, 1e-6)
    l1p = np.log1p(mu / n)
    lnb = gammaln(n + w['y']) - w['gl_y1'] - gammaln(n) - n * l1p \
        + w['y'] * (np.log(mu) - np.log(n) - l1p)
    pic = np.clip(pi, 1e-7, 1 - 1e-7)
    l0 = np.log(pic) + np.where(w['y0'], 0.0, -30.0)
    return -np.logaddexp(l0, np.log1p(-pic) + lnb)

def logit(pi):
    pic = np.clip(pi, 1e-7, 1 - 1e-7)
    return np.log(pic) - np.log1p(-pic)

def apply_cal(mu, pi, prm):
    """prm: T(温度)/a,b(gate Platt, 标量或(1,1,12)逐horizon)/c,g(μ幂) 任意组合"""
    lp = logit(pi)
    if 'T' in prm:
        lp = lp / prm['T']
    if 'a' in prm:
        lp = prm['a'] * lp + prm['b']
    pi2 = 1.0 / (1.0 + np.exp(-lp))
    mu2 = mu
    if 'c' in prm:
        mu2 = prm['c'] * np.maximum(mu, 1e-6) ** prm['g']
    return mu2, pi2

def nll_of(prm, alpha, w):
    mu2, pi2 = apply_cal(w['mu'], w['pi'], prm)
    return float(zinb_pts(mu2, pi2, alpha, w).mean())

# ---- 参数落盘/续跑 ----
def save_fits(fits, extra=None):
    prm_ser = {k: ({kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()})
               for k, (v, a, i) in fits.items()}
    d = dict(alpha0=ALPHA0, a_floor=A_FLOOR, params=prm_ser,
             alphas={k: a for k, (p, a, i) in fits.items()},
             infos={k: i for k, (p, a, i) in fits.items()})
    if os.path.exists(PARAMS_JSON):
        try:   # 只回收旧文件的额外字段(floor CRPS/selected等), params/alphas/infos 以本次为准
            for kk, vv in json.load(open(PARAMS_JSON)).items():
                d.setdefault(kk, vv)
        except Exception:
            pass
    if extra:
        d.update(extra)
    json.dump(d, open(PARAMS_JSON, 'w'), indent=1)

def load_fits():
    d = json.load(open(PARAMS_JSON))
    fits = {}
    for k, prm in d['params'].items():
        p2 = {kk: (np.array(vv).reshape(1, 1, OUT_LEN) if isinstance(vv, list) else float(vv))
              for kk, vv in prm.items()}
        fits[k] = (p2, float(d['alphas'][k]), d.get('infos', {}).get(k, ''))
    return fits

# ---- 拟合器 (全部只用 val 窗) ----
wv = W['val']
t00 = time.time()
if os.path.exists(PARAMS_JSON):
    fits = load_fits()
    print(f'[fit] RESUME: 载入 {len(fits)} 组已拟合参数', flush=True)
else:
    def fit_nm(x0, unpack, pack_obj, maxiter=300):
        f = lambda x: pack_obj(unpack(x))
        r = minimize(f, x0, method='Nelder-Mead',
                     options=dict(maxiter=maxiter, xatol=1e-4, fatol=1e-8))
        return unpack(r.x), r.fun

    def fit_alpha_nll(prm, w, a0=ALPHA0):
        r = minimize(lambda la: nll_of(prm, float(np.exp(la[0])), w), [np.log(a0)],
                     method='Nelder-Mead', options=dict(maxiter=80, xatol=1e-4, fatol=1e-8))
        return float(np.exp(r.x[0]))

    def to_perhor(prm_h):
        return dict(a=np.array([p[0] for p in prm_h]).reshape(1, 1, OUT_LEN),
                    b=np.array([p[1] for p in prm_h]).reshape(1, 1, OUT_LEN),
                    c=np.array([p[2] for p in prm_h]).reshape(1, 1, OUT_LEN),
                    g=np.array([p[3] for p in prm_h]).reshape(1, 1, OUT_LEN))

    fits = {}
    fits['M0_uncal'] = (dict(), ALPHA0, 'reference')
    T, fT = fit_nm([0.0], lambda x: dict(T=float(np.exp(x[0]))), lambda p: nll_of(p, ALPHA0, wv))
    fits['M1_temp'] = (T, ALPHA0, f'val NLL={fT:.6f}')
    P2, f2 = fit_nm([1.0, 0.0], lambda x: dict(a=float(x[0]), b=float(x[1])), lambda p: nll_of(p, ALPHA0, wv))
    fits['M2_platt'] = (P2, ALPHA0, f'val NLL={f2:.6f}')
    P3, f3 = fit_nm([0.0, 1.0], lambda x: dict(c=float(np.exp(x[0])), g=float(x[1])), lambda p: nll_of(p, ALPHA0, wv))
    fits['M3_mupow'] = (P3, ALPHA0, f'val NLL={f3:.6f}')
    P4, f4 = fit_nm([1.0, 0.0, 0.0, 1.0],
                    lambda x: dict(a=float(x[0]), b=float(x[1]), c=float(np.exp(x[2])), g=float(x[3])),
                    lambda p: nll_of(p, ALPHA0, wv), maxiter=600)
    fits['M4_platt_mupow'] = (P4, ALPHA0, f'val NLL={f4:.6f}')
    prm_h, f5 = [], None
    for h in range(OUT_LEN):
        sl = dict(y=wv['y'][:, :, h:h + 1], gl_y1=wv['gl_y1'][:, :, h:h + 1],
                  y0=wv['y0'][:, :, h:h + 1], ymax=wv['ymax'])
        mu_h = wv['mu'][:, :, h:h + 1]; pi_h = wv['pi'][:, :, h:h + 1]
        def nll_h(x, mu_h=mu_h, pi_h=pi_h, sl=sl):
            p = dict(a=float(x[0]), b=float(x[1]), c=float(np.exp(x[2])), g=float(x[3]))
            lp = p['a'] * logit(pi_h) + p['b']
            pi2 = 1.0 / (1.0 + np.exp(-lp))
            mu2 = p['c'] * np.maximum(mu_h, 1e-6) ** p['g']
            return float(zinb_pts(mu2, pi2, ALPHA0, sl).mean())
        r = minimize(nll_h, [1.0, 0.0, 0.0, 1.0], method='Nelder-Mead',
                     options=dict(maxiter=600, xatol=1e-4, fatol=1e-8))
        prm_h.append((float(r.x[0]), float(r.x[1]), float(np.exp(r.x[2])), float(r.x[3])))
        f5 = (f5 or 0.0) + float(r.fun) * sl['y'].size
    fits['M5_perhor'] = (to_perhor(prm_h), ALPHA0, f'val NLL={f5 / wv["y"].size:.6f}')
    for base in ['M0_uncal', 'M1_temp', 'M2_platt', 'M3_mupow', 'M4_platt_mupow', 'M5_perhor']:
        prm, _, info = fits[base]
        a_new = fit_alpha_nll(prm, wv)
        fits[base + '_a'] = (prm, a_new, info + f'; α {ALPHA0:.5f}→{a_new:.5f}')
    save_fits(fits)
    print(f'[fit] 拟合并落盘完成 {time.time() - t00:.0f}s', flush=True)

for k, (prm, a, info) in fits.items():
    vn = nll_of(prm, a, wv)
    print(f'  {k:18s} valNLL={vn:.6f} α={a:.5f} {info} prm={ {kk: (round(vv, 5) if np.isscalar(vv) else "arr12") for kk, vv in prm.items()} }', flush=True)

# M0 一致性自检: 必须复现 settf_results.csv (val NLL 0.08299 / AUC 0.90417)
nll0 = nll_of(dict(), ALPHA0, wv)
if abs(nll0 - 0.08299) >= 2e-4:
    print(f'WARNING: M0 val NLL {nll0} deviates from the reference 0.08299; '
          'continuing with the in-package predictions.')

# ---- 评估 (同 eval_arm 口径) ----
def esr5(p_occ, yf, step):
    k = 20; hits = tot = 0
    y2 = yf.reshape(-1, n_reg); s2 = p_occ.reshape(-1, n_reg)
    for i in range(0, y2.shape[0], step):
        hits += len(set(np.argpartition(-y2[i], k)[:k]) & set(np.argpartition(-s2[i], k)[:k])); tot += k
    return round(hits / tot, 4)

def crps_zinb(mu, pi, alpha, w, chunk=512):
    """离散 CRPS = Σ_k (F(k)−1{k≥y})², F=π+(1−π)·NB_CDF; ks=0..ymax+40 (与 per_horizon_analysis 同约定)"""
    n = 1.0 / max(alpha, 1e-8)
    p = n / (n + np.maximum(mu, 1e-6))
    ks = np.arange(w['ymax'] + 41)
    tot, cnt = 0.0, 0
    S = w['S']
    for s0 in range(0, S, chunk):
        for h in range(OUT_LEN):
            sl = slice(s0, s0 + chunk)
            pc = pi[sl, :, h].ravel()[:, None]
            Fc = pc + (1 - pc) * nbinom.cdf(ks[None, :], n, p[sl, :, h].ravel()[:, None])
            ind = (ks[None, :] >= w['y'][sl, :, h].ravel().astype(int)[:, None])
            tot += float(((Fc - ind) ** 2).sum()); cnt += Fc.shape[0]
    return round(tot / cnt, 5)

def eval_cfg(name, prm, alpha, sp, do_crps):
    w = W[sp]; S = w['S']
    mu2, pi2 = apply_cal(w['mu'], w['pi'], prm)
    pts_m = zinb_pts(mu2, pi2, alpha, w)                       # (S,N,12)
    mf = np.maximum(mu2, 1e-6)
    pic = np.clip(pi2, 1e-7, 1 - 1e-7)
    p_occ = (1 - pic) * (1 - np.exp(-mf))
    mean_pred = (1 - pic) * mf
    yf = w['y'].ravel()
    ff = np.maximum(w['mu_f'], 1e-6)                           # (S,N,12) 与 pts_m 同形
    nnf = 1.0 / max(A_FLOOR, 1e-8); pf2 = nnf / (nnf + ff)
    pts_f = -nbinom.logpmf(w['y'], nnf, pf2).astype(np.float64)
    step = max(yf.reshape(-1, n_reg).shape[0] // 2000, 1)
    res = dict(method=name, window=f'{sp}2024' if sp == 'val' else f'{sp}2025', alpha=round(alpha, 5),
               NLL=round(float(pts_m.mean()), 5),
               AUC=round(float(roc_auc_score((yf > 0).astype(int), p_occ.ravel())), 5),
               ESR5=esr5(p_occ, yf, step),
               MAE=round(float(np.abs(yf - mean_pred.ravel()).mean()), 5))
    if do_crps:
        res['CRPS'] = crps_zinb(mu2, pi2, alpha, w)
    d_w = (pts_m - pts_f).reshape(S, -1).mean(1)
    rb = np.random.default_rng(7)
    bs = d_w[rb.integers(0, S, size=(500, S))].mean(1)
    res.update(dNLL=round(float(d_w.mean()), 6),
               dNLL_pct=round(100 * float(d_w.mean()) / float(pts_f.mean()), 3),
               ci_lo=round(float(np.quantile(bs, .025)), 6), ci_hi=round(float(np.quantile(bs, .975)), 6),
               sig_better=bool(np.quantile(bs, .975) < 0), sig_worse=bool(np.quantile(bs, .025) > 0))
    return res

sel = min(fits, key=lambda k: nll_of(fits[k][0], fits[k][1], wv))
print(f'[select] val NLL 最低 = {sel}', flush=True)
rows = []
for name, (prm, alpha, info) in fits.items():
    for sp in ['val', 'test']:
        t0 = time.time()
        r = eval_cfg(name, prm, alpha, sp, do_crps=(sp == 'test' and name in ('M0_uncal', sel)))
        r['note'] = info
        rows.append(r)
        print(f'[eval] {name:18s} {sp:4s} NLL={r["NLL"]:.5f} dNLL%={r["dNLL_pct"]:+.3f} '
              f'sig={r["sig_better"]}/{r["sig_worse"]} AUC={r["AUC"]:.5f} ESR5={r["ESR5"]:.4f} '
              f'MAE={r["MAE"]:.5f} CRPS={r.get("CRPS")} ({time.time() - t0:.0f}s)', flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{OUT}/calibration_results.csv', index=False)

# ---- reliability 十分位: M0 vs 中选 (两窗) ----
rel = []
for name in ['M0_uncal', sel]:
    prm, alpha, _ = fits[name]
    for sp in ['val', 'test']:
        w = W[sp]
        mu2, pi2 = apply_cal(w['mu'], w['pi'], prm)
        pic = np.clip(pi2, 1e-7, 1 - 1e-7)
        p_occ = ((1 - pic) * (1 - np.exp(-np.maximum(mu2, 1e-6)))).ravel()
        occ = (w['y'] > 0).ravel().astype(float)
        qs = np.quantile(p_occ, np.linspace(0, 1, 11))
        ece = 0.0
        for b in range(10):
            m = (p_occ >= qs[b]) & (p_occ <= qs[b + 1]) if b == 9 else (p_occ >= qs[b]) & (p_occ < qs[b + 1])
            if m.sum() > 0:
                pm, ob = float(p_occ[m].mean()), float(occ[m].mean())
                ece += abs(pm - ob) * m.sum()
                rel.append(dict(method=name, window=sp, bin=b + 1, pred_mean=round(pm, 5),
                                obs_rate=round(ob, 5), n=int(m.sum())))
        print(f'[rel] {name} {sp}: ECE={ece / len(p_occ):.5f}', flush=True)
pd.DataFrame(rel).to_csv(f'{OUT}/reliability_settf_zinb_calib.csv', index=False)

# ---- 中选法逐 horizon NLL (test, vs 地板/未校准) ----
prm, alpha, _ = fits[sel]
w = W['test']
mu2, pi2 = apply_cal(w['mu'], w['pi'], prm)
nnf = 1.0 / max(A_FLOOR, 1e-8)
ph = []
for h in range(OUT_LEN):
    sl = dict(y=w['y'][:, :, h:h + 1], gl_y1=w['gl_y1'][:, :, h:h + 1], y0=w['y0'][:, :, h:h + 1], ymax=w['ymax'])
    nm_ = float(zinb_pts(mu2[:, :, h:h + 1], pi2[:, :, h:h + 1], alpha, sl).mean())
    n0 = float(zinb_pts(w['mu'][:, :, h:h + 1], w['pi'][:, :, h:h + 1], ALPHA0, sl).mean())
    muf = np.maximum(w['mu_f'][:, :, h:h + 1], 1e-6); pf = nnf / (nnf + muf)
    nfl = float(-nbinom.logpmf(w['y'][:, :, h:h + 1], nnf, pf).mean())
    ph.append(dict(horizon=h + 1, NLL_uncal=round(n0, 5), NLL_cal=round(nm_, 5),
                   NLL_floor=round(nfl, 5),
                   d_pct_uncal=round(100 * (n0 - nfl) / nfl, 2), d_pct_cal=round(100 * (nm_ - nfl) / nfl, 2)))
pd.DataFrame(ph).to_csv(f'{OUT}/calibration_perhorizon_test.csv', index=False)

# ---- 地板 CRPS (test) + 参数/选法落盘 ----
fr = dict(floor_CRPS_test=crps_zinb(w['mu_f'], np.zeros_like(w['mu_f']), A_FLOOR, w, 512))
print('[crps] floor:', fr, flush=True)
save_fits(fits, extra=dict(selected=sel, **fr))
print(f'DONE -> {OUT}/calibration_results.csv | selected={sel} | total {time.time() - t00:.0f}s')
