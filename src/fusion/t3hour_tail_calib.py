# (internal review ): 高风险尾部校准诊断 — 真实 grid-hour 去重空间
# 动机: 审稿人指出整体 ECE 不足以证明告警阈值附近校准可靠 → 补 (a) 尾部分箱 (pred≥0.05 区域)
#   等数分箱 pred vs obs + 样本数 + 日簇 bootstrap 95% CI; (b) 通信阈值集合 {p≥τ} 的兑现率与 CI;
#   (c) 尾部 ECE。口径: NYC crash 通道, SetTF-ZINB (与 F5 同源 mu_settf_zinb_test.npz),
#   M0_uncal vs M5_perhor_a (49参: 逐时距 gate Platt 24 + μ幂 24? — 以 calibration_params.json 为准),
#   校准参数只在 val2024 拟合 (由 t3hour_settf_calibration.py 完成, 本脚本只读)。
# 去重 ( 同协议): 每真实 (cell, hour) 只计一次, 取 h∈{0,1,2} (lead 12-14h 最新预报), 平铺无缝。
# CI: 日簇 bootstrap (500 次, 重抽 365 天), 成员集合固定 (分箱/阈值在全体上定), 只重抽兑现率。
# 输出: experiments/results/settf/tail_calib_real.csv + stdout
import numpy as np, pandas as pd, os, json
WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
R = f'{WS}/data/model_inputs'
OUT = f'{WS}/experiments/results/settf'
IN_LEN, OUT_LEN, STRIDE = 12, 12, 3
B_TRAIN_END, B_VAL_END = 24*1095, 24*1461
def _alpha0():
    p = f'{OUT}/settf_results.csv'
    if os.path.exists(p):
        df = pd.read_csv(p)
        m = df.loc[df['arm'] == 'settf_zinb', 'model_alpha']
        if len(m):
            return float(m.iloc[0])
    return 0.50094

ALPHA0 = _alpha0()
NB_BOOT = 500

t3 = np.load(f'{WS}/data/processed/T3_region_hour/labels.npz')
n_reg, n_hours = t3['region_hour'].shape
starts_test = np.array([i for i in range(0, n_hours-23, STRIDE) if i >= B_VAL_END])

d = np.load(f'{OUT}/mu_settf_zinb_test.npz')
y_all = np.transpose(np.load(f'{R}/crash_hour/test.npz')['y'][:, :, :, 0], (0, 2, 1)).astype(np.float64)
S = len(starts_test)
mu = d['mu'][:S].astype(np.float64); pi = d['pi'][:S].astype(np.float64)
prm_js = json.load(open(f'{OUT}/calibration_params.json'))
fits = {k: ({kk: (np.array(vv).reshape(1, 1, OUT_LEN) if isinstance(vv, list) else float(vv)) for kk, vv in p.items()},
            float(prm_js['alphas'][k])) for k, p in prm_js['params'].items()}
print('methods in params json:', sorted(fits.keys()))

def apply_cal(mu, pi, prm):
    lp = np.log(np.clip(pi, 1e-7, 1-1e-7)) - np.log1p(-np.clip(pi, 1e-7, 1-1e-7))
    if 'T' in prm: lp = lp / prm['T']
    if 'a' in prm: lp = prm['a'] * lp + prm['b']
    pi2 = 1.0 / (1.0 + np.exp(-lp))
    mu2 = prm['c'] * np.maximum(mu, 1e-6) ** prm['g'] if 'c' in prm else mu
    return mu2, pi2

def pocc(mu, pi, alpha):
    n = 1.0 / alpha
    nb0 = (n / (n + np.maximum(mu, 1e-6))) ** n
    pic = np.clip(pi, 1e-7, 1-1e-7)
    return 1.0 - (pic + (1.0 - pic) * nb0)

M0 = 'M0_uncal'; M5 = 'M5_perhor_a'
p_raw = pocc(mu, pi, ALPHA0)
mu5, pi5 = apply_cal(mu, pi, fits[M5][0])
p_cal = pocc(mu5, pi5, fits[M5][1])

# ---- 真实空间去重切片 ----
def dedup(a):   # (S,N,12) → (3S, N) 行按真实时刻升序平铺
    return np.transpose(a[:, :, 0:3], (0, 2, 1)).reshape(-1, n_reg)
tau_row = (starts_test[:S, None] + IN_LEN + np.arange(3)[None, :]).ravel()   # 每行真实小时
day_row = tau_row // 24
Y = dedup(y_all) > 0
P_RAW, P_CAL = dedup(p_raw), dedup(p_cal)
n_days = len(np.unique(day_row))
print(f'real space: rows={Y.shape[0]} (=3×{S} windows), cells={n_reg}, days={n_days}, base rate={Y.mean():.5f}')
udays = np.unique(day_row)
flat_Y = Y.ravel(); flat_day = np.repeat(day_row, n_reg)
rng = np.random.default_rng(7)
boot_day_sets = [rng.integers(0, len(udays), len(udays)) for _ in range(NB_BOOT)]

def rate_ci(mask_flat):
    """固定成员集合 (平铺 cell-hour 掩码) 的兑现率 + 日簇 bootstrap 95% CI"""
    idx = np.where(mask_flat)[0]
    if len(idx) == 0: return 0.0, np.nan, np.nan, 0
    r = float(flat_Y[idx].mean())
    days_of = flat_day[idx]; ys = flat_Y[idx].astype(np.float64)
    ud, inv = np.unique(days_of, return_inverse=True)
    cnt = np.bincount(inv, minlength=len(ud)); tot = np.bincount(inv, weights=ys, minlength=len(ud))
    pos = {dd: j for j, dd in enumerate(ud)}
    gday = np.array([pos.get(dd, -1) for dd in udays])
    bs = []
    for ds in boot_day_sets:
        sel = gday[ds]; sel = sel[sel >= 0]
        if len(sel): bs.append(tot[sel].sum() / max(cnt[sel].sum(), 1))
    lo, hi = (float(np.quantile(bs, .025)), float(np.quantile(bs, .975))) if bs else (np.nan, np.nan)
    return r, lo, hi, len(idx)

rows = []
# ---- (a) 等数十分位 (各方法自己的 pred 分箱); 点估计全箱, 日簇 CI 补尾箱 8-10 ----
for name, P in [(M0, P_RAW), (M5, P_CAL)]:
    flat_p = P.ravel(); flat_y = Y.ravel()
    qs = np.quantile(flat_p, np.linspace(0, 1, 11))
    qs[0] -= 1e-12; qs[-1] += 1e-12
    b = np.clip(np.searchsorted(qs, flat_p, side='right') - 1, 0, 9).reshape(Y.shape)
    for k in range(10):
        m = (b == k)
        rec = dict(method=name, part='decile', key=str(k+1), n=int(m.sum()),
                   pred_mean=float(flat_p[m.ravel()].mean()), obs_rate=float(flat_y[m.ravel()].mean()),
                   ci_lo=np.nan, ci_hi=np.nan)
        if k >= 7:   # 尾部三箱补日簇 bootstrap CI
            r, lo, hi, _ = rate_ci(m.ravel())
            rec['ci_lo'], rec['ci_hi'] = lo, hi
        rows.append(rec)

# ---- (b) 通信阈值集合 {pred ≥ τ} ----
for name, P in [(M0, P_RAW), (M5, P_CAL)]:
    for tau in [0.05, 0.10, 0.13, 0.20]:
        r, lo, hi, n = rate_ci((P >= tau).ravel())
        rows.append(dict(method=name, part='thresh', key=f'tau{tau}', n=n,
                         pred_mean=float(P.ravel()[(P >= tau).ravel()].mean()) if n else np.nan,
                         obs_rate=r, ci_lo=lo, ci_hi=hi))
# ---- (c) 尾部 ECE: pred≥0.05 区域内 5 等数箱加权 |pred−obs| ----
for name, P in [(M0, P_RAW), (M5, P_CAL)]:
    flat_p = P.ravel(); flat_y = Y.ravel()
    m = flat_p >= 0.05
    if m.sum() > 50:
        qs = np.quantile(flat_p[m], np.linspace(0, 1, 6)); qs[0] -= 1e-12; qs[-1] += 1e-12
        b = np.clip(np.searchsorted(qs, flat_p[m], side='right') - 1, 0, 4)
        ece = float(sum(((b == k).sum()/m.sum()) * abs(flat_p[m][b == k].mean() - flat_y[m][b == k].mean())
                        for k in range(5) if (b == k).sum() > 0))
    else:
        ece = np.nan
    # 全体 ECE (10 等数箱) 参照
    qs = np.quantile(flat_p, np.linspace(0, 1, 11)); qs[0] -= 1e-12; qs[-1] += 1e-12
    ball = np.clip(np.searchsorted(qs, flat_p, side='right') - 1, 0, 9)
    ece_all = float(sum(((ball == k).sum()/flat_p.size) * abs(flat_p[ball == k].mean() - flat_y[ball == k].mean()) for k in range(10)))
    rows.append(dict(method=name, part='ece', key='tail5_ge0.05', n=int(m.sum()), pred_mean=np.nan,
                     obs_rate=ece, ci_lo=np.nan, ci_hi=np.nan))
    rows.append(dict(method=name, part='ece', key='all10', n=int(flat_p.size), pred_mean=np.nan,
                     obs_rate=ece_all, ci_lo=np.nan, ci_hi=np.nan))

df = pd.DataFrame(rows)
df.to_csv(f'{OUT}/tail_calib_real.csv', index=False)
pd.set_option('display.width', 200)
print(df.to_string(index=False))
print('SAVED', f'{OUT}/tail_calib_real.csv')
