#  (): 伤情通道事后校准 (joint3_zi 的 π_i/μ_i), M0/M2(platt-on-gate)/M3(mu-power)/M5(per-horizon all+α refit)
# 纪律: 全部参数只在 val2024 拟合, test2025 终判; ECE=等计数十分位; NLL vs EB inj 地板 0.05448.
import numpy as np, pandas as pd, os, json
from scipy.stats import nbinom
from scipy.optimize import minimize, minimize_scalar
WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
R = f'{WS}/data/model_inputs'; JT = f'{WS}/experiments/results/joint'
OUT = f'{WS}/experiments/results/settf'
IN_LEN = OUT_LEN = 12; H = 12
FLOOR_I = 0.05448
jr = pd.read_csv(f'{JT}/joint_results.csv')
ai = float(jr[(jr.arm == 'joint3_zi') & (jr.window == 'test2025')].alpha_i.iloc[0])
def load(sp):
    d = np.load(f'{JT}/mu_joint3_zi_{sp}.npz')
    mu_c = d['mu_c'].astype(np.float64); pic = np.clip(d['pi_c'].astype(np.float64), 1e-7, 1 - 1e-7)
    r = d['r'].astype(np.float64); pii = np.clip(d['pi_i'].astype(np.float64), 1e-7, 1 - 1e-7)
    Ec = (1 - pic) * mu_c; mu_i = r * Ec + 1e-9
    y = np.transpose(np.load(f'{R}/crash_hour_inj/{sp}.npz')['y'][:, :, :, 0], (0, 2, 1)).astype(np.float64)
    return mu_i, pii, y
def zinb_nll(y, mu, pi, a):
    n = 1 / a; mu = np.maximum(mu, 1e-6)
    lnb = nbinom.logpmf(y, n, n / (n + mu))
    return -np.logaddexp(np.log(pi) + np.where(y == 0, 0.0, -30.0), np.log1p(-pi) + lnb)
def pocc(mu, pi, a):
    n = 1 / a
    return 1 - (pi + (1 - pi) * (n / (n + np.maximum(mu, 1e-6))) ** n)
def ece(p, y, B=10):
    p = p.ravel(); yb = (y.ravel() > 0).astype(float)
    o = np.argsort(p); q = np.array_split(o, B)
    e = 0.0
    for idx in q:
        e += abs(p[idx].mean() - yb[idx].mean()) * len(idx)
    return e / len(p)
mu_v, pi_v, y_v = load('val'); mu_t, pi_t, y_t = load('test')
logit_v = np.log(pi_v) - np.log1p(-pi_v); logit_t = np.log(pi_t) - np.log1p(-pi_t)
hor_v = np.arange(H)[None, None, :]; hor_t = hor_v
results = {}
def fit_gate_platt(mu, pi, y, a, per_horizon):
    """Fit the gate transform pi' = sigmoid(a*logit(pi)+b) by ZINB NLL.

    The target is the zero-inflation gate, not the occurrence probability;
    fitting Platt on P(y>0) and reusing it as the gate is a mis-specification.
    """
    lg = np.log(pi) - np.log1p(-pi)
    if per_horizon:
        out = {}
        for h in range(H):
            muh = np.maximum(mu[:, :, h], 1e-6); yh = y[:, :, h]
            lgh = lg[:, :, h]
            def nll(w):
                p2 = 1 / (1 + np.exp(-(w[0] * lgh + w[1])))
                return zinb_nll(yh, muh, np.clip(p2, 1e-7, 1 - 1e-7), a).mean()
            out[h] = minimize(nll, [1.0, 0.0], method='BFGS').x
        return out
    def nll(w):
        p2 = 1 / (1 + np.exp(-(w[0] * lg + w[1])))
        return zinb_nll(y, mu, np.clip(p2, 1e-7, 1 - 1e-7), a).mean()
    return minimize(nll, [1.0, 0.0], method='BFGS').x
def apply_platt(logit, w):
    if isinstance(w, dict):
        return np.stack([1 / (1 + np.exp(-(w[h][0] * logit[:, :, h] + w[h][1]))) for h in range(H)], axis=2)
    return 1 / (1 + np.exp(-(w[0] * logit + w[1])))
def fit_alpha(y, mu, pi):
    f = minimize_scalar(lambda la: zinb_nll(y, mu, pi, np.exp(la)).mean(), bounds=(-6, 3), method='bounded')
    return float(np.exp(f.x))
def fit_power(y, mu, pi, a):
    f = minimize_scalar(lambda g: zinb_nll(y, mu ** g, pi, a).mean(), bounds=(0.5, 1.5), method='bounded')
    return float(f.x)
# M0
results['M0_uncal'] = dict(alpha=ai, pi=pi_t, mu=mu_t)
# M2 global platt
w2 = fit_gate_platt(mu_v, pi_v, y_v, ai, False); results['M2_platt'] = dict(alpha=ai, pi=apply_platt(logit_t, w2), mu=mu_t)
# M3 mu power
a3 = fit_alpha(y_v, mu_v, pi_v); g3 = fit_power(y_v, mu_v, pi_v, a3)
results['M3_mupow_a'] = dict(alpha=a3, pi=pi_t, mu=mu_t ** g3)
# M5 per-horizon platt + power + alpha
w5 = fit_gate_platt(mu_v, pi_v, y_v, ai, True); a5 = fit_alpha(y_v, mu_v, apply_platt(logit_v, w5))
g5 = fit_power(y_v, mu_v ** 1.0, apply_platt(logit_v, w5), a5)
results['M5_perhor_a'] = dict(alpha=a5, pi=apply_platt(logit_t, w5), mu=mu_t ** g5)
vres = {'M0_uncal': dict(alpha=ai, pi=pi_v, mu=mu_v),
        'M2_platt': dict(alpha=ai, pi=apply_platt(logit_v, w2), mu=mu_v),
        'M3_mupow_a': dict(alpha=a3, pi=pi_v, mu=mu_v ** g3),
        'M5_perhor_a': dict(alpha=a5, pi=apply_platt(logit_v, w5), mu=mu_v ** g5)}
rows = []
for window, yx, src in [('val', y_v, vres), ('test', y_t, results)]:
    for name, cfgd in src.items():
        nll = float(zinb_nll(yx, cfgd['mu'], cfgd['pi'], cfgd['alpha']).mean())
        e = float(ece(pocc(cfgd['mu'], cfgd['pi'], cfgd['alpha']), yx))
        rows.append(dict(method=name, window=window, NLL=round(nll, 5),
                         dNLL_pct=round(100 * (nll - FLOOR_I) / FLOOR_I, 3),
                         ECE=round(e, 5), alpha=round(cfgd['alpha'], 5)))
D = pd.DataFrame(rows)
D.to_csv(f'{OUT}/calibration_results_injury.csv', index=False)
print(D.to_string(index=False))
print('val ECE M0/M5:', round(ece(pocc(mu_v, pi_v, ai), y_v), 5), round(ece(pocc(mu_v ** g5, apply_platt(logit_v, w5), a5), y_v), 5))
