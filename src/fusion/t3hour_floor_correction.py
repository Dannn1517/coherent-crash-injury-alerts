# Data build 审计修正(): 地板口径修正重算 (零GPU, 全部用持久化 mu/pi/ckpt)
# 背景(internal audit): 气候学地板 EB 收缩的 g_tr 单位 bug——代码 C[:,tr_h].sum()/len(tr_h)
#   = 全细胞小时总量(~10.78) 而非每细胞均值(~0.0269), 伪计数 0.5/g_tr=0.046(≈无收缩) 而非 18.6(真EB)
#   → legacy 地板系统性偏弱, 头条 dNLL% 被夸大 (crash ~0.64pp / inj ~1.4pp); 同配方三脚本同源
#   (t3hour_baselines/t3hour_stllm_transfer/t3hour_settf), 臂间对比不受影响(同地板)
# 本脚本: 4 地板口径 (legacy/EB修正 × α矩匹配/αNLL最优, α均只用train窗) 重算全部 headline 臂:
#   NLL + dNLL% + 窗簇bootstrap(rng7×500) + 日簇bootstrap(×500) + 地板AUC
#   + 修正 ESR5 (真(window,h)切片 + 双侧随机tiebreak R=5 MC) + Recall@20 (|top20∩pos|/min(20,n_pos))
# 主修正口径 = EB_mom (只修 clim bug, α 协议与 legacy 相同); EB_NLLopt 为最强地板压力口径
# 输出: experiments/results/settf/floor_correction.csv + esr5_correction.csv
import numpy as np, pandas as pd, os, json, time
import torch
from scipy.stats import nbinom
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
R = f'{WS}/data/model_inputs'
OUT = f'{WS}/experiments/results/settf'
TR = f'{WS}/experiments/results/stllm_transfer'
IN_LEN, OUT_LEN, STRIDE = 12, 12, 3
B_TRAIN_END, B_VAL_END, WARM = 24 * 1095, 24 * 1461, 168
R_TIE = 5            # tiebreak MC 次数
K = 20               # top-k

t3 = np.load(f'{WS}/data/processed/T3_region_hour/labels.npz')
n_hours = t3['region_hour'].shape[1]
how = np.arange(n_hours) % 168
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
N_REG = 400

def clim_of(C, g):
    cl = np.zeros((N_REG, 168))
    for w in range(168):
        m = how[tr_h] == w
        cl[:, w] = (C[:, tr_h][:, m].sum(1) + 0.5) / (m.sum() + 0.5 / max(g, 1e-9))
    return cl

def floor_mu(cl, sp, S):
    tt = starts[sp][:S, None] + IN_LEN + np.arange(OUT_LEN)[None, :]
    return cl[:, how[tt.ravel()]].T.reshape(S, OUT_LEN, N_REG).transpose(0, 2, 1)

def nb_nll(y, mu, a):
    n = 1.0 / max(a, 1e-8)
    mu = np.maximum(mu, 1e-6)
    return float(-nbinom.logpmf(y, n, n / (n + mu)).mean())

def zinb_nll(y, mu, pi, a):
    n = 1.0 / max(a, 1e-8)
    mu = np.maximum(mu, 1e-6)
    lnb = nbinom.logpmf(y, n, n / (n + mu))
    if pi is None:
        return float(-lnb.mean())
    pic = np.clip(pi, 1e-7, 1 - 1e-7)
    return float(-np.logaddexp(np.log(pic) + np.where(y == 0, 0.0, -30.0),
                               np.log1p(-pic) + lnb).mean())

def pts_of(y, mu, pi, a):
    n = 1.0 / max(a, 1e-8)
    mu = np.maximum(mu, 1e-6)
    lnb = nbinom.logpmf(y, n, n / (n + mu))
    if pi is None:
        return -lnb
    pic = np.clip(pi, 1e-7, 1 - 1e-7)
    return -np.logaddexp(np.log(pic) + np.where(y == 0, 0.0, -30.0), np.log1p(-pic) + lnb)

def mom_alpha(y, mu):
    raw = (((y - mu) ** 2 - mu).mean()) / max((mu ** 2).mean(), 1e-12)
    return float(max(raw, 1e-4)), float(raw)

def nll_alpha(y, mu):
    r = minimize_scalar(lambda la: nb_nll(y, mu, float(np.exp(la))), bounds=(-25, 3), method='bounded')
    return float(np.exp(r.x))

def boot(d_w, days, rng_seed):
    """窗簇 (rng7×500, 与 eval_arm 同) + 日簇 (窗按起点//24 聚日, 日均值重采样×500)"""
    rb = np.random.default_rng(rng_seed)
    S = len(d_w)
    bs = d_w[rb.integers(0, S, size=(500, S))].mean(1)
    win = dict(ci_lo=float(np.quantile(bs, .025)), ci_hi=float(np.quantile(bs, .975)))
    ud = np.unique(days)
    dmean = np.array([d_w[days == d].mean() for d in ud])
    rb2 = np.random.default_rng(rng_seed + 1)
    bs2 = dmean[rb2.integers(0, len(ud), size=(500, len(ud)))].mean(1)
    day = dict(ci_lo=float(np.quantile(bs2, .025)), ci_hi=float(np.quantile(bs2, .975)))
    return win, day

def esr5_mc(score, y, rng_seed=11):
    """修正 ESR5: 真(window,h)切片(S,H,N) + 双侧随机 tiebreak R_TIE 次 MC 均值; 附 Recall@20"""
    sc = np.transpose(score, (0, 2, 1)).reshape(-1, N_REG)     # (S*12, N)
    yy = np.transpose(y, (0, 2, 1)).reshape(-1, N_REG)
    rng = np.random.default_rng(rng_seed)
    esrs, recs = [], []
    pos = yy > 0
    npos = pos.sum(1)
    haspos = npos > 0
    rows = sc.shape[0]
    ridx = np.arange(rows)
    for r in range(R_TIE):
        perm_s = rng.permutation(N_REG); perm_y = rng.permutation(N_REG)
        # 列置换后 stable argsort = 均匀随机 tiebreak; perm_s[ps] 还原原列号
        top_s = perm_s[np.argsort(-sc[:, perm_s], axis=1, kind='stable')[:, :K]]
        top_y = perm_y[np.argsort(-yy[:, perm_y], axis=1, kind='stable')[:, :K]]
        oh_s = np.zeros((rows, N_REG), bool); oh_s[ridx[:, None], top_s] = True
        oh_y = np.zeros((rows, N_REG), bool); oh_y[ridx[:, None], top_y] = True
        esrs.append(float((oh_s & oh_y).sum(1).mean()) / K)
        cap = (oh_s[haspos] & pos[haspos]).sum(1)
        recs.append(float((cap / np.minimum(K, npos[haspos])).mean()))
    return round(float(np.mean(esrs)), 4), round(float(np.mean(recs)), 4)

# ---- 通道与臂注册 ----
CH = {'crash': dict(label='region_hour', dirs={'settf': f'{R}/crash_hour', 'inj': None}),
      'inj': dict(label='region_hour_injured', dirs={'settf': f'{R}/crash_hour_inj'})}
def mu_npz(ch, arm, sp):
    """返回 (mu, pi) 或 None"""
    if arm.startswith('settf'):
        f = f'{OUT}/mu_{arm}_{sp}.npz'
    elif arm.startswith('cal_'):
        f = f'{OUT}/mu_settf_zinb_{sp}.npz'
    else:
        f = f'{TR}/mu_{arm}_{sp}.npz'
    if not os.path.exists(f):
        return None
    d = np.load(f)
    mu = d['mu'].astype(np.float64); pi = d['pi'].astype(np.float64) if 'pi' in d.files else None
    if arm.startswith('cal_'):
        prm_d = json.load(open(f'{OUT}/calibration_params.json'))
        key = arm[4:]
        p = {kk: (np.array(vv).reshape(1, 1, OUT_LEN) if isinstance(vv, list) else float(vv))
             for kk, vv in prm_d['params'][key].items()}
        lp = np.log(np.clip(pi, 1e-7, 1 - 1e-7)) - np.log1p(-np.clip(pi, 1e-7, 1 - 1e-7))
        if 'a' in p:
            lp = p['a'] * lp + p['b']
        pi = 1.0 / (1.0 + np.exp(-lp))
        if 'c' in p:
            mu = p['c'] * np.maximum(mu, 1e-6) ** p['g']
    return mu, pi

def alpha_of(arm, ch):
    if arm.startswith('cal_'):
        d = json.load(open(f'{OUT}/calibration_params.json'))
        return float(d['alphas'][arm[4:]])
    if arm.startswith('settf'):
        suf = arm[len('settf_zinb'):] if arm.startswith('settf_zinb') else arm[len('settf_nb'):]
        head = 'zinb' if arm.startswith('settf_zinb') else 'nb'
        ck = torch.load(f'{OUT}/ckpt_settf_{head}{suf}.pt', map_location='cpu', weights_only=True)
    else:
        ck = torch.load(f'{TR}/ckpt_{arm}.pt', map_location='cpu', weights_only=True)
    return float(np.exp(ck['la']))

ARMS = {'crash': ['settf_zinb', 'settf_zinb_s1234', 'settf_zinb_s2025', 'settf_zinb_s777', 'settf_nb',
                  'nb_head', 'zinb_head', 'nb_head_nopre', 'cal_M5_perhor_a', 'cal_M3_mupow_a'],
        'inj': ['settf_zinb_inj']}

t00 = time.time()
rows, esr_rows = [], []
for ch, cfgd in CH.items():
    C = t3[cfgd['label']].astype(np.float64)
    g_code = C[:, tr_h].sum() / len(tr_h)
    g_cell = C[:, tr_h].mean()
    cl_leg = clim_of(C, g_code); cl_eb = clim_of(C, g_cell)
    # 地板 α (只用 train)
    y_tr = np.transpose(np.load(f'{cfgd["dirs"]["settf"]}/train.npz')['y'][:, :, :, 0], (0, 2, 1)).astype(np.float64)
    mf_leg_tr = floor_mu(cl_leg, 'train', len(y_tr)); mf_eb_tr = floor_mu(cl_eb, 'train', len(y_tr))
    a_leg_m, raw_leg = mom_alpha(y_tr, mf_leg_tr)
    a_eb_m, raw_eb = mom_alpha(y_tr, mf_eb_tr)
    a_leg_s = nll_alpha(y_tr, mf_leg_tr); a_eb_s = nll_alpha(y_tr, mf_eb_tr)
    floors = {'legacy_mom': (cl_leg, a_leg_m), 'legacy_NLLopt': (cl_leg, a_leg_s),
              'EB_mom': (cl_eb, a_eb_m), 'EB_NLLopt': (cl_eb, a_eb_s)}
    print(f'[{ch}] g_code={g_code:.4f}(伪计数{0.5 / g_code:.3f}) g_cell={g_cell:.5f}(伪计数{0.5 / g_cell:.1f}) '
          f'α: leg_mom={a_leg_m:.5f}(未钳位矩估计{raw_leg:+.4f}) leg_NLLopt={a_leg_s:.5f} '
          f'EB_mom={a_eb_m:.5f}(raw{raw_eb:+.4f}) EB_NLLopt={a_eb_s:.5f}', flush=True)
    for sp in ['val', 'test']:
        y = np.transpose(np.load(f'{cfgd["dirs"]["settf"]}/{sp}.npz')['y'][:, :, :, 0], (0, 2, 1)).astype(np.float64)
        S = y.shape[0]
        days = starts[sp][:S] // 24
        fl_mu = {k: floor_mu(v[0], sp, S) for k, v in floors.items()}
        fl_nll = {k: nb_nll(y, fl_mu[k], floors[k][1]) for k in floors}
        for k, v in fl_nll.items():
            print(f'  [{ch}/{sp}] floor {k}: NLL={v:.5f}', flush=True)
        # 地板 AUC (clim 两种; 与 α 无关)
        for tag, cl in [('legacy', cl_leg), ('EB', cl_eb)]:
            muf = fl_mu['EB_mom' if tag == 'EB' else 'legacy_mom']
            auc_f = roc_auc_score((y.ravel() > 0).astype(int), (1 - np.exp(-np.maximum(muf, 1e-6))).ravel())
            esr_f, rec_f = esr5_mc(muf, y)
            esr_rows.append(dict(channel=ch, window=sp, scorer=f'floor_{tag}', ESR5_mc=esr_f, Recall20=rec_f,
                                 AUC=round(float(auc_f), 5)))
        for arm in ARMS[ch]:
            got = mu_npz(ch, arm, sp)
            if got is None:
                print(f'  [{ch}/{sp}] {arm}: mu npz 缺失, 跳过'); continue
            mu, pi = got
            mu = mu[:S]; pi = pi[:S] if pi is not None else None
            a_m = alpha_of(arm, ch)
            m_nll = zinb_nll(y, mu, pi, a_m)
            pts_m = pts_of(y, mu, pi, a_m)
            d_w_all = {k: (pts_m - pts_of(y, fl_mu[k], None, floors[k][1])).reshape(S, -1).mean(1) for k in floors}
            for k in floors:
                d_w = d_w_all[k]
                win, day = boot(d_w, days, 7)
                rows.append(dict(channel=ch, arm=arm, window=f'{sp}2024' if sp == 'val' else f'{sp}2025',
                                 floor=k, alpha_model=round(a_m, 5), alpha_floor=round(floors[k][1], 6),
                                 model_NLL=round(m_nll, 5), floor_NLL=round(fl_nll[k], 5),
                                 dNLL=round(float(d_w.mean()), 6),
                                 dNLL_pct=round(100 * float(d_w.mean()) / fl_nll[k], 3),
                                 win_ci_lo=round(win['ci_lo'], 6), win_ci_hi=round(win['ci_hi'], 6),
                                 win_sig=bool(win['ci_hi'] < 0), win_sig_worse=bool(win['ci_lo'] > 0),
                                 day_ci_lo=round(day['ci_lo'], 6), day_ci_hi=round(day['ci_hi'], 6),
                                 day_sig=bool(day['ci_hi'] < 0), day_sig_worse=bool(day['ci_lo'] > 0)))
            print(f'  [{ch}/{sp}] {arm}: NLL={m_nll:.5f} dNLL% leg_mom={rows[-4]["dNLL_pct"]:+.3f} '
                  f'EB_mom={rows[-2]["dNLL_pct"]:+.3f} EB_NLLopt={rows[-1]["dNLL_pct"]:+.3f} '
                  f'(win_sig EB_mom={rows[-2]["win_sig"]}, day_sig={rows[-2]["day_sig"]})', flush=True)
            if arm in ('settf_zinb', 'settf_zinb_inj', 'settf_zinb_kil', 'cal_M5_perhor_a') or \
               (ch == 'crash' and arm in ('settf_nb',)):
                pic = np.clip(pi, 1e-7, 1 - 1e-7) if pi is not None else 0.0
                p_occ = (1 - pic) * (1 - np.exp(-np.maximum(mu, 1e-6))) if pi is not None \
                    else 1 - np.exp(-np.maximum(mu, 1e-6))
                esr_m, rec_m = esr5_mc(p_occ, y)
                esr_rows.append(dict(channel=ch, window=sp, scorer=arm, ESR5_mc=esr_m, Recall20=rec_m,
                                     AUC=round(float(roc_auc_score((y.ravel() > 0).astype(int), p_occ.ravel())), 5)))
                print(f'      ESR5_mc={esr_m} Recall20={rec_m}', flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/floor_correction.csv', index=False)
pd.DataFrame(esr_rows).to_csv(f'{OUT}/esr5_correction.csv', index=False)
print(f'DONE -> {OUT}/floor_correction.csv + esr5_correction.csv | {time.time() - t00:.0f}s')
