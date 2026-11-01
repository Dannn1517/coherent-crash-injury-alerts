# JointSetTF: frequency-severity model for crash and injury counts.
# A shared set-transformer encoder feeds (i) a ZINB crash head and (ii) an injury head whose mean
# is r * E[crash], with a zero-initialized severity-rate residual, so injury-without-crash has zero
# probability by construction. Compound NB/ZTNB variants replace the injury head with a collective
# risk mixture; the ZTNB variant makes P(injury=0)=P(crash=0) exact.
# Protocol: locked test windows, empirical-Bayes floor, four seeds, window- and day-clustered bootstraps.

import numpy as np, pandas as pd, os, sys, json, time, warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.nn.functional as TF
from scipy.stats import nbinom
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
R = f'{WS}/data/model_inputs'
OUT = f'{WS}/experiments/results/joint'
os.makedirs(OUT, exist_ok=True)
DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH = 64
EPOCHS, PATIENCE = 250, 30
MIN_ES = int(os.environ.get('JOINT_MIN_ES', '40'))   # : =0 使带 EVAL_TAG 的重评走纯 RESUME-EVAL (同权重换输入)
LR = 1e-3
SEED = int(os.environ.get('JOINT_SEED', '6666'))
# 双城参数化 (8.1 §8-1): JOINT_T3/JOINT_DATA_C/JOINT_DATA_I/JOINT_TAG, 默认 NYC 行为不变
T3_PATH = os.environ.get('JOINT_T3', f'{WS}/data/processed/T3_region_hour/labels.npz')
DATA_C = os.environ.get('JOINT_DATA_C', f'{R}/crash_hour')
DATA_I = os.environ.get('JOINT_DATA_I', f'{R}/crash_hour_inj')
TAG = os.environ.get('JOINT_TAG', '')
WX = os.environ.get('JOINT_WX') == '1'   # 天气6通道侧车(D38硬伤#7直接证据臂): temp/precip/snow/wind/pany/code
COUPLE = os.environ.get('JOINT_COUPLE') == '1'
ES_MARG = os.environ.get('JOINT_ES_MARG') == '1'  # cmp 臂用 val 边际混合 NLL 做 ES (方法学审查 M1)  # 支撑约束: π_i←1−(1−π_i)(1−π_c), 使 P(y_i>0|y_c=0)=0
WX_PATH = os.environ.get('JOINT_WX_PATH', f'{WS}/data/processed/T3_gsnet/weather_hour.npy')
SUF = f'{TAG}' + ('' if SEED == 6666 else f'_s{SEED}')
EVAL_TAG = os.environ.get('JOINT_EVAL_TAG', '')
# 窗口年份标签可 env 覆盖 ( 台中: val=2023/test=2024); 默认 NYC/CHI 不变
VAL_YEAR = os.environ.get('JOINT_VAL_YEAR', '2024')
TEST_YEAR = os.environ.get('JOINT_TEST_YEAR', '2025')
D_MODEL, N_LAYER, N_HEAD, D_FF = 128, 4, 8, 256
IN_LEN, OUT_LEN, STRIDE = 12, 12, 3
# 划分边界可 env 覆盖 ( 台中四年轴: JOINT_B_TRAIN_END=17520/JOINT_B_VAL_END=26280); 默认五年轴不变
B_TRAIN_END = int(os.environ.get('JOINT_B_TRAIN_END', 24 * 1095))
B_VAL_END = int(os.environ.get('JOINT_B_VAL_END', 24 * 1461))
WARM = 168
A_STATIC = 300.0      # r_static EB 先验强度 (baseline val 网格选定)
ROLL = 37             # 负对照空间错位

t3 = np.load(T3_PATH)
YC_L = t3['region_hour'].astype(np.float64)
YI_L = t3['region_hour_injured'].astype(np.float64)
n_reg, n_hours = YC_L.shape
how = np.arange(n_hours) % 168
starts = {'train': [], 'val': [], 'test': []}
for i in range(0, n_hours - 23, STRIDE):
    o = i + 24
    if o <= B_TRAIN_END: starts['train'].append(i)
    elif i >= B_TRAIN_END and o <= B_VAL_END: starts['val'].append(i)
    elif i >= B_VAL_END: starts['test'].append(i)
starts = {k: np.array(v) for k, v in starts.items()}
tr_h = np.arange(WARM, B_TRAIN_END)

# ---- r_static: train 小时 EB severity 率 (与 baseline 同式) ----
r0 = float(YI_L[:, tr_h].sum() / YC_L[:, tr_h].sum())
R_STATIC = ((YI_L[:, tr_h].sum(1) + A_STATIC * r0) / (YC_L[:, tr_h].sum(1) + A_STATIC)).astype(np.float32)

# ---- EB 修正地板 (per-cell g_tr, internal audit 修正口径) ----
def clim_eb(C):
    g = C[:, tr_h].mean()
    cl = np.zeros((n_reg, 168))
    for w in range(168):
        m = how[tr_h] == w
        cl[:, w] = (C[:, tr_h][:, m].sum(1) + 0.5) / (m.sum() + 0.5 / max(g, 1e-9))
    return cl
def floor_mu(cl, sp, S):
    tt = starts[sp][:S, None] + IN_LEN + np.arange(OUT_LEN)[None, :]
    return cl[:, how[tt.ravel()]].T.reshape(S, OUT_LEN, n_reg).transpose(0, 2, 1)
def fit_alpha(y, mu):
    mu = np.clip(mu.astype(np.float64), 1e-6, None); y = y.astype(np.float64)
    return float(max((((y - mu) ** 2 - mu).mean()) / max((mu ** 2).mean(), 1e-12), 1e-4))
CLIM_C, CLIM_I = clim_eb(YC_L), clim_eb(YI_L)
y_tr_c = np.transpose(np.load(f'{DATA_C}/train.npz')['y'][:, :, :, 0], (0, 2, 1))
y_tr_i = np.transpose(np.load(f'{DATA_I}/train.npz')['y'][:, :, :, 0], (0, 2, 1))
A_FC = fit_alpha(y_tr_c, floor_mu(CLIM_C, 'train', len(y_tr_c)))
A_FI = fit_alpha(y_tr_i, floor_mu(CLIM_I, 'train', len(y_tr_i)))

# ---- 数据: crash 3ch + inj 历史第4通道(joint4) ----
_raw = {}
for cat in ['train', 'val', 'test']:
    dc = np.load(f'{DATA_C}/{cat}.npz')
    di = np.load(f'{DATA_I}/{cat}.npz')
    xc = dc['x'].astype(np.float32)                       # (S,12,400,3)
    xi0 = di['x'][..., 0:1].astype(np.float32)            # inj 历史计数通道
    _raw['x_' + cat] = xc
    _raw['xi_' + cat] = xi0
    _raw['yc_' + cat] = dc['y'].astype(np.float32)
    _raw['yi_' + cat] = di['y'].astype(np.float32)
MEAN = _raw['x_train'][..., 0].mean(); STD = _raw['x_train'][..., 0].std()
MEAN_I = _raw['xi_train'][..., 0].mean(); STD_I = _raw['xi_train'][..., 0].std()
for cat in ['train', 'val', 'test']:
    _raw['x_' + cat][..., 0] = (_raw['x_' + cat][..., 0] - MEAN) / STD
    _raw['xi_' + cat][..., 0] = (_raw['xi_' + cat][..., 0] - MEAN_I) / STD_I
SHIFT = int(os.environ.get('JOINT_SHIFT_H', '0'))   # : 输入上报延迟模拟 (val/test 历史整体前移 SHIFT 小时, train 不变; 标准化常数仍用未移 train)
if SHIFT > 0:
    for cat in ['val', 'test']:
        ss = np.asarray(starts[cat])
        hc = np.stack([YC_L[:, q-SHIFT:q-SHIFT+IN_LEN].T for q in ss])          # (S,12,400)
        hi = np.stack([YI_L[:, q-SHIFT:q-SHIFT+IN_LEN].T for q in ss])
        hh = ss[:, None] - SHIFT + np.arange(IN_LEN)[None, :]
        _raw['x_' + cat][..., 0] = (hc - MEAN) / STD
        _raw['x_' + cat][..., 1] = (((hh % 24) + 0.5) / 24.0)[:, :, None]
        _raw['x_' + cat][..., 2] = ((((hh // 24) + 4) % 7).astype(np.float32))[:, :, None]
        _raw['xi_' + cat][..., 0] = (hi - MEAN_I) / STD_I
    print(f'[shift] JOINT_SHIFT_H={SHIFT} applied to val/test inputs', flush=True)
if WX:   # 天气侧车: train 小时统计标准化, 按 starts 切窗 (S,12,6)
    w_all = np.load(WX_PATH).astype(np.float32)
    wm = w_all[:B_TRAIN_END].mean(0); wsd = w_all[:B_TRAIN_END].std(0) + 1e-6
    w_all = (w_all - wm) / wsd
    for cat in ['train', 'val', 'test']:
        _raw['wx_' + cat] = np.stack([w_all[s:s + IN_LEN] for s in starts[cat]]).astype(np.float32)
YI_VAL = np.transpose(np.load(f'{DATA_I}/val.npz')['y'][:, :, :, 0], (0, 2, 1)).astype(np.float64)[:len(starts['val'])]
perm = np.random.default_rng(SEED).permutation(len(_raw['x_train']))
for k in (['x_train', 'xi_train', 'yc_train', 'yi_train'] + (['wx_train'] if WX else [])):
    _raw[k] = _raw[k][perm]

def batches(cat, bs=BATCH):
    n = len(_raw['x_' + cat])
    for i in range(0, n, bs):
        yield (_raw['x_' + cat][i:i + bs], _raw['xi_' + cat][i:i + bs],
               _raw['yc_' + cat][i:i + bs], _raw['yi_' + cat][i:i + bs],
               _raw['wx_' + cat][i:i + bs] if WX else None)

# ---- 模型 ----
class JointSetTF(nn.Module):
    def __init__(self, n_nodes, in_ch=3, zi=False, use_wx=False, cmp=False, cmp2=False, cmp3=False, cmpq=False, cmp5=False, cmp6=False):
        super().__init__()
        self.zi = zi
        self.use_wx = use_wx
        self.win_proj = nn.Linear(IN_LEN * (in_ch + (6 if use_wx else 0)), D_MODEL)
        self.node_emb = nn.Parameter(torch.empty(n_nodes, D_MODEL)); nn.init.xavier_uniform_(self.node_emb)
        self.hour_emb = nn.Parameter(torch.empty(24, D_MODEL)); nn.init.xavier_uniform_(self.hour_emb)
        self.dow_emb = nn.Parameter(torch.empty(7, D_MODEL)); nn.init.xavier_uniform_(self.dow_emb)
        enc = nn.TransformerEncoderLayer(D_MODEL, N_HEAD, dim_feedforward=D_FF,
                                         batch_first=True, dropout=0.1, activation='gelu')
        self.enc = nn.TransformerEncoder(enc, N_LAYER)
        self.ln = nn.LayerNorm(D_MODEL)
        self.out_mu = nn.Linear(D_MODEL, OUT_LEN)
        self.out_gate = nn.Linear(D_MODEL, OUT_LEN)
        self.out_rate = nn.Linear(D_MODEL, OUT_LEN)
        nn.init.zeros_(self.out_rate.weight); nn.init.zeros_(self.out_rate.bias)   # 起点=hybrid
        # Model 诊断: 边际 inj 模型的 NLL 优势主要来自其自有 ZI 门( D4b gate主导) → zi 臂补门
        self.out_gate_i = nn.Linear(D_MODEL, OUT_LEN) if zi else None
        # C1 (IDEAS_R2): compound 臂 — 每起事故伤情 ~ NB(n_s, p_s), y_i|y_c=k ~ NB(k·n_s, p_s) 闭式卷积
        self.cmp = cmp
        self.cmp2 = cmp2
        if cmp:
            self.log_alpha_s = nn.Parameter(torch.tensor(np.log(0.5), dtype=torch.float32))
        if cmp2:
            self.out_as = nn.Linear(D_MODEL, OUT_LEN)
        self.cmp3 = cmp3
        self.cmpq = cmpq
        if cmp3 and not cmpq:
            self.out_q = nn.Linear(D_MODEL, OUT_LEN)
        if cmpq:
            self.logit_q = nn.Parameter(torch.tensor(-0.5, dtype=torch.float32))
        self.cmp6 = cmp6
        self.cmp5 = cmp5
        if cmp5:
            self.log_alpha_s_cell = nn.Parameter(torch.full((n_nodes,), np.log(0.5), dtype=torch.float32))
        self.register_buffer('rs', torch.tensor(R_STATIC, dtype=torch.float32).view(1, -1, 1))

    def forward(self, x, xi=None, wx=None):
        B, T, N, F = x.shape
        h = x.permute(0, 2, 1, 3).reshape(B, N, T * F)
        if xi is not None:
            hi = xi.permute(0, 2, 1, 3).reshape(B, N, T * 1)
            h = torch.cat([h, hi], dim=-1)
        if wx is not None:   # (B,12,6) 空间恒定 → 广播到每节点
            hw = wx.reshape(B, T * 6).unsqueeze(1).expand(B, N, T * 6)
            h = torch.cat([h, hw], dim=-1)
        tok = self.win_proj(h) + self.node_emb.unsqueeze(0)
        last_tod = x[:, -1, 0, 1] * 24
        last_dow = x[:, -1, 0, 2]
        tok = tok + self.hour_emb[last_tod.long().clamp(0, 23)].unsqueeze(1) \
                  + self.dow_emb[last_dow.long().clamp(0, 6)].unsqueeze(1)
        z = self.ln(self.enc(tok))
        mu_raw = self.out_mu(z)
        gate = self.out_gate(z)
        r = (self.rs * torch.exp(self.out_rate(z).clamp(-5, 3))).clamp(1e-4, 10.0)
        gate_i = self.out_gate_i(z) if self.out_gate_i is not None else None
        as_raw = self.out_as(z) if self.cmp2 else None
        q_raw = self.out_q(z) if (self.cmp3 and not self.cmpq) else None
        return mu_raw, gate, r, gate_i, as_raw, q_raw

def nb_nll_var(y, n, p):
    """NB nll with per-point size n and prob p (compound severity head)."""
    p = p.clamp(1e-7, 1 - 1e-7)
    return -(torch.lgamma(n.clamp(min=1e-8) + y) - torch.lgamma(y + 1) - torch.lgamma(n.clamp(min=1e-8))
             + n * torch.log(p) + y * torch.log1p(-p))

def nb_nll_t(y, mu, la):
    n = torch.exp(-la)
    log_p = torch.log(n) - torch.log(n + mu)
    log_1mp = torch.log(mu) - torch.log(n + mu)
    return -(torch.lgamma(n + y) - torch.lgamma(y + 1) - torch.lgamma(n) + n * log_p + y * log_1mp)

def zinb_nll_t(y, mu, pi, la):
    n = torch.exp(-la)
    log_p = torch.log(n) - torch.log(n + mu)
    log_1mp = torch.log(mu) - torch.log(n + mu)
    lnb = torch.lgamma(n + y) - torch.lgamma(y + 1) - torch.lgamma(n) + n * log_p + y * log_1mp
    l0 = torch.log(pi.clamp(1e-7, 1 - 1e-7)) + torch.where(y == 0, torch.zeros_like(lnb), torch.full_like(lnb, -30.0))
    lm = torch.log((1 - pi).clamp(1e-7, 1 - 1e-7)) + lnb
    return -torch.logaddexp(l0, lm)

def to_counts(mu_raw):
    return TF.softplus(mu_raw * STD + MEAN) + 1e-4

# ---- 臂配置 ----
ARM_CFG = {
    'joint3':        dict(in_ch=3, detach=False, yroll=False, zi=False),
    'joint4':        dict(in_ch=4, detach=False, yroll=False, zi=False),
    'joint3_detach': dict(in_ch=3, detach=True,  yroll=False, zi=False),
    'joint3_yroll':  dict(in_ch=3, detach=False, yroll=True,  zi=False),
    'joint3_zi':     dict(in_ch=3, detach=False, yroll=False, zi=True),
    'joint4_zi':     dict(in_ch=4, detach=False, yroll=False, zi=True),
    'joint3_cmp':    dict(in_ch=3, detach=False, yroll=False, zi=False, cmp=True),
    # P0-3 (internal review 2026-11): 条件严重度改 zero-truncated NB — y_i|y_c=k ~ ZTNB(k n_s, p_s), k≥1
    # 目的: 把 Taichung +4.25pp 精确性代价分解为 [条件族失配] vs [支撑约束本身]; 其余模块与 joint3_cmp 全同
    'joint3_cmpzt':  dict(in_ch=3, detach=False, yroll=False, zi=False, cmp=True, cmpzt=True),
    'joint3_cmp2':   dict(in_ch=3, detach=False, yroll=False, zi=False, cmp=True, cmp2=True),
    'joint3_cmp3':   dict(in_ch=3, detach=False, yroll=False, zi=False, cmp=True, cmp2=True, cmp3=True),
    'joint3_cmp4':   dict(in_ch=3, detach=False, yroll=False, zi=False, cmp=True, cmp3=True, cmpq=True),
    'joint3_cmp5':   dict(in_ch=3, detach=False, yroll=False, zi=False, cmp=True, cmp5=True),
    'joint3_cmp6':   dict(in_ch=3, detach=False, yroll=False, zi=False, cmp=True, cmp6=True),
}

_LA_C = 0.0
def run_arm(arm):
    cfg = ARM_CFG[arm]
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = JointSetTF(n_reg, cfg['in_ch'], zi=cfg['zi'], use_wx=WX, cmp=cfg.get('cmp', False), cmp2=cfg.get('cmp2', False), cmp3=cfg.get('cmp3', False), cmpq=cfg.get('cmpq', False), cmp5=cfg.get('cmp5', False), cmp6=cfg.get('cmp6', False)).to(DEV)
    la_c = torch.zeros(1, requires_grad=True, device=DEV)
    la_i = torch.zeros(1, requires_grad=True, device=DEV)
    params = list(model.parameters()) + [la_c, la_i]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    n_par = sum(p.numel() for p in model.parameters())
    print(f'[{arm}] params={n_par} r_static: mean={R_STATIC.mean():.4f} max={R_STATIC.max():.3f}', flush=True)
    ck_path = f'{OUT}/ckpt_{arm}{SUF}.pt'
    start_ep, best_val, best_state, patience = 0, 1e18, None, 0
    best_la = (0.0, 0.0)
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=DEV)
        model.load_state_dict(ck['state'])
        if ck.get('ep', -1) >= MIN_ES:
            print(f'[{arm}] RESUME-EVAL ckpt ep{ck["ep"]}', flush=True)
            return model, ck['la_c'], ck['la_i'], ck['ep']
        start_ep = ck.get('ep', -1) + 1
        best_val = float(ck.get('best_val', 1e18)); best_la = (ck['la_c'], ck['la_i'])
        best_state = {k: v.clone() for k, v in ck['state'].items()}
        with torch.no_grad():
            la_c.fill_(best_la[0]); la_i.fill_(best_la[1])
        print(f'[{arm}] RESUME-CONTINUE ep{ck.get("ep")}', flush=True)

    def loss_of(xb, xib, ycb, yib, wxb=None):
        mu_raw, gate, r, gate_i, as_raw, q_raw = model(xb, xib if cfg['in_ch'] == 4 else None, wxb)
        mu_c = to_counts(mu_raw)
        pi_c = torch.sigmoid(gate)
        y_c = ycb[:, :, :, 0].permute(0, 2, 1)
        y_i = yib[:, :, :, 0].permute(0, 2, 1)
        if cfg['yroll']:
            y_i = torch.roll(y_i, ROLL, dims=1)
        L_c = zinb_nll_t(y_c, mu_c, pi_c, la_c).mean()
        Ec = (1 - pi_c) * mu_c
        if cfg['detach']:
            Ec = Ec.detach()
        mu_i = r * Ec + 1e-9
        if cfg.get('cmp'):
            if cfg.get('cmp2'):
                n_s = torch.exp(-as_raw)
            elif cfg.get('cmp5'):
                n_s = torch.exp(-model.log_alpha_s_cell).view(1, -1, 1)
            else:
                n_s = torch.exp(-model.log_alpha_s)
            p_s = n_s / (n_s + r.clamp(1e-6, 50.0))
            mask = (y_c > 0).float()
            li = nb_nll_var(y_i, y_c * n_s, p_s)
            if cfg.get('cmpzt'):
                # ZTNB: 减 log(1−t^k), t^k = p_s^{k n_s}; log1mexp 稳定式; y_i=0 且 k>0 在 ZTNB 下
                # 概率为零 → 罚 30 (与 (y_c=0,y_i>0) 支撑违例同约定, 概率上限 e^-30)
                logt = (y_c * n_s * torch.log(p_s.clamp(1e-7, 1 - 1e-7))).clamp(max=-1e-7)
                log1mt = torch.where(logt > -0.6931, torch.log(-torch.expm1(logt)),
                                     torch.log1p(-torch.exp(logt)))
                li = li + log1mt   # −log[NB/(1−t^k)] = nb_nll + log(1−t^k), log1mt<0 → 截断后 nll 变小
                L_i = (mask * torch.where(y_i == 0, torch.full_like(li, 30.0), li)
                       + (1 - mask) * (y_i > 0).float() * 30.0).mean()
            elif cfg.get('cmp3') or cfg.get('cmp6'):
                if cfg.get('cmp6'):
                    q = pi_c
                elif cfg.get('cmpq'):
                    q = torch.sigmoid(model.logit_q)
                else:
                    q = torch.sigmoid(q_raw)
                p0s = p_s ** (y_c * n_s)
                l0 = -torch.log(q + (1 - q) * p0s)
                lpos = -torch.log1p(-q.clamp(1e-7, 1 - 1e-7)) + li
                lix = torch.where(y_i == 0, l0, lpos)
                L_i = (mask * lix + (1 - mask) * (y_i > 0).float() * 30.0).mean()
            else:
                L_i = (mask * li + (1 - mask) * (y_i > 0).float() * 30.0).mean()
        elif cfg['zi']:
            pi_i = torch.sigmoid(gate_i)
            if COUPLE:
                pi_i = 1 - (1 - pi_i) * (1 - pi_c)
            L_i = zinb_nll_t(y_i, mu_i, pi_i, la_i).mean()
        else:
            L_i = nb_nll_t(y_i, mu_i, la_i).mean()
        return L_c + L_i, L_c, L_i, mu_c, pi_c, r, y_c, y_i

    for ep in range(start_ep, EPOCHS):
        model.train()
        tot, tc, ti, cnt = 0.0, 0.0, 0.0, 0
        for x, xi, yc, yi, wx in batches('train'):
            xb = torch.tensor(x, device=DEV); xib = torch.tensor(xi, device=DEV)
            ycb = torch.tensor(yc, device=DEV); yib = torch.tensor(yi, device=DEV)
            wxb = torch.tensor(wx, device=DEV) if WX else None
            opt.zero_grad()
            loss, L_c, L_i, *_ = loss_of(xb, xib, ycb, yib, wxb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5)
            opt.step()
            tot += loss.item(); tc += L_c.item(); ti += L_i.item(); cnt += 1
        model.eval()
        vl, vc = 0.0, 0
        with torch.no_grad():
            for x, xi, yc, yi, wx in batches('val'):
                l, _, _, *_ = loss_of(torch.tensor(x, device=DEV), torch.tensor(xi, device=DEV),
                                      torch.tensor(yc, device=DEV), torch.tensor(yi, device=DEV),
                                      torch.tensor(wx, device=DEV) if WX else None)
                vl += l.item(); vc += 1
        vl /= max(vc, 1)
        if ES_MARG and cfg.get('cmp') and ep % 5 == 0:
            mc, pc, rr, _pi, _as, _q = predict(model, arm, 'val')
            _a_s = _as if _as is not None else float(torch.exp(model.log_alpha_s).detach())
            vl = marg_mix_nll(mc, pc, rr, _a_s, YI_VAL, float(la_c.detach()))
        if np.isfinite(vl) and vl < best_val - 1e-5:
            best_val, patience = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_la = (float(la_c.detach()), float(la_i.detach()))
            torch.save({'state': best_state, 'la_c': best_la[0], 'la_i': best_la[1],
                        'ep': ep, 'best_val': best_val}, ck_path)
        else:
            patience += 1
        if ep % 5 == 0 or patience == 0:
            print(f'[{arm}] ep{ep} train={tot / max(cnt, 1):.4f}(c={tc / max(cnt, 1):.4f}/i={ti / max(cnt, 1):.4f}) '
                  f'val={vl:.4f} ac={np.exp(la_c.item()):.3f} ai={np.exp(la_i.item()):.3f}', flush=True)
        if ep >= MIN_ES and patience >= PATIENCE:
            print(f'[{arm}] ES at ep{ep}', flush=True)
            break
    if best_state is None:
        raise RuntimeError(f'{arm}: 无有限 val 改善')
    model.load_state_dict(best_state)
    global _LA_C
    _LA_C = best_la[0]
    return model, best_la[0], best_la[1], ep + 1

# ---- 评估 (EB_mom 地板 + 双 bootstrap + 一致性 + 率诊断) ----
def predict(model, arm, cat):
    cfg = ARM_CFG[arm]
    model.eval()
    Mc, Pc, Rr, Pi, As, Qs = [], [], [], [], [], []
    with torch.no_grad():
        for x, xi, yc, yi, wx in batches(cat):
            mu_raw, gate, r, gate_i, as_raw, q_raw = model(torch.tensor(x, device=DEV),
                                            torch.tensor(xi, device=DEV) if cfg['in_ch'] == 4 else None,
                                            torch.tensor(wx, device=DEV) if WX else None)
            Mc.append(to_counts(mu_raw).cpu().numpy())
            Pc.append(torch.sigmoid(gate).cpu().numpy())
            Rr.append(r.cpu().numpy())
            if gate_i is not None:
                Pi.append(torch.sigmoid(gate_i).cpu().numpy())
            if as_raw is not None:
                As.append(torch.exp(-as_raw).cpu().numpy())
            if q_raw is not None:
                Qs.append(torch.sigmoid(q_raw).cpu().numpy())
    S = len(starts[cat])
    pi_i = np.concatenate(Pi)[:S] if Pi else None
    a_s = np.concatenate(As)[:S] if As else None
    q = np.concatenate(Qs)[:S] if Qs else None
    return np.concatenate(Mc)[:S], np.concatenate(Pc)[:S], np.concatenate(Rr)[:S], pi_i, a_s, q

def marg_mix_nll(mu_c, pi_c, r, a_s, y, ac, K=24):
    pic = np.clip(pi_c, 1e-7, 1 - 1e-7); mu = np.maximum(mu_c, 1e-6)
    n_c = 1 / ac; p_c = n_c / (n_c + mu)
    a_s = float(np.asarray(a_s).mean())
    n_s = 1 / a_s; p_s = n_s / (n_s + np.clip(r, 1e-6, 50))
    acc = np.where(y == 0, pic + (1 - pic) * p_c ** n_c, 0.0)
    for k in range(1, K + 1):
        acc = acc + (1 - pic) * nbinom.pmf(k, n_c, p_c) * np.exp(nbinom.logpmf(y, k * n_s, p_s))
    v = float(-np.log(np.maximum(acc, 1e-300)).mean())
    return v if np.isfinite(v) else 1e9

def boot2(d_w, days):
    S = len(d_w)
    rb = np.random.default_rng(7)
    bs = d_w[rb.integers(0, S, size=(500, S))].mean(1)
    win = (float(np.quantile(bs, .025)), float(np.quantile(bs, .975)))
    ud = np.unique(days)
    dmean = np.array([d_w[days == d].mean() for d in ud])
    bs2 = dmean[np.random.default_rng(8).integers(0, len(ud), size=(500, len(ud)))].mean(1)
    day = (float(np.quantile(bs2, .025)), float(np.quantile(bs2, .975)))
    return win, day

def nb_pts_np(y, mu, a):
    n = 1.0 / max(a, 1e-8); mu = np.maximum(mu.astype(np.float64), 1e-6)
    return -nbinom.logpmf(y.astype(np.float64), n, n / (n + mu))

def recall20_mc(score, y, Rr=3, seed=11, K=20):
    sc = np.transpose(score, (0, 2, 1)).reshape(-1, n_reg)
    yy = np.transpose(y, (0, 2, 1)).reshape(-1, n_reg)
    pos = yy > 0; npos = pos.sum(1); hp = npos > 0
    rng = np.random.default_rng(seed); out = []
    for _ in range(Rr):
        perm = rng.permutation(n_reg)
        top = perm[np.argsort(-sc[:, perm], axis=1, kind='stable')[:, :K]]
        oh = np.zeros((sc.shape[0], n_reg), bool); oh[np.arange(sc.shape[0])[:, None], top] = True
        out.append(float(((oh[hp] & pos[hp]).sum(1) / np.minimum(K, npos[hp])).mean()))
    return round(float(np.mean(out)), 4)

def eval_arm(arm, model, la_c, la_i):
    cfg = ARM_CFG[arm]
    ac, ai = float(np.exp(la_c)), float(np.exp(la_i))
    res_rows = []
    diag = {}
    for sp in ['val', 'test']:
        mu_c, pi_c, r, pi_i, a_s, qv = predict(model, arm, sp)
        if pi_i is not None and COUPLE:
            pi_i = 1 - (1 - pi_i) * (1 - pi_c)
        yc = np.transpose(np.load(f'{DATA_C}/{sp}.npz')['y'][:, :, :, 0], (0, 2, 1)).astype(np.float64)
        yi = np.transpose(np.load(f'{DATA_I}/{sp}.npz')['y'][:, :, :, 0], (0, 2, 1)).astype(np.float64)
        # 负对照语义: yroll 臂训练用错位标签, 评估一律用真实标签 (增益须来自真实空间配对)
        yi_eval = yi
        S = len(starts[sp]); days = starts[sp][:S] // 24
        np.savez_compressed(f'{OUT}/mu_{arm}{SUF}{EVAL_TAG}_{sp}.npz',
                            mu_c=mu_c.astype(np.float32), pi_c=pi_c.astype(np.float32),
                            r=r.astype(np.float32),
                            **({'pi_i': pi_i.astype(np.float32)} if pi_i is not None else {}),
                            **({'alpha_s': (a_s.astype(np.float32) if a_s is not None else np.float32(np.exp(model.log_alpha_s.detach().cpu())))} if cfg.get('cmp') else {}))
        Ec = (1 - np.clip(pi_c, 1e-7, 1 - 1e-7)) * np.maximum(mu_c, 1e-6)
        mu_i = r * Ec + 1e-9
        # crash 通道
        pic = np.clip(pi_c, 1e-7, 1 - 1e-7)
        pts_c = -np.logaddexp(np.log(pic) + np.where(yc == 0, 0.0, -30.0),
                              np.log1p(-pic) + nbinom.logpmf(yc, 1 / ac, (1 / ac) / ((1 / ac) + np.maximum(mu_c, 1e-6))))
        mf_c = floor_mu(CLIM_C, sp, S)
        pts_fc = nb_pts_np(yc, mf_c, A_FC)
        n_c0 = 1.0 / ac
        pocc_c = 1 - (pic + (1 - pic) * (n_c0 / (n_c0 + np.maximum(mu_c, 1e-6))) ** n_c0)
        # inj 通道: compound 边际混合 / ZINB(zi臂, π_i 自有门) / NB(无gate)
        n_i = 1.0 / max(ai, 1e-8)
        if cfg.get('cmp'):
            if a_s is not None:
                a_s_arr = a_s
            elif cfg.get('cmp5'):
                a_s_arr = np.exp(model.log_alpha_s_cell.detach().cpu().numpy())[None, :, None]
            else:
                a_s_arr = float(np.exp(model.log_alpha_s.detach().cpu()))
            n_s = 1.0 / np.maximum(a_s_arr, 1e-6)
            p_s = n_s / (n_s + np.clip(r, 1e-6, 50.0))
            n_c = 1.0 / ac; p_c = n_c / (n_c + np.maximum(mu_c, 1e-6))
            Pc0 = pic + (1 - pic) * p_c ** n_c
            if cfg.get('cmpzt'):
                # ZTNB 边际: P(y_i=0)=P(y_c=0) 精确; P(y_i=j≥1)=Σ_{k≥1} P(y_c=k)·NB(j;k n_s,p_s)/(1−t^k)
                lsp = np.log(np.clip(p_s, 1e-7, 1 - 1e-7))
                pos = yi_eval > 0
                acc = np.where(yi_eval == 0, Pc0, 0.0)
                Ei_mean = np.zeros_like(acc)
                for k in range(1, 25):
                    pk = (1 - pic) * nbinom.pmf(k, n_c, p_c)
                    logt = np.minimum(k * n_s * lsp, -1e-7)
                    log1mt = np.where(logt > -0.6931, np.log(-np.expm1(logt)), np.log1p(-np.exp(logt)))
                    acc = acc + np.where(pos, pk * np.exp(nbinom.logpmf(yi_eval, k * n_s, p_s) - log1mt), 0.0)
                    Ei_mean = Ei_mean + pk * (k * r / np.exp(log1mt))   # E[ZTNB]=k·r/(1−t^k)
                pts_i = -np.log(np.maximum(acc, 1e-300))
                pocc_i = 1 - Pc0
                mu_i = Ei_mean
            else:
                if cfg.get('cmp6'):
                    qq = pic
                elif cfg.get('cmpq'):
                    qq = 1 / (1 + np.exp(-float(model.logit_q.detach().cpu())))
                else:
                    qq = qv if qv is not None else 0.0
                acc = np.where(yi_eval == 0, Pc0, 0.0)
                for k in range(1, 25):
                    acc = acc + (1 - pic) * nbinom.pmf(k, n_c, p_c) * np.exp(nbinom.logpmf(yi_eval, k * n_s, p_s))
                acc = np.where(yi_eval == 0, qq + (1 - qq) * acc, (1 - qq) * acc)
                pts_i = -np.log(np.maximum(acc, 1e-300))
                t = p_s ** n_s
                G = pic + (1 - pic) * (p_c / (1 - (1 - p_c) * t)) ** n_c
                pocc_i = (1 - qq) * (1 - G)
                Ei_mean = Ec * r
                mu_i = Ei_mean
        elif pi_i is not None:
            pii = np.clip(pi_i, 1e-7, 1 - 1e-7)
            lnb_i = nbinom.logpmf(yi_eval, n_i, n_i / (n_i + np.maximum(mu_i, 1e-6)))
            pts_i = -np.logaddexp(np.log(pii) + np.where(yi_eval == 0, 0.0, -30.0),
                                  np.log1p(-pii) + lnb_i)
            p0_nb = (n_i / (n_i + np.maximum(mu_i, 1e-6))) ** n_i
            pocc_i = 1 - (pii + (1 - pii) * p0_nb)
        else:
            pts_i = nb_pts_np(yi_eval, mu_i, ai)
            pocc_i = 1 - (n_i / (n_i + np.maximum(mu_i, 1e-6))) ** n_i
        mf_i = floor_mu(CLIM_I, sp, S)
        pts_fi = nb_pts_np(yi_eval, mf_i, A_FI)
        Ei_mean = (1 - np.clip(pi_i, 1e-7, 1 - 1e-7)) * mu_i if pi_i is not None else mu_i
        row = dict(arm=arm + SUF, window=sp + (VAL_YEAR if sp == 'val' else TEST_YEAR if sp == 'test' else ''),
                   alpha_c=round(ac, 5), alpha_i=round(ai, 5),
                   c_NLL=round(float(pts_c.mean()), 5), c_floor=round(float(pts_fc.mean()), 5),
                   c_dNLL_pct=round(100 * float((pts_c - pts_fc).mean()) / float(pts_fc.mean()), 3),
                   i_NLL=round(float(pts_i.mean()), 5), i_floor=round(float(pts_fi.mean()), 5),
                   i_dNLL_pct=round(100 * float((pts_i - pts_fi).mean()) / float(pts_fi.mean()), 3),
                   c_AUC=round(float(roc_auc_score((yc.ravel() > 0).astype(int), pocc_c.ravel())), 5),
                   i_AUC=round(float(roc_auc_score((yi_eval.ravel() > 0).astype(int), pocc_i.ravel())), 5),
                   i_Recall20=recall20_mc(pocc_i, yi_eval),
                   inconsist=round(float((Ei_mean > Ec).mean()), 5),
                   r_mean=round(float(r.mean()), 4), r_p99=round(float(np.quantile(r, .99)), 4))
        for tag, d_w, fl in [('c', (pts_c - pts_fc).reshape(S, -1).mean(1), float(pts_fc.mean())),
                             ('i', (pts_i - pts_fi).reshape(S, -1).mean(1), float(pts_fi.mean()))]:
            win, day = boot2(d_w, days)
            row[f'{tag}_win_ci'] = f'{win[0]:.6f},{win[1]:.6f}'
            row[f'{tag}_win_sig'] = bool(win[1] < 0)
            row[f'{tag}_day_sig'] = bool(day[1] < 0)
            row[f'{tag}_day_sig_worse'] = bool(day[0] > 0)
        res_rows.append(row)
        if sp == 'test':
            # 率诊断: 学到的细胞级率 vs 静态先验 vs 观测 (zi臂用 E_i=(1−π_i)μ_i)
            rc = Ei_mean.sum(axis=(0, 2)) / np.maximum(Ec.sum(axis=(0, 2)), 1e-9)
            obs = yi.sum(axis=(0, 2)) / np.maximum(yc.sum(axis=(0, 2)), 1e-9)
            m30 = yc.sum(axis=(0, 2)) >= 30
            diag = dict(dyn_vs_static=round(float(np.corrcoef(rc[m30], R_STATIC[m30])[0, 1]), 4),
                        dyn_vs_obs=round(float(np.corrcoef(rc[m30], obs[m30])[0, 1]), 4),
                        n_cells=int(m30.sum()))
        print(f'[{arm}/{sp}] c_NLL={row["c_NLL"]}({row["c_dNLL_pct"]:+.2f}%) i_NLL={row["i_NLL"]}({row["i_dNLL_pct"]:+.2f}%) '
              f'iAUC={row["i_AUC"]} iR20={row["i_Recall20"]} inconsist={row["inconsist"]} '
              f'csig={row["c_win_sig"]}/{row["c_day_sig"]} isig={row["i_win_sig"]}/{row["i_day_sig"]}', flush=True)
    return res_rows, diag

res_path = f'{OUT}/joint_results.csv'
done_arms = set() if os.environ.get('JOINT_FRESH') == '1' else (set(pd.read_csv(res_path)['arm'].tolist()) if os.path.exists(res_path) else set())
ARMS = [a for a in os.environ.get('JOINT_ARMS', 'joint3,joint4,joint3_detach,joint3_yroll').split(',') if a.strip()]
for arm in ARMS:
    if EVAL_TAG == '' and arm + SUF in done_arms:
        continue
    t0 = time.time()
    model, la_c, la_i, eps = run_arm(arm)
    rows, diag = eval_arm(arm, model, la_c, la_i)
    if EVAL_TAG == '':
        for row in rows:
            row['epochs'] = eps; row['secs'] = round(time.time() - t0)
            row['rate_diag'] = json.dumps(diag) if row['window'].startswith('test') else ''
            pd.DataFrame([row]).to_csv(res_path, mode='a', header=not os.path.exists(res_path), index=False)
    else:
        print(f'[{arm}] EVAL_TAG={EVAL_TAG}: npz only, joint_results.csv untouched', flush=True)
    print(f'[{arm}] DONE {time.time() - t0:.0f}s diag={diag}', flush=True)
print('ALL DONE ->', res_path)
