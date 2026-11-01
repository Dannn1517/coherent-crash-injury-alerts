# SetTF: compact set-transformer forecaster for sparse grid-hour crash counts.
# Each grid cell is a token: a 12h window of (count, time-of-day, day-of-week) is projected to
# d_model, combined with learned node and time embeddings, and encoded by four full-attention
# layers over the 400 cells. A count head (NB or ZINB) emits the next 12 hours (0.59M parameters).
# Protocol: locked test year, empirical-Bayes floor, window- and day-clustered bootstraps.

import numpy as np, pandas as pd, os, sys, json, time, warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.nn.functional as TF
from scipy.stats import nbinom
from sklearn.metrics import roc_auc_score

WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
R = f'{WS}/data/model_inputs'
OUT = f'{WS}/experiments/results/settf'
os.makedirs(OUT, exist_ok=True)
DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH = int(os.environ.get('SETTF_BS', '64'))   # 40×40分辨率注意力显存防护()
EPOCHS, PATIENCE, MIN_ES = 250, 30, 40
LR = 1e-3
SEED = int(os.environ.get('SETTF_SEED', '6666'))
VAL_YEAR = os.environ.get('SETTF_VAL_YEAR', '2024')
TEST_YEAR = os.environ.get('SETTF_TEST_YEAR', '2025')
TAG = os.environ.get('SETTF_TAG', '')       # 候选③严重度通道: _inj/_kil (产物名/arm 后缀区分)
LABEL_KEY = os.environ.get('SETTF_LABEL_KEY', 'region_hour')   # 地板气候学所用标签
DATA_DIR = os.environ.get('SETTF_DATA_DIR', f'{R}/crash_hour')
SUF = f'{TAG}' + ('' if SEED == 6666 else f'_s{SEED}')   # 6666 无tag 保持原产物名向后兼容
D_MODEL, N_LAYER, N_HEAD, D_FF = 128, 4, 8, 256
IN_LEN, OUT_LEN, STRIDE = 12, 12, 3

# ---- 数据 (与transfer script同 npz; val/test 不打乱保持 starts 对齐) ----
# SETTF_T3: 地板气候学所用标签文件必须与 DATA_DIR 同城 (stage18 bug 教训 :
# 只参数化 DATA_DIR/LABEL_KEY 漏了标签路径 → Chicago 臂用 NYC 地板, dNLL 虚高 −25%/−34%)
t3 = np.load(os.environ.get('SETTF_T3', f'{WS}/data/processed/T3_region_hour/labels.npz'))
C = t3[LABEL_KEY]; n_reg, n_hours = C.shape
how = (np.arange(n_hours) % 168)
B_TRAIN_END = int(os.environ.get('SETTF_B_TRAIN_END', 24 * 1095))
B_VAL_END = int(os.environ.get('SETTF_B_VAL_END', 24 * 1461))
WARM = 168
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
# internal audit修正(): 旧式 sum()/len(tr_h)=全细胞小时总量(~10.8)→伪计数0.046≈无EB收缩,
# 地板系统性偏弱(dNLL%夸大~0.6pp); EB本意=每细胞均值率(0.027)→伪计数18.6。mean()为正确口径。
# 历史csv行(~0083)为legacy地板产物, 修正对照表: experiments/results/settf/floor_correction.csv
g_tr = C[:, tr_h].mean()
clim = np.zeros((n_reg, 168), np.float32)
for w in range(168):
    m = how[tr_h] == w
    clim[:, w] = (C[:, tr_h][:, m].sum(1) + 0.5) / (m.sum() + 0.5 / max(g_tr, 1e-9))

_raw = {}
for cat in ['train', 'val', 'test']:
    d = np.load(f'{DATA_DIR}/{cat}.npz')
    _raw['x_' + cat] = d['x'].astype(np.float32)
    _raw['y_' + cat] = d['y'].astype(np.float32)
MEAN = _raw['x_train'][..., 0].mean(); STD = _raw['x_train'][..., 0].std()
for cat in ['train', 'val', 'test']:
    _raw['x_' + cat][..., 0] = (_raw['x_' + cat][..., 0] - MEAN) / STD
perm = np.random.default_rng(SEED).permutation(len(_raw['x_train']))
_raw['x_train'] = _raw['x_train'][perm]; _raw['y_train'] = _raw['y_train'][perm]
# starts['train'] 不置换 (eval 地板α用原始序配对; bug教训 b7ee896 后登记)

def batches(cat, bs=BATCH):
    x, y = _raw['x_' + cat], _raw['y_' + cat]
    for i in range(0, len(x), bs):
        yield x[i:i + bs], y[i:i + bs]

# ---- 模型 ----
ARCH = os.environ.get('SETTF_ARCH', 'tf')   # mlp=attention消融对照(D38硬伤#8): 参数量匹配 per-node ResMLP
class ResMLPLayer(nn.Module):
    """参数量匹配 TransformerEncoderLayer(132,480/层): Linear(128,512)+GELU+Drop+Linear(512,128)+LN ≈132,000
    去跨节点注意力、保深度/宽度/残差/LN/dropout 结构 → 隔离"集合注意力本身"的贡献"""
    def __init__(self, d=D_MODEL, hid=512, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, hid), nn.GELU(), nn.Dropout(dropout), nn.Linear(hid, d))
        self.ln = nn.LayerNorm(d)
    def forward(self, x):
        return self.ln(x + self.mlp(x))

class SetTF(nn.Module):
    def __init__(self, n_nodes, head='nb'):
        super().__init__()
        self.head = head
        self.win_proj = nn.Linear(IN_LEN * 3, D_MODEL)          # 12h×(count,tod,dow) 展平
        self.node_emb = nn.Parameter(torch.empty(n_nodes, D_MODEL)); nn.init.xavier_uniform_(self.node_emb)
        self.hour_emb = nn.Parameter(torch.empty(24, D_MODEL)); nn.init.xavier_uniform_(self.hour_emb)
        self.dow_emb = nn.Parameter(torch.empty(7, D_MODEL)); nn.init.xavier_uniform_(self.dow_emb)
        if ARCH == 'mlp':
            self.enc = nn.Sequential(*[ResMLPLayer() for _ in range(N_LAYER)])
        else:
            enc = nn.TransformerEncoderLayer(D_MODEL, N_HEAD, dim_feedforward=D_FF,
                                             batch_first=True, dropout=0.1, activation='gelu')
            self.enc = nn.TransformerEncoder(enc, N_LAYER)
        self.ln = nn.LayerNorm(D_MODEL)
        self.out_mu = nn.Linear(D_MODEL, OUT_LEN)
        self.out_gate = nn.Linear(D_MODEL, OUT_LEN) if head == 'zinb' else None

    def forward(self, x):
        # x: (B, 12, N, 3) 标准化空间 (ch0 已标准化, ch1=tod, ch2=dow)
        B, T, N, F = x.shape
        h = x.permute(0, 2, 1, 3).reshape(B, N, T * F)          # (B,N,36)
        tok = self.win_proj(h) + self.node_emb.unsqueeze(0)
        # 时间嵌入: 窗内最后一步的 hour/dow (与 ST-LLM+ Temb 取 x[:,-1] 同语义)
        last_tod = x[:, -1, 0, 1] * 24                          # 还原 hour (中点编码: floor=hour)
        last_dow = x[:, -1, 0, 2]
        tok = tok + self.hour_emb[last_tod.long().clamp(0, 23)].unsqueeze(1) \
                  + self.dow_emb[last_dow.long().clamp(0, 6)].unsqueeze(1)
        z = self.ln(self.enc(tok))                              # 全注意力集合编码 (B,N,D)
        mu_raw = self.out_mu(z)                                 # (B,N,12) 标准化空间
        if self.out_gate is not None:
            return mu_raw, self.out_gate(z)
        return mu_raw, None

def nb_nll_t(y, mu, log_alpha):
    n = torch.exp(-log_alpha)
    log_p = torch.log(n) - torch.log(n + mu)
    log_1mp = torch.log(mu) - torch.log(n + mu)
    return -(torch.lgamma(n + y) - torch.lgamma(y + 1) - torch.lgamma(n) + n * log_p + y * log_1mp)

def zinb_nll_t(y, mu, pi, log_alpha):
    n = torch.exp(-log_alpha)
    log_p = torch.log(n) - torch.log(n + mu)
    log_1mp = torch.log(mu) - torch.log(n + mu)
    lnb = torch.lgamma(n + y) - torch.lgamma(y + 1) - torch.lgamma(n) + n * log_p + y * log_1mp
    l0 = torch.log(pi.clamp(1e-7, 1 - 1e-7)) + torch.where(y == 0, torch.zeros_like(lnb), torch.full_like(lnb, -30.0))
    lm = torch.log((1 - pi).clamp(1e-7, 1 - 1e-7)) + lnb
    return -torch.logaddexp(l0, lm)

def to_counts(mu_raw):          # 标准化空间 → 计数空间 mu (与训练同变换, b7ee896 教训)
    return TF.softplus(mu_raw * STD + MEAN) + 1e-4

def fit_alpha(y, mu):
    mu = np.clip(mu.astype(np.float64), 1e-6, None); y = y.astype(np.float64)
    return float(max((((y - mu) ** 2 - mu).mean()) / max((mu ** 2).mean(), 1e-12), 1e-4))

# ---- 训练 ----
def run_arm(head):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = SetTF(n_reg, head).to(DEV)
    log_alpha = torch.zeros(1, requires_grad=True, device=DEV)
    n_par = sum(p.numel() for p in model.parameters())
    params = list(model.parameters()) + [log_alpha]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    print(f'[settf_{head}] params={n_par} (ST-LLM+ 83M/可训3.1M 的 {n_par / 3.1e6:.2f}×可训量)', flush=True)
    ck_path = f'{OUT}/ckpt_settf_{head}{SUF}.pt'
    start_ep, best_val, best_la, best_state, patience = 0, 1e18, 0.0, None, 0
    if os.path.exists(ck_path):
        # ckpt['ep']=最优轮(非末轮); RESUME-CONTINUE 重建 AdamW(无动量)、patience 清零(保守方向, internal auditMINOR-7);
        # RESUME-EVAL 返回 epochs=最优轮, 与新训路径的终止轮+1 语义不同(csv epochs 列跨路径仅作量级参考)
        ck = torch.load(ck_path, map_location=DEV)
        model.load_state_dict(ck['state'])
        if ck.get('ep', -1) >= MIN_ES:
            print(f'[settf_{head}] RESUME-EVAL ckpt ep{ck["ep"]}', flush=True)
            return model, ck['la'], ck['ep']
        start_ep = ck.get('ep', -1) + 1
        best_val = float(ck.get('best_val', 1e18)); best_la = float(ck.get('la', 0.0))
        best_state = {k: v.clone() for k, v in ck['state'].items()}
        with torch.no_grad():
            log_alpha.fill_(best_la)
        print(f'[settf_{head}] RESUME-CONTINUE ep{ck.get("ep")}', flush=True)

    def loss_of(xb, yb):
        mu_raw, gate = model(xb)
        mu = to_counts(mu_raw)                                   # (B,N,12) 计数空间
        y = yb[:, :, :, 0].permute(0, 2, 1)                      # (B,12,N,1)→(B,N,12)
        if head == 'zinb':
            return zinb_nll_t(y, mu, torch.sigmoid(gate), log_alpha).mean()
        return nb_nll_t(y, mu, log_alpha).mean()

    for ep in range(start_ep, EPOCHS):
        model.train()
        tot, cnt = 0.0, 0
        for x, y in batches('train'):
            xb = torch.tensor(x, device=DEV); yb = torch.tensor(y, device=DEV)
            opt.zero_grad()
            loss = loss_of(xb, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5)
            opt.step()
            tot += loss.item(); cnt += 1
        model.eval()
        vl, vc = 0.0, 0
        with torch.no_grad():
            for x, y in batches('val'):
                vl += loss_of(torch.tensor(x, device=DEV), torch.tensor(y, device=DEV)).item(); vc += 1
        vl /= max(vc, 1)
        if not np.isfinite(vl):   # NaN 防护(internal auditMINOR-6): 非有限按未改善计并显式告警
            print(f'[settf_{head}] WARNING ep{ep} val={vl} 非有限', flush=True)
        if np.isfinite(vl) and vl < best_val - 1e-5:
            best_val, patience = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_la = float(log_alpha.detach())
            torch.save({'state': best_state, 'la': best_la, 'ep': ep, 'best_val': best_val}, ck_path)
        else:
            patience += 1
        if ep % 5 == 0 or patience == 0:
            print(f'[settf_{head}] ep{ep} train={tot / max(cnt, 1):.4f} val={vl:.4f} la={float(log_alpha):.3f}', flush=True)
        if ep >= MIN_ES and patience >= PATIENCE:
            print(f'[settf_{head}] ES at ep{ep}', flush=True)
            break
    if best_state is None:   # 全程无有限 val 改善 → 显式失败而非静默用末轮权重(internal auditMINOR-6)
        raise RuntimeError(f'settf_{head}{SUF}: 无任何有限 val 改善, 拒绝静默评估末轮权重')
    model.load_state_dict(best_state)
    return model, best_la, ep + 1

# ---- 评估 (与transfer script同口径: 同点集 vs H1_multi 地板 + 窗簇bootstrap门) ----
def predict(model, head, cat):
    model.eval()
    mus, pis = [], []
    with torch.no_grad():
        for x, y in batches(cat):
            mu_raw, gate = model(torch.tensor(x, device=DEV))
            mus.append(to_counts(mu_raw).cpu().numpy())
            if gate is not None:
                pis.append(torch.sigmoid(gate).cpu().numpy())
    mu = np.concatenate(mus)
    assert mu.shape[0] == len(starts[cat]), f'predict 行数 {mu.shape[0]} != starts {len(starts[cat])}'
    pi = np.concatenate(pis) if pis else None
    assert pi is None or pi.shape[0] == len(starts[cat])
    return mu, pi

def eval_arm(mu, pi, sp, alpha_model):
    y = np.transpose(np.load(f'{DATA_DIR}/{sp}.npz')['y'][:, :, :, 0], (0, 2, 1))
    S = y.shape[0]
    tt = starts[sp][:S, None] + IN_LEN + np.arange(OUT_LEN)[None, :]
    mu_floor = clim[:, how[tt.ravel()]].T.reshape(S, OUT_LEN, n_reg).transpose(0, 2, 1)
    y_tr = np.transpose(np.load(f'{DATA_DIR}/train.npz')['y'][:, :, :, 0], (0, 2, 1))
    tt_tr = starts['train'][:y_tr.shape[0], None] + IN_LEN + np.arange(OUT_LEN)[None, :]
    mu_floor_tr = clim[:, how[tt_tr.ravel()]].T.reshape(y_tr.shape[0], OUT_LEN, n_reg).transpose(0, 2, 1)
    a_floor = fit_alpha(y_tr.ravel(), mu_floor_tr.ravel())
    yf = y.ravel().astype(np.float64)
    mf = np.clip(mu.ravel().astype(np.float64), 1e-6, None)
    nn_ = 1.0 / max(alpha_model, 1e-8); p = nn_ / (nn_ + mf)
    if pi is not None:
        pf = np.clip(pi.ravel().astype(np.float64), 1e-7, 1 - 1e-7)
        lnb = nbinom.logpmf(yf, nn_, p)
        pts_m = -np.logaddexp(np.log(pf) + np.where(yf == 0, 0.0, -30.0), np.log(1 - pf) + lnb)
        p_occ = (1 - pf) * (1 - np.exp(-mf)); mean_pred = (1 - pf) * mf
    else:
        pts_m = -nbinom.logpmf(yf, nn_, p)
        p_occ = 1 - np.exp(-mf); mean_pred = mf
    ff = np.clip(mu_floor.ravel().astype(np.float64), 1e-6, None)
    nnf = 1.0 / max(a_floor, 1e-8); pf2 = nnf / (nnf + ff)
    pts_f = -nbinom.logpmf(yf, nnf, pf2)
    p_occ_f = 1 - np.exp(-ff)
    k = 20
    yt2 = yf.reshape(-1, n_reg)
    step = max(yt2.shape[0] // 2000, 1)
    def esr(sc):
        hits = tot = 0
        s2 = sc.reshape(-1, n_reg)
        for i in range(0, yt2.shape[0], step):
            hits += len(set(np.argpartition(-yt2[i], k)[:k]) & set(np.argpartition(-s2[i], k)[:k])); tot += k
        return round(hits / tot, 4)
    res = {'model': dict(NLL=round(float(pts_m.mean()), 5),
                         AUC_occ=round(float(roc_auc_score((yf > 0).astype(int), p_occ)), 5),
                         ESR5=esr(p_occ), MAE=round(float(np.abs(yf - mean_pred).mean()), 5),
                         alpha=round(alpha_model, 5)),
           'floor': dict(NLL=round(float(pts_f.mean()), 5),
                         AUC_occ=round(float(roc_auc_score((yf > 0).astype(int), p_occ_f)), 5),
                         ESR5=esr(p_occ_f), MAE=round(float(np.abs(yf - ff).mean()), 5), alpha=round(a_floor, 5))}
    d_w = (pts_m - pts_f).reshape(S, -1).mean(1)
    rb = np.random.default_rng(7)
    bs = d_w[rb.integers(0, S, size=(500, S))].mean(1)
    res['gate'] = dict(dNLL=float(d_w.mean()), pct=round(100 * float(d_w.mean()) / max(res['floor']['NLL'], 1e-9), 3),
                       ci_lo=float(np.quantile(bs, .025)), ci_hi=float(np.quantile(bs, .975)),
                       sig_better=bool(np.quantile(bs, .975) < 0), sig_worse=bool(np.quantile(bs, .025) > 0))
    return res

res_path = f'{OUT}/settf_results.csv'
done_arms = set() if os.environ.get('SETTF_FRESH') == '1' else (set(pd.read_csv(res_path)['arm'].tolist()) if os.path.exists(res_path) else set())
ARMS = [a for a in os.environ.get('SETTF_ARMS', 'nb,zinb').split(',') if a.strip()]
for head in ARMS:
    arm = f'settf_{head}{SUF}'
    if arm in done_arms:
        continue
    t0 = time.time()
    model, la, eps = run_arm(head)
    alpha_m = float(np.exp(la))
    for sp in ['val', 'test']:
        mu, pi = predict(model, head, sp)
        np.savez_compressed(f'{OUT}/mu_{arm}_{sp}.npz', mu=mu.astype(np.float32),
                            **({'pi': pi.astype(np.float32)} if pi is not None else {}))
        r = eval_arm(mu, pi, sp, alpha_m)
        g = r['gate']
        row = dict(arm=arm, window=f'{sp}{VAL_YEAR}' if sp == 'val' else f'{sp}{TEST_YEAR}', epochs=eps,
                   model_NLL=r['model']['NLL'], model_AUC=r['model']['AUC_occ'], model_ESR5=r['model']['ESR5'],
                   model_MAE=r['model']['MAE'], model_alpha=r['model']['alpha'],
                   floor_NLL=r['floor']['NLL'], floor_AUC=r['floor']['AUC_occ'], floor_ESR5=r['floor']['ESR5'],
                   floor_MAE=r['floor']['MAE'], dNLL=g['dNLL'], dNLL_pct=g['pct'],
                   ci_lo=g['ci_lo'], ci_hi=g['ci_hi'], sig_better=g['sig_better'], sig_worse=g['sig_worse'],
                   dAUC=round(r['model']['AUC_occ'] - r['floor']['AUC_occ'], 5),
                   dESR5=round(r['model']['ESR5'] - r['floor']['ESR5'], 4), secs=round(time.time() - t0))
        print(json.dumps(row), flush=True)
        pd.DataFrame([row]).to_csv(res_path, mode='a', header=not os.path.exists(res_path), index=False)
print('DONE ->', res_path)
