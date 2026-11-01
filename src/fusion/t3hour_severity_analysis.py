# Data build 后续候选③分析(): crash vs injured 模型排序差异 + 细胞级严重度率学习检验 (零GPU)
# 问题: 严重度感知预警(injured 通道)是否在 crash 预警之外提供新信息?
# 输出: experiments/results/settf/severity_analysis.csv (指标名/值) + stdout
import numpy as np, pandas as pd, os

WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
OUT = f'{WS}/experiments/results/settf'
R = f'{WS}/data/model_inputs'
K = 20

d_c = np.load(f'{OUT}/mu_settf_zinb_test.npz')
d_i = np.load(f'{OUT}/mu_settf_zinb_inj_test.npz')

def p_occ(d):
    mu = d['mu'].astype(np.float64)
    p = 1 - np.exp(-np.maximum(mu, 1e-6))
    if 'pi' in d.files:
        p = p * (1 - np.clip(d['pi'].astype(np.float64), 1e-7, 1 - 1e-7))
    return p

pc, pij = p_occ(d_c), p_occ(d_i)
S, N, H = pc.shape

# 1) 每窗 top-20 重合度 (抽样窗×全 horizon)
ov = []
for s in range(0, S, 7):
    for h in range(H):
        tc = set(np.argpartition(-pc[s, :, h], K)[:K])
        ti = set(np.argpartition(-pij[s, :, h], K)[:K])
        ov.append(len(tc & ti) / K)
top20_overlap = float(np.mean(ov))

# 2) 细胞级平均风险相关
cell_corr = float(np.corrcoef(pc.mean(axis=(0, 2)), pij.mean(axis=(0, 2)))[0, 1])

# 3) 细胞级 severity rate (inj/crash): 模型预测 vs 观测 (test2025)
y_c = np.transpose(np.load(f'{R}/crash_hour/test.npz')['y'][:, :, :, 0], (0, 2, 1))
y_i = np.transpose(np.load(f'{R}/crash_hour_inj/test.npz')['y'][:, :, :, 0], (0, 2, 1))
mu_c = d_c['mu'].astype(np.float64) * (1 - d_c['pi'].astype(np.float64))
mu_i = d_i['mu'].astype(np.float64) * (1 - d_i['pi'].astype(np.float64))
pred_rate = mu_i.sum(axis=(0, 2)) / np.maximum(mu_c.sum(axis=(0, 2)), 1e-9)
obs_rate = y_i.sum(axis=(0, 2)) / np.maximum(y_c.sum(axis=(0, 2)), 1e-9)
crashes = y_c.sum(axis=(0, 2))
m = crashes >= 30
sev_corr = float(np.corrcoef(pred_rate[m], obs_rate[m])[0, 1])

# 4) 观测 severity 最高的细胞 (crashes>=100) 模型是否也判高
m2 = crashes >= 100
t5o = np.argsort(-np.where(m2, obs_rate, 0))[:5]
top5_obs = [(int(c), round(float(obs_rate[c]), 3), round(float(pred_rate[c]), 3)) for c in t5o]
# 预测 severity 最高 top5 (含小样本细胞的失真登记)
t5p = np.argsort(-pred_rate)[:5]
top5_pred = [(int(c), round(float(pred_rate[c]), 3), round(float(obs_rate[c]), 3), int(crashes[c])) for c in t5p]

rows = [
    dict(metric='top20_overlap_crash_vs_inj', value=round(top20_overlap, 4),
         note=f'test2025 抽样{len(ov)}窗×horizon, k={K}'),
    dict(metric='cell_mean_pocc_corr', value=round(cell_corr, 4), note='400细胞 crash vs inj 平均p_occ'),
    dict(metric='severity_rate_pred_obs_corr', value=round(sev_corr, 4),
         note=f'细胞级 inj/crash 率, n={int(m.sum())} cells ≥30 crashes'),
    dict(metric='top5_obs_severity_cells', value=str(top5_obs), note='(cell,obs_rate,pred_rate) crashes≥100'),
    dict(metric='top5_pred_severity_cells', value=str(top5_pred), note='(cell,pred,obs,crashes) 小分母失真登记'),
]
df = pd.DataFrame(rows)
df.to_csv(f'{OUT}/severity_analysis.csv', index=False)
print(df.to_string(index=False))
print('DONE ->', f'{OUT}/severity_analysis.csv')
