# Data build ST-LLM+ data format: T3 网格×小时 → ST-LLM+ 数据契约 (与 bike_pick 同格式, 已实证核对)
# x (S,12,N,3): ch0=事故计数(raw, loader 内标准化) ch1=tod(hour/24) ch2=dow(0-6); y (S,12,N,1): 未来12步计数
# 窗: 12h 输入 → 12h 输出, stride 3; 划分按输出窗整段落界: train≤2023 / val=2024 / test=2025 (跨界窗丢弃)
# adj_mx.pkl: 400×400 8邻域 0/1 float64 (与 bike adj 同风格: dense binary)
import numpy as np, pandas as pd, os, pickle, json

WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
# 参数化: T3_DIR=标签目录(NYC默认/Chicago跨城), OUT_DIR=ST-LLM格式输出 (env 覆盖)
T3_DIR = os.environ.get('T3_DIR', f'{WS}/data/processed/T3_region_hour')
OUT = os.environ.get('OUT_DIR', f'{WS}/data/model_inputs/crash_hour')
LABEL_KEY = os.environ.get('LABEL_KEY', 'region_hour')   # 候选③严重度通道: region_hour_injured/killed
os.makedirs(OUT, exist_ok=True)
IN_LEN, OUT_LEN, STRIDE = 12, 12, 3
N = None; G = None   # 多分辨率: 由标签形状导出 ()

t3 = np.load(f'{T3_DIR}/labels.npz')
C = t3[LABEL_KEY].astype(np.float64)          # (N, 43824) 与官方 float64 一致
n_reg, n_hours = C.shape
N = n_reg; G = int(round(n_reg ** 0.5)); assert G * G == N   # 多分辨率: N/G 由标签导出()
hidx = pd.read_parquet(f'{T3_DIR}/hour_index.parquet')   # internal auditMINOR: 跟随T3_DIR(原硬编码NYC路径, 跨城恰因时间轴相同未出错)
dow = hidx['dow'].values.astype(np.float64)
hour = hidx['hour'].values.astype(np.float64)
# tod 中点编码: TemporalEmbedding 用 (tod×24).long() 截断索引, hour/24 的 fp32 舍入
# 会把部分小时错位 off-by-one (如 23/24→22.9999→22); (hour+0.5)/24 ×24=hour+0.5 截断恒稳
tod = (hour + 0.5) / 24.0

# 划分边界可 env 覆盖 ( 台中四年轴: 730/1095 天); 默认 NYC/CHI 五年轴不变
B_TRAIN_END = int(os.environ.get('B_TRAIN_END_H', 24 * 1095))   # 2024-01-01 00时
B_VAL_END = int(os.environ.get('B_VAL_END_H', 24 * 1461))       # 2025-01-01 00时

xs, ys, splits = [], [], []
for i in range(0, n_hours - (IN_LEN + OUT_LEN) + 1, STRIDE):
    o_end = i + IN_LEN + OUT_LEN                    # 输出窗结束(排他)
    if o_end <= B_TRAIN_END:
        sp = 'train'
    elif i >= B_TRAIN_END and o_end <= B_VAL_END:
        sp = 'val'
    elif i >= B_VAL_END:
        sp = 'test'
    else:
        continue                                    # 跨界窗丢弃(防泄漏)
    x = np.empty((IN_LEN, n_reg, 3))
    x[:, :, 0] = C[:, i:i + IN_LEN].T
    x[:, :, 1] = tod[i:i + IN_LEN, None]
    x[:, :, 2] = dow[i:i + IN_LEN, None]
    y = np.empty((OUT_LEN, n_reg, 1))
    y[:, :, 0] = C[:, i + IN_LEN:o_end].T
    xs.append(x); ys.append(y); splits.append(sp)

X = np.stack(xs); Y = np.stack(ys)
splits = np.array(splits)
print('X', X.shape, 'Y', Y.shape, '| 划分:', {s: int((splits == s).sum()) for s in ['train', 'val', 'test']})
for sp in ['train', 'val', 'test']:
    m = splits == sp
    np.savez_compressed(f'{OUT}/{sp}.npz', x=X[m].astype(np.float32), y=Y[m].astype(np.float32))
    print(sp, 'saved', int(m.sum()), ' nonzero frac y:', float((Y[m] > 0).mean()).__round__(5))

# adj: 8邻域 binary (含自环=0, 与 bike nnz/节点≈80 不同登记: 网格只有≤8邻居, 机制同=邻接作attention偏置)
A = np.zeros((N, N), np.float64)
for r in range(N):
    gy, gx = divmod(r, G)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            yy, xx = gy + dy, gx + dx
            if 0 <= yy < G and 0 <= xx < G:
                A[r, yy * G + xx] = 1.0
with open(f'{OUT}/adj_mx.pkl', 'wb') as f:
    pickle.dump(A, f)
json.dump(dict(task=f'{LABEL_KEY}_hour_12to12', source=f'T3 ({T3_DIR}) label={LABEL_KEY}',
               n_nodes=N, in_len=IN_LEN, out_len=OUT_LEN, stride=STRIDE,
               splits={s: int((splits == s).sum()) for s in ['train', 'val', 'test']},
               ch='[count_raw, tod((hour+0.5)/24 中点编码), dow]', adj='8邻域binary 400x400',
               note='官方 float64→此处存 float32 减体积(loader 转 Tensor 不受影响); 跨界窗丢弃防泄漏',
               built_at=pd.Timestamp.now().isoformat()),
          open(f'{OUT}/build_meta.json', 'w'), indent=2, ensure_ascii=False)
print('adj nnz:', int(A.sum()), '| DONE ->', OUT)
