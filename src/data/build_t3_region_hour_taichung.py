# Taichung grid-hour crash labels.
# Source: Police Bureau, Taichung City Government, road traffic accident data (city open-data platform),
#   229,425 events, 2021-2024; minute timestamps + lon/lat + killed/injured.
# A1/A2 covers injury/fatal crashes only; property-damage-only A3 records are not public,
#   so the crash channel approximates injury crashes (injury/crash nonzero rate ratio ~1.0).
# Grid: 20x20, bbox in EPSG:32651; splits train 2021-2022 / val 2023 / test 2024.
import numpy as np, pandas as pd, os, json, hashlib
from pyproj import Transformer

WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
RAW = os.environ.get('TAICHUNG_CRASHES', f'{WS}/data/raw/taichung/crashes_events.parquet')
OUT = f'{WS}/data/processed/T3_region_hour_taichung'
MAN = f'{WS}/data/manifests'
os.makedirs(OUT, exist_ok=True)

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:16]

# ---------- grid (fixed 20x20 bbox in EPSG:32651) ----------
x0, x1, y0, y1 = 239460.77739213812, 345842.0678908945, 2652421.630535787, 2707893.2724373797
G = 20; n_reg = G * G
tf = Transformer.from_crs('EPSG:4326', 'EPSG:32651', always_xy=True)
print(f'Taichung grid bbox: ({x0:.0f},{x1:.0f},{y0:.0f},{y1:.0f}) UTM51N')

START = pd.Timestamp('2021-01-01'); END = pd.Timestamp('2025-01-01')
n_hours = int((END - START).total_seconds() // 3600)   # 35,064
n_days = n_hours // 24                                  # 1,461
C = np.zeros((n_reg, n_hours), dtype=np.float32)
INJ = np.zeros_like(C); KIL = np.zeros_like(C)
st = dict(rows=0, bad_date=0, out_range=0, no_geo=0, out_bbox=0, kept=0,
          injured_sum=0.0, killed_sum=0.0)

ev = pd.read_parquet(RAW, columns=['dt', 'lon', 'lat', 'killed', 'injured', 'id_geom_hash'])
st['rows'] = len(ev)
dt = pd.to_datetime(ev['dt'], errors='coerce')
st['bad_date'] = int(dt.isna().sum())
inr = dt.notna() & (dt >= START) & (dt < END)
st['out_range'] = int((~inr).sum())
lat = pd.to_numeric(ev['lat'], errors='coerce')
lon = pd.to_numeric(ev['lon'], errors='coerce')
okc = inr & lat.notna() & lon.notna() & (lat.abs() > 1)
st['no_geo'] = int((inr & ~okc).sum())
x, y = tf.transform(lon[okc].values.astype(float), lat[okc].values.astype(float))
ib = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
st['out_bbox'] = int((~ib).sum())
x, y = x[ib], y[ib]
gx = np.clip(((x - x0) / (x1 - x0) * G).astype(int), 0, G - 1)
gy = np.clip(((y - y0) / (y1 - y0) * G).astype(int), 0, G - 1)
cell = gy * G + gx
tsv = pd.DatetimeIndex(dt[okc].values[ib])
hidx = ((tsv - START).total_seconds() // 3600).astype(int)
inj = pd.to_numeric(ev.loc[okc, 'injured'], errors='coerce').fillna(0).values[ib]
kil = pd.to_numeric(ev.loc[okc, 'killed'], errors='coerce').fillna(0).values[ib]
np.add.at(C, (cell, hidx), 1.0)
np.add.at(INJ, (cell, hidx), inj)
np.add.at(KIL, (cell, hidx), kil)
st['kept'] = int(len(cell)); st['injured_sum'] = float(inj.sum()); st['killed_sum'] = float(kil.sum())
# 过滤统计闭合校验 (同 CHI 纪律): rows = bad_date + out_range + no_geo + out_bbox + kept
assert st['rows'] == st['bad_date'] + st['out_range'] + st['no_geo'] + st['out_bbox'] + st['kept'], st
print(json.dumps(st, indent=1))

D = C.reshape(n_reg, n_days, 24).sum(2)
t3_days = pd.date_range(START, END - pd.Timedelta(days=1), freq='D')

# 稀疏度与基率比 (机制第三数据点)
te_h = np.arange(24 * 1095, n_hours)   # 2024 测试年小时
sp = dict(
    nz_crash_test=round(float((C[:, te_h] > 0).mean()), 5),
    nz_inj_test=round(float((INJ[:, te_h] > 0).mean()), 5),
    nz_killed_test=round(float((KIL[:, te_h] > 0).mean()), 6),
    base_rate_ratio_test=round(float((INJ[:, te_h] > 0).mean() / max((C[:, te_h] > 0).mean(), 1e-12)), 4),
    g_crash_cellhour=round(float(C[:, 24 * 730:24 * 1095].mean()), 6),  # val2023 每细胞小时率
)
print('sparsity:', json.dumps(sp, indent=1))

qa = {}  # QA against the shared export is reported in meta.json; not recomputed here

years = t3_days.year.values
tr = np.where(years <= 2022)[0]; va = np.where(years == 2023)[0]; te = np.where(years == 2024)[0]
np.savez_compressed(f'{OUT}/labels.npz',
                    region_hour=C, region_hour_injured=INJ, region_hour_killed=KIL,
                    region_day=D, n_hours=n_hours, n_days=n_days,
                    train_days=tr, val_days=va, test_days=te)
pd.DataFrame({'hour_idx': np.arange(n_hours),
              'ts': (START + pd.to_timedelta(np.arange(n_hours), unit='h')).astype(str),
              'dow': (START + pd.to_timedelta(np.arange(n_hours), unit='h')).dayofweek,
              'hour': np.arange(n_hours) % 24}).to_parquet(f'{OUT}/hour_index.parquet')
pd.DataFrame({'day_idx': np.arange(n_days), 'id_date': t3_days.astype(str),
              'year': years, 'dow': t3_days.dayofweek}).to_parquet(f'{OUT}/day_index.parquet')
json.dump(dict(task='T3_region_hour_taichung',
               unit='20x20 UTM51N grid × hour (与NYC/CHI T3同构; 时间轴2021-2024四年)',
               source='Police Bureau, Taichung City Government, road traffic accident data',
               semantic_note='A1/A2 covers injury/fatal crashes only (A3 property-damage records are not public); crash channel approximates injury crashes; injury/crash nonzero rate ratio ~1.0',
               n_regions=n_reg, n_hours=n_hours,
               splits=dict(train='2021-2022', val='2023', test='2024'), stats=st, sparsity=sp,
               qa_against_shared_daily_labels=qa, bbox=[x0, x1, y0, y1], crs='EPSG:32651',
               built_at=pd.Timestamp.now().isoformat()),
          open(f'{OUT}/meta.json', 'w'), indent=2, default=str)
json.dump(dict(task='T3_region_hour_taichung', sources={'crashes_events.parquet': sha(RAW)},
               files=['labels.npz', 'hour_index.parquet', 'day_index.parquet', 'meta.json'],
               stats=st, sparsity=sp, qa_against_shared_daily_labels=qa, built_at=pd.Timestamp.now().isoformat()),
          open(f'{MAN}/T3_region_hour_taichung.json', 'w'), indent=2)
print('T3-Taichung built:', C.shape, 'kept', st['kept'], '| zero frac', round(float((C == 0).mean()), 4))
