# Chicago grid-hour crash labels.
# Source: Chicago Traffic Crashes (Chicago Data Portal 85ca-t3if), crash date+time and lat/lon.
# Grid: 20x20 UTM 16N (EPSG:32616).
# Splits: train 2021-2023 / val 2024 / test 2025, aligned with New York City.
import numpy as np, pandas as pd, os, json, hashlib
from pyproj import Transformer

WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
RAW = f'{WS}/data/raw/phase7_exposure/chicago_crashes/85ca-t3if_crashes.csv'
OUT = f'{WS}/data/processed/T3_region_hour_chicago'
MAN = f'{WS}/data/manifests'
os.makedirs(OUT, exist_ok=True)

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:16]

# ---------- grid (fixed 20x20 bbox in EPSG:32616) ----------
x0, x1, y0, y1 = 421741.70615222614, 456737.4409564454, 4610049.955374941, 4653029.234621085
G = 20; n_reg = G * G
tf = Transformer.from_crs('EPSG:4326', 'EPSG:32616', always_xy=True)
print(f'Chicago grid bbox: ({x0:.0f},{x1:.0f},{y0:.0f},{y1:.0f}) UTM16N')

START = pd.Timestamp('2021-01-01'); END = pd.Timestamp('2026-01-01')
n_hours = int((END - START).total_seconds() // 3600)
n_days = n_hours // 24
C = np.zeros((n_reg, n_hours), dtype=np.float32)
INJ = np.zeros_like(C); KIL = np.zeros_like(C)
st = dict(rows=0, bad_date=0, out_range=0, no_geo=0, out_bbox=0, kept=0, est_date=0,
          injured_sum=0.0, killed_sum=0.0)

usecols = ['CRASH_DATE', 'CRASH_DATE_EST_I', 'LATITUDE', 'LONGITUDE', 'INJURIES_TOTAL', 'INJURIES_FATAL']
for chunk in pd.read_csv(RAW, usecols=usecols, chunksize=300_000, low_memory=False):
    st['rows'] += len(chunk)
    dt = pd.to_datetime(chunk['CRASH_DATE'], errors='coerce', format='%m/%d/%Y %I:%M:%S %p')
    st['bad_date'] += int(dt.isna().sum())
    st['est_date'] += int((chunk['CRASH_DATE_EST_I'].astype(str).str.upper() == 'Y').sum())
    inr = dt.notna() & (dt >= START) & (dt < END)
    st['out_range'] += int((~inr).sum())
    lat = pd.to_numeric(chunk['LATITUDE'], errors='coerce')
    lon = pd.to_numeric(chunk['LONGITUDE'], errors='coerce')
    okc = inr & lat.notna() & lon.notna() & (lat.abs() > 1)
    st['no_geo'] += int((inr & ~okc).sum())
    if okc.sum() == 0:
        continue
    x, y = tf.transform(lon[okc].values, lat[okc].values)
    ib = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    st['out_bbox'] += int((~ib).sum())
    x, y = x[ib], y[ib]
    gx = np.clip(((x - x0) / (x1 - x0) * G).astype(int), 0, G - 1)
    gy = np.clip(((y - y0) / (y1 - y0) * G).astype(int), 0, G - 1)
    cell = gy * G + gx
    tsv = dt[okc].values[ib]
    hidx = ((pd.DatetimeIndex(tsv) - START).total_seconds() // 3600).astype(int)
    inj = pd.to_numeric(chunk.loc[okc, 'INJURIES_TOTAL'], errors='coerce').fillna(0).values[ib]
    kil = pd.to_numeric(chunk.loc[okc, 'INJURIES_FATAL'], errors='coerce').fillna(0).values[ib]
    np.add.at(C, (cell, hidx), 1.0)
    np.add.at(INJ, (cell, hidx), inj)
    np.add.at(KIL, (cell, hidx), kil)
    st['kept'] += int(len(cell)); st['injured_sum'] += float(inj.sum()); st['killed_sum'] += float(kil.sum())
    print(f"  chunk rows={st['rows']} kept={st['kept']}", flush=True)
print(json.dumps(st, indent=1))

D = C.reshape(n_reg, n_days, 24).sum(2)
t3_days = pd.date_range(START, END - pd.Timedelta(days=1), freq='D')

qa = {}  # QA against the shared export is reported in meta.json; not recomputed here

years = t3_days.year.values
tr = np.where(years <= 2023)[0]; va = np.where(years == 2024)[0]; te = np.where(years == 2025)[0]
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
json.dump(dict(task='T3_region_hour_chicago', unit='20x20 UTM16N grid × hour (与NYC T3同构同时间轴)',
               source='Chicago 85ca-t3if (data.cityofchicago.org)', n_regions=n_reg, n_hours=n_hours,
               splits=dict(train='2021-2023', val='2024', test='2025'), stats=st, qa_against_shared_daily_labels=qa,
               bbox=[x0, x1, y0, y1], crs='EPSG:32616',
               built_at=pd.Timestamp.now().isoformat()),
          open(f'{OUT}/meta.json', 'w'), indent=2, default=str)
json.dump(dict(task='T3_region_hour_chicago', sources={'85ca-t3if_crashes.csv': sha(RAW)},
               files=['labels.npz', 'hour_index.parquet', 'day_index.parquet', 'meta.json'],
               stats=st, qa_against_shared_daily_labels=qa, built_at=pd.Timestamp.now().isoformat()),
          open(f'{MAN}/T3_region_hour_chicago.json', 'w'), indent=2)
print('T3-Chicago built:', C.shape, 'kept', st['kept'], '| zero frac', round(float((C == 0).mean()), 4))
