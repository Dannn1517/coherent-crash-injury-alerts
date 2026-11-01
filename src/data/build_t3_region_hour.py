# New York City grid-hour crash labels.
# Source: NYPD Motor Vehicle Collisions (NYC Open Data h9gi-nx95), hourly timestamps + lat/lon.
# Grid: 20x20 UTM 18N (EPSG:32618); G overridable for resolution sweeps.
# Splits: train 2021-2023 / val 2024 / test 2025; the test year is locked.
import numpy as np, pandas as pd, os, json, hashlib
from pyproj import Transformer

WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
RAW = f'{WS}/data/raw/phase7_exposure/nyc_collisions/h9gi-nx95_crashes.csv'
OUT = os.environ.get('T3_OUT', f'{WS}/data/processed/T3_region_hour')   # 多分辨率(): T3_OUT/T3_G 覆盖
MAN = f'{WS}/data/manifests'
os.makedirs(OUT, exist_ok=True); os.makedirs(MAN, exist_ok=True)

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:16]

# ---------- grid (fixed 20x20 bbox in EPSG:32618; G overridable for resolution sweep) ----------
x0, x1, y0, y1 = 563144.2036175783, 610524.2962249531, 4483351.195653233, 4530556.855261125
G = int(os.environ.get('T3_G', '20'))   # same bbox, different resolution (10/20/40)
n_reg = G * G
tf = Transformer.from_crs('EPSG:4326', 'EPSG:32618', always_xy=True)

# ---------- 小时索引: 2021-01-01 00时 .. 2025-12-31 23时 ----------
START = pd.Timestamp('2021-01-01'); END = pd.Timestamp('2026-01-01')
n_hours = int((END - START).total_seconds() // 3600)
n_days = n_hours // 24
print(f'grid {G}x{G} bbox=({x0:.0f},{x1:.0f},{y0:.0f},{y1:.0f}) | hours={n_hours} days={n_days}')

C = np.zeros((n_reg, n_hours), dtype=np.float32)
INJ = np.zeros((n_reg, n_hours), dtype=np.float32)
KIL = np.zeros((n_reg, n_hours), dtype=np.float32)
st = dict(rows=0, bad_date=0, bad_time=0, out_range=0, no_geo=0, out_bbox=0, kept=0,
          injured_sum=0.0, killed_sum=0.0)

usecols = ['CRASH DATE', 'CRASH TIME', 'LATITUDE', 'LONGITUDE',
           'NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED']
for chunk in pd.read_csv(RAW, usecols=usecols, chunksize=500_000):
    st['rows'] += len(chunk)
    dt = pd.to_datetime(chunk['CRASH DATE'], errors='coerce', format='%m/%d/%Y')
    st['bad_date'] += int(dt.isna().sum())
    tm = chunk['CRASH TIME'].astype(str).str.strip()
    hh = pd.to_numeric(tm.str.slice(0, 2), errors='coerce')
    bad_t = hh.isna() | (hh < 0) | (hh > 23)
    st['bad_time'] += int(bad_t.sum())
    ts = dt + pd.to_timedelta(hh.fillna(0), unit='h')
    inr = ts.notna() & (ts >= START) & (ts < END) & ~bad_t
    st['out_range'] += int((~inr).sum())
    lat = pd.to_numeric(chunk['LATITUDE'], errors='coerce')
    lon = pd.to_numeric(chunk['LONGITUDE'], errors='coerce')
    ok = inr & lat.notna() & lon.notna()
    st['no_geo'] += int((inr & ~ok).sum())
    if ok.sum() == 0:
        continue
    x, y = tf.transform(lon[ok].values, lat[ok].values)
    ib = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    st['out_bbox'] += int((~ib).sum())
    x, y = x[ib], y[ib]
    gx = np.clip(((x - x0) / (x1 - x0) * G).astype(int), 0, G - 1)
    gy = np.clip(((y - y0) / (y1 - y0) * G).astype(int), 0, G - 1)
    cell = gy * G + gx
    tsv = ts[ok].values[ib]
    hidx = ((pd.DatetimeIndex(tsv) - START).total_seconds() // 3600).astype(int)
    inj = pd.to_numeric(chunk.loc[ok, 'NUMBER OF PERSONS INJURED'], errors='coerce').fillna(0).values[ib]
    kil = pd.to_numeric(chunk.loc[ok, 'NUMBER OF PERSONS KILLED'], errors='coerce').fillna(0).values[ib]
    np.add.at(C, (cell, hidx), 1.0)
    np.add.at(INJ, (cell, hidx), inj)
    np.add.at(KIL, (cell, hidx), kil)
    st['kept'] += int(len(cell)); st['injured_sum'] += float(inj.sum()); st['killed_sum'] += float(kil.sum())
    print(f"  chunk done rows={st['rows']} kept={st['kept']}", flush=True)

print(json.dumps(st, indent=1))

# ---------- daily aggregate ----------
D = C.reshape(n_reg, n_days, 24).sum(2)
t3_days = pd.date_range(START, END - pd.Timedelta(days=1), freq='D')
qa = {}  # QA against the shared export is reported in meta.json; not recomputed here

# ---------- 划分 (对齐 T2) ----------
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
json.dump(dict(task='T3_region_hour', unit='20x20 UTM grid region × hour (与T2同网格)',
               source='NYPD h9gi-nx95 (data.cityofnewyork.us)', n_regions=n_reg, n_hours=n_hours,
               splits=dict(train='2021-2023', val='2024', test='2025'), stats=st, qa_against_shared_daily_labels=qa,
               bbox=[x0, x1, y0, y1], crs='EPSG:32618',
               built_at=pd.Timestamp.now().isoformat()),
          open(f'{OUT}/meta.json', 'w'), indent=2)
man_name = 'T3_region_hour' + ('' if G == 20 else f'_g{G}')
json.dump(dict(task='T3_region_hour', grid=G, sources={'h9gi-nx95_crashes.csv': sha(RAW)},
               files=['labels.npz', 'hour_index.parquet', 'day_index.parquet', 'meta.json'],
               stats=st, qa_against_shared_daily_labels=qa, built_at=pd.Timestamp.now().isoformat()),
          open(f'{MAN}/{man_name}.json', 'w'), indent=2)
print('T3 built:', C.shape, 'kept', st['kept'], '| zero frac', float((C == 0).mean()).__round__(4))
