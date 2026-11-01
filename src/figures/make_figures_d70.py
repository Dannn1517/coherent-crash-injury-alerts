# : 双面板证据图 (论文 Fig.2): (a) 机制剂量-反应价格梯度 (旧 f4 内容精简版)
#      (b) 顺序逐周期协议 Δrecall vs 容量, 7天块 paired 95% CI ( 实验可视化)
# 数据源: (a) experiments/results/joint/zt_analysis.csv (4-seed means) + g40 登记值; (b) sequential_capacity_ci.csv
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(f'{HERE}/../..')
FIG = f'{WS}/figures'; os.makedirs(FIG, exist_ok=True)
BLUE, GREEN, BROWN, RED, PURPLE = '#0072B2', '#009E73', '#D55E00', '#CC79A7', '#56B4E9'
INK, GRAY, PAPER = '#333333', '#7F7F7F', '#FFFFFF'
CCOL = {'nyc': BLUE, 'chi': GREEN, 'taichung': BROWN}
MK = {'nyc': 'o', 'chi': 's', 'taichung': 'D'}
plt.rcParams.update({'font.size': 8.8, 'pdf.fonttype': 42, 'figure.facecolor': PAPER, 'axes.facecolor': PAPER,
                     'axes.edgecolor': INK, 'axes.labelcolor': INK, 'text.color': INK,
                     'xtick.color': INK, 'ytick.color': INK, 'axes.linewidth': 0.6})

fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.0, 2.05), gridspec_kw=dict(width_ratios=[1, 1.08], wspace=0.34))

# ---------- (a) mechanism price gradient ----------
zt = pd.read_csv(f'{WS}/experiments/results/joint/zt_analysis.csv')
agg = zt.groupby('city').mean(numeric_only=True)
META = {'chi': (0.177, 2.82, 'Chicago'), 'nyc': (0.453, 2.23, 'New York'), 'taichung': (0.999, 1.52, 'Taichung')}
ORDER = ['chi', 'nyc', 'taichung']
nb = {c: agg.loc[c, 'cmp_dNLL'] - agg.loc[c, 'zi_dNLL'] for c in ORDER}
ztp = {c: agg.loc[c, 'zt_dNLL'] - agg.loc[c, 'zi_dNLL'] for c in ORDER}
mt = {c: min(nb[c], ztp[c]) for c in ORDER}
xs = [META[c][0] for c in ORDER]
ZT_CLIP = 10.5
axa.axhline(0, color=INK, ls='--', lw=0.6)
axa.plot(xs, [nb[c] for c in ORDER], color=INK, lw=0.9, zorder=1, alpha=0.75)
axa.plot(xs, [min(ztp[c], ZT_CLIP) for c in ORDER], color=GRAY, lw=0.9, ls=':', zorder=1)
for c, x in zip(ORDER, xs):
    nz = META[c][1]
    axa.scatter([x], [nb[c]], s=nz*30, facecolor=CCOL[c], edgecolor=INK, lw=0.5, zorder=3, marker='o', alpha=0.92)
    axa.scatter([x], [min(ztp[c], ZT_CLIP)], s=nz*22, facecolor='white', edgecolor=CCOL[c], lw=0.9, zorder=3, marker='^')
    if ztp[c] > ZT_CLIP:
        axa.annotate('', xy=(x, ZT_CLIP+0.5), xytext=(x, ZT_CLIP-0.05),
                     arrowprops=dict(arrowstyle='-|>', lw=0.8, color=INK), zorder=5)
    # regime-matched = ink halo ring around the winning family's marker (no glyph-on-glyph overlap)
    if nb[c] <= ztp[c]:
        axa.scatter([x], [nb[c]], s=nz*30+50, facecolor='none', edgecolor=INK, lw=0.9, zorder=5, marker='o')
    else:
        axa.scatter([x], [min(ztp[c], ZT_CLIP)], s=nz*22+45, facecolor='none', edgecolor=INK, lw=0.9, zorder=5, marker='^')
for c, x in zip(ORDER, xs):
    # label each marker on the side away from both price curves
    if nb[c] >= 0:
        axa.annotate(f'{nb[c]:+.1f}', (x, nb[c]), textcoords='offset points', xytext=(7, -9),
                     ha='left', va='top', fontsize=7.4)
    else:
        axa.annotate(f'{nb[c]:+.1f}', (x, nb[c]), textcoords='offset points', xytext=(-7, -6),
                     ha='right', va='top', fontsize=7.4)
    lab = f'{ztp[c]:+.1f}' if ztp[c] <= ZT_CLIP else f'\u2265+{ztp[c]:.0f}'
    axa.annotate(lab, (x, min(ztp[c], ZT_CLIP)), textcoords='offset points', xytext=(-7, -7),
                 ha='right', va='top', fontsize=7.4)
axa.plot([0.442, 0.442], [-0.85, 0.0], color=BLUE, lw=0.5, ls=':', zorder=2)
axa.scatter([0.442], [-1.05], s=0.58*26, marker='D', facecolor='white', edgecolor=BLUE, lw=0.9, zorder=4)
axa.set_xlabel('injury / crash base-rate ratio', fontsize=8.2)
axa.set_ylabel('price vs free-gate\n(pp of injury NLL)', fontsize=8.2)
axa.set_ylim(-2.2, 12.6); axa.set_xlim(-0.08, 1.25)
axa.tick_params(labelsize=7.4, width=0.5, length=2.2)
for s_ in ('top', 'right'):
    axa.spines[s_].set_visible(False)
axa.annotate('(a)', (-0.02, 1.06), xycoords='axes fraction', fontsize=8.8, va='bottom', ha='left', weight='bold')

# ---------- (b) sequential per-cycle Δrecall vs capacity ----------
ci = pd.read_csv(f'{WS}/experiments/results/joint/sequential_capacity_ci.csv')
for c in ['nyc', 'chi', 'taichung']:
    for m, ls, lw, eb in [('cmp', '-', 1.1, True), ('zi', '--', 0.7, False)]:
        sub = ci[(ci.city == c) & (ci.model == m)].sort_values('B_frac')
        x = sub.B_frac.values*100; y = sub.d_recall.values*100
        err = np.vstack([y - sub.d_recall_ci_lo.values*100, sub.d_recall_ci_hi.values*100 - y]) if eb else None
        axb.errorbar(x, y, yerr=err, color=CCOL[c], ls=ls, lw=lw, marker=MK[c], ms=2.8,
                     elinewidth=0.6, capthick=0.6, capsize=1.6, label=None,
                     zorder=3 if eb else 2, alpha=1.0 if eb else 0.75)
axb.axhline(0, color=INK, ls='--', lw=0.6)
axb.set_xlabel('capacity B (% of cell-hours)', fontsize=8.2)
axb.set_ylabel('injury recall gain over EB (pp)', fontsize=8.2)
axb.tick_params(labelsize=7.4, width=0.5, length=2.2)
for s_ in ('top', 'right'):
    axb.spines[s_].set_visible(False)
# 图例: 城市色 + 线型说明
_hb = [Line2D([], [], color=CCOL[c], marker=MK[c], ms=3.2, lw=1.0, label=META[c][2])
       for c in ['nyc', 'chi', 'taichung']]
axb.legend(handles=_hb, loc='upper left', frameon=False, fontsize=7.4,
           handletextpad=0.4, borderaxespad=0.2, labelspacing=0.25)
axb.annotate('(b)', (-0.02, 1.06), xycoords='axes fraction', fontsize=8.8, va='bottom', ha='left', weight='bold')

fig.savefig(f'{FIG}/f2_evidence.pdf', bbox_inches='tight')
print('f2_evidence saved')
