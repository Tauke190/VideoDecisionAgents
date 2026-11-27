"""Compact architecture figure of uncertainty-aware hierarchical sampling (5.5 in wide, NeurIPS text width).

/usr/bin/python3 VideoJevOpenQA/figures/make_method_arch.py  ->  figures/method_arch.{pdf,svg,png}
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon

HERE = Path(__file__).resolve().parent
WIDTH_IN = 5.5
W, H = 1400, 452
INK, GREY, TICK = '#222222', '#6f6f6f', '#4d4d4d'
# draw.io palette (fill, stroke)
BLUE = ('#dae8fc', '#6c8ebf')
GREEN = ('#d5e8d4', '#82b366')
ORANGE = ('#ffe6cc', '#d79b00')
PURPLE = ('#e1d5e7', '#9673a6')
YELLOW = ('#fff2cc', '#d6b656')
RED = ('#f8cecc', '#b85450')
plt.rcParams.update({'font.family': 'Liberation Sans', 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                     'mathtext.fontset': 'custom', 'mathtext.it': 'Liberation Sans:italic'})

fig = plt.figure(figsize=(WIDTH_IN, WIDTH_IN * H / W))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')
PT = W / WIDTH_IN / 72  # data units per point


def rbox(x, y, w, h, fc='white', ec=INK, lw=0.8, ls='-', r=4, alpha=1.0, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw,
                                ls=ls, alpha=alpha, zorder=z))


def txt(x, y, s, size=6, color=INK, ha='center', va='center', weight='normal', z=6, alpha=1.0, style='normal',
        rot=0):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, weight=weight, zorder=z, alpha=alpha, style=style,
            rotation=rot, linespacing=1.15)


def line(pts, color=INK, lw=0.8, ls='-', z=2, alpha=1.0, head=False):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z, alpha=alpha, solid_capstyle='butt')
    if head:
        ax.annotate('', xy=pts[-1], xytext=pts[-2], zorder=z,
                    arrowprops=dict(arrowstyle='-|>,head_length=0.4,head_width=0.2', color=color, lw=lw,
                                    shrinkA=0, shrinkB=0, alpha=alpha))


def mark(x, y, sure, alpha=1.0, r=11):
    ax.add_patch(plt.Circle((x, y), r, fc=(GREEN if sure else ORANGE)[1], ec='white', lw=0.6, zorder=7, alpha=alpha))
    if sure:
        line([(x - .45 * r, y), (x - .1 * r, y + .4 * r), (x + .5 * r, y - .4 * r)], 'white', 1.0, z=8, alpha=alpha)
    else:
        txt(x, y + 1, '?', 5.5, 'white', weight='bold', z=8)


def node(x, y, w, bumps, sure, window, alpha=1.0, h=62, ticks=32):
    # one grounding pass: 64 frames spread over this span (ticks) + relevance curve
    fill, edge = GREEN if sure else ORANGE
    rbox(x, y, w, h, fc=fill, ec=edge, lw=0.9, alpha=alpha, z=2)
    pad, ch = 5, 32
    rbox(x + pad, y + pad, w - 2 * pad, h - 2 * pad, fc='white', ec='none', r=2, alpha=alpha, z=2.5)
    u = np.linspace(0, 1, 300)
    f = sum(a * np.exp(-0.5 * ((u - m) / s) ** 2) for a, m, s in bumps) + 0.03
    f = f / f.max()
    xs, base = x + pad + u * (w - 2 * pad), y + pad + ch
    inside = (u >= window[0]) & (u <= window[1])
    ax.fill_between(xs[inside], base, base - f[inside] * (ch - 3), color=edge, alpha=0.6 * alpha, lw=0, zorder=3)
    ax.plot(xs, base - f * (ch - 3), color=INK, lw=0.6, alpha=alpha, zorder=4)
    line([(x + pad, base + 2), (x + w - pad, base + 2)], '#d0d0d0', 0.4, z=3, alpha=alpha)
    for t in np.linspace(x + pad + 3, x + w - pad - 3, ticks):
        line([(t, base + 7), (t, y + h - pad - 4)], TICK, 0.45, z=3, alpha=alpha)
    mark(x + w, y, sure, alpha)
    return dict(x=x, y=y, w=w, h=h, tx=lambda v: x + pad + v * (w - 2 * pad))


def zoom(parent, a, b, child, alpha=1.0):
    pa, pb, by = parent['tx'](a), parent['tx'](b), parent['y'] + parent['h']
    line([(pa, by + 4), (pb, by + 4)], ORANGE[1], 2.0, z=4, alpha=alpha)
    top = child['y'] - 2
    ax.add_patch(Polygon([(pa, by + 6), (pb, by + 6), (child['x'] + child['w'], top), (child['x'], top)],
                         fc='#efefef', alpha=alpha, ec='none', zorder=1))
    for p, c in [(pa, child['x']), (pb, child['x'] + child['w'])]:
        line([(p, by + 6), (c, top)], GREY, 0.5, (0, (3, 2)), alpha=alpha)


def group(x, y, w, h, title, colors, size=6.5):
    rbox(x, y, w, h, fc='white', ec=colors[1], lw=0.8, ls=(0, (4, 2.5)), r=5, z=0)
    tw = 0.56 * size * PT * len(title) + 26
    rbox(x, y, tw, 32, fc=colors[0], ec=colors[1], lw=0.8, r=5, z=0.5)
    txt(x + 12, y + 16, title, size, ha='left', weight='bold')


# input
for k in range(3):
    rbox(12 + 7 * k, 170 + 7 * k, 86, 54, fc=BLUE[0], ec=BLUE[1], lw=0.7, r=2, z=2 + k)
txt(62, 252, 'long video', 6, weight='bold')
rbox(10, 272, 106, 34, fc='white', ec=INK, lw=0.7)
txt(63, 289, 'question $q$', 5.5)
line([(116, 212), (136, 212)], INK, 0.8, head=True)

# System 1
G1 = (140, 4, 852, 444)
group(*G1, 'System 1: Decision Model', BLUE)
X0, X1 = 330, 930
rows = [(48, 'depth 0', '64 frames over\nthe whole video'),
        (160, 'depth 1', '64 frames over\neach uncertain peak'),
        (272, 'depth 2', '64 frames over\na sub-peak')]
for y, d, s in rows:
    txt(156, y + 16, d, 6, ha='left', weight='bold')
    txt(156, y + 44, s, 5, GREY, ha='left')

root = node(X0, 48, X1 - X0, [(0.7, 0.3, 0.045), (1.0, 0.7, 0.05)], False, (0.62, 0.78))
a = node(390, 160, 270, [(0.6, 0.28, 0.06), (0.8, 0.6, 0.05), (0.3, 0.85, 0.04)], False, (0.52, 0.7))
b = node(720, 160, 200, [(1.0, 0.5, 0.06)], True, (0.38, 0.62))
zoom(root, 0.24, 0.36, a)
zoom(root, 0.62, 0.78, b)
a2 = node(380, 272, 120, [(1.0, 0.45, 0.1), (0.4, 0.82, 0.06)], True, (0.25, 0.65), alpha=0.4)
a1 = node(540, 272, 120, [(1.0, 0.5, 0.09)], True, (0.32, 0.68))
zoom(a, 0.18, 0.38, a2, alpha=0.5)
zoom(a, 0.52, 0.7, a1)

line([(962, 56), (962, 330)], GREY, 0.8, head=True)
txt(977, 193, 'narrower span, denser frames', 5, GREY, rot=-90)

for nd, r, used in [(b, 1, True), (a1, 2, True), (a2, 3, False)]:
    txt(nd['x'] + nd['w'] / 2, nd['y'] + nd['h'] + 13, f'leaf #{r}', 5.5, INK if used else '#b0b0b0', weight='bold')
bus = 372
for nd in (a1, b):
    cx = nd['x'] + nd['w'] / 2
    line([(cx, nd['y'] + nd['h'] + 25), (cx, bus)], INK, 0.8)
line([(a1['x'] + a1['w'] / 2, bus), (1016, bus)], INK, 0.8, head=True)
txt(740, bus + 16, 'rank leaves by path prior × confidence, keep top-$K$', 5.5, weight='bold')

ly = 424
mark(166, ly, False, r=9); txt(180, ly, 'uncertain: zoom into peaks', 5, ha='left')
mark(408, ly, True, r=9); txt(422, ly, 'confident: leaf', 5, ha='left')
line([(560, ly), (584, ly)], ORANGE[1], 2.0); txt(592, ly, 'span to zoom', 5, ha='left')
for t in np.linspace(720, 740, 5):
    line([(t, ly - 7), (t, ly + 7)], TICK, 0.5)
txt(748, ly, 'sampled frames', 5, ha='left')

# System 2
group(1000, 4, 396, 444, 'System 2: VideoQA', GREEN)
rbox(1016, 318, 364, 116, fc='white', ec=INK, lw=0.7)
txt(1198, 334, 'top-$K$ leaves, in time order', 5, GREY)
for k, lab in enumerate(['leaf #2', 'leaf #1']):
    y = 348 + k * 42
    txt(1028, y + 16, lab, 5.5, ha='left', weight='bold')
    for i in range(6):
        rbox(1110 + i * 44, y, 40, 32, fc=GREEN[0], ec=GREEN[1], lw=0.6, r=1, z=3)
line([(1198, 316), (1198, 290)], INK, 0.8, head=True)
rbox(1068, 230, 260, 58, fc=PURPLE[0], ec=PURPLE[1], lw=0.9)
txt(1198, 250, 'VLM (Qwen)', 7, weight='bold')
txt(1198, 272, 'frames + question $q$', 5, GREY)
line([(1198, 228), (1198, 202)], INK, 0.8, head=True)
rbox(1068, 158, 260, 42, fc=YELLOW[0], ec=YELLOW[1], lw=0.9)
txt(1198, 179, 'short answer', 7, weight='bold')
line([(1198, 156), (1198, 130)], INK, 0.8, head=True)
rbox(1030, 44, 336, 84, fc=RED[0], ec=RED[1], lw=0.9, ls=(0, (3, 2)))
txt(1198, 66, 'LLM judge (Qwen3-8B)', 6.5, weight='bold')
txt(1198, 89, 'answer vs. ground truth → correct?', 5, GREY)
txt(1198, 109, '(evaluation only)', 5, RED[1], style='italic')

for ext in ('pdf', 'svg', 'png'):
    fig.savefig(HERE / f'method_arch.{ext}', dpi=400 if ext == 'png' else None)
print(HERE / 'method_arch.pdf')
