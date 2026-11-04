"""
Figures for the v3 measurement series (README "Measurements"): one PNG per experiment from results/v3/results.jsonl.

    python -P experiments/v3/plots.py                 # writes results/v3/figures/*.png
    python -P experiments/v3/plots.py --only E3 E10   # a subset

Every figure reads the same rows the README tables read (the last row per key in results.jsonl, status ok; the
baselines from the experiment's own FA-2 / SDPA rows, else E1's / E6's at the same node, N, D, dtype and direction,
exactly as bench.py's report does). Time is shown as the ratio to one FlashAttention-2 call on the same
device-resident tensors (1.0 = FA-2; the SDPA ratio is the gray line), memory as the peak device allocation in GiB.
One style for all: the four series keep their colour in every figure (classic CUDA blue, wave CUDA orange, classic
Triton green, wave Triton yellow), the baselines are gray, forward on the top row and forward+backward below.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bench import BASELINES, SERIES, baseline_lookup, latest_by_key, read_rows  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DEFAULT = os.path.join(ROOT, "results", "v3", "figures")
K, M = 1024, 1024 * 1024

# --- palette (the validated default of the data-viz method: categorical slots 1-4 in fixed order; grays for the
# baselines; the sequential blue / orange ramps for ordered settings) ---------------------------------------------
SURFACE = "#fcfcfb"
TEXT, TEXT2, TEXT3 = "#0b0b0b", "#52514e", "#8a8984"
GRID, AXIS = "#e6e5e1", "#c9c8c3"
COLOR = {"classic_CUDA": "#2a78d6", "wave_CUDA": "#eb6834", "classic_Triton": "#1baf7a", "wave_Triton": "#eda100",
         "FA-2": "#52514e", "SDPA": "#a3a29c"}
MARKER = {"classic_CUDA": "o", "wave_CUDA": "s", "classic_Triton": "^", "wave_Triton": "D", "FA-2": "x", "SDPA": "+"}
LABEL = {"classic_CUDA": "classic CUDA", "wave_CUDA": "wave CUDA", "classic_Triton": "classic Triton",
         "wave_Triton": "wave Triton", "FA-2": "FlashAttention-2", "SDPA": "SDPA"}
BLUES = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]          # ordinal ramp, light -> dark (steps 250/350/450/600)
ORANGES = ["#f3a98c", "#eb6834", "#b6431b"]
DIRS = (("fwd", "forward"), ("bwd", "forward + backward"))

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "axes.titleweight": "normal", "axes.titlecolor": TEXT, "axes.labelcolor": TEXT2,
    "xtick.color": TEXT2, "ytick.color": TEXT2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.axisbelow": True,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.minor.size": 0, "ytick.minor.size": 0,
    "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 1.8, "lines.markersize": 6.5,
    "lines.markeredgewidth": 1.2, "lines.markeredgecolor": SURFACE,
})


def nfmt(n):
    return f"{n // M}M" if n >= M else f"{n // K}K"


def wall(r):
    return r["wall_s"]


# ---------------------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------------------
class Data:
    def __init__(self, out_dir):
        rows = list(latest_by_key(read_rows(out_dir)).values())
        self.ok = [r for r in rows if r.get("status", "ok") == "ok" and r.get("wall_s") is not None]
        self.base = baseline_lookup(self.ok)

    def rows(self, exp, **flt):
        out = []
        for r in self.ok:
            if r["exp"] != exp:
                continue
            if all(r.get(k) == v for k, v in flt.items()):
                out.append(r)
        return out

    def ratio(self, r, arm="FA-2"):
        b = self.base(r, arm)
        return wall(r) / wall(b) if b else None

    def baseline(self, r, arm="FA-2"):
        return self.base(r, arm)


# ---------------------------------------------------------------------------------------------------------------
# style helpers
# ---------------------------------------------------------------------------------------------------------------
def n_axis(ax, Ns):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(Ns))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: nfmt(int(round(v)))))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("N (tokens)")


def c_axis(ax, cs):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(cs))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: str(int(round(v)))))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("c (quorum size = subproblems at depth 1)")


def gib_axis(ax, log=False):
    ax.set_ylabel("peak device memory (GiB)")
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_locator(LogLocator(subs=[]))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_ylim(bottom=0)


def ratio_axis(ax, label="wall time / FlashAttention-2", log=False):
    ax.set_ylabel(label)
    ax.axhline(1.0, color=COLOR["FA-2"], lw=1.0, zorder=1)
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator([1, 1.2, 1.5, 2, 3, 4, 6]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_locator(LogLocator(subs=[]))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(bottom=0.86)


def ratio_line_label(ax, x, text="FA-2 = 1.0", y=1.0, ha="left"):
    ax.annotate(text, (x, y), xytext=(0, -3), textcoords="offset points", ha=ha, va="top", fontsize=8, color=TEXT3)


def note_arrow(ax, x, y, text, dx, dy, color=TEXT2, size=8):
    """A label pushed away from a crowded point, with a hairline leader."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha="center", va="center", fontsize=size,
                color=color, zorder=6, arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 0.8, "shrinkA": 0,
                                                   "shrinkB": 4})


def coincide_note(ax, series_vals, text):
    """When every series in the panel lies within 1% of the first, say so (lines drawn on top of each other)."""
    keys = list(series_vals)
    if len(keys) < 2:
        return
    ref = series_vals[keys[0]]
    for k in keys[1:]:
        v = series_vals[k]
        if len(v) != len(ref) or any(a is None or b is None or abs(a - b) > 0.01 * abs(b) for a, b in zip(v, ref)):
            return
    ax.text(0.02, 0.04, text, transform=ax.transAxes, fontsize=8, color=TEXT3, ha="left", va="bottom")


def series_line(ax, xs, ys, arm, ls="-", filled=True, label=None, z=3, **kw):
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if not pts:
        return
    xs, ys = zip(*pts)
    face = COLOR[arm] if filled else SURFACE
    edge = SURFACE if filled else COLOR[arm]
    ax.plot(xs, ys, ls, color=COLOR[arm], marker=MARKER[arm], markerfacecolor=face, markeredgecolor=edge,
            label=label if label is not None else LABEL[arm], zorder=z, solid_capstyle="round", **kw)


def gray_line(ax, xs, ys, arm, label=None, z=2):
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if not pts:
        return
    xs, ys = zip(*pts)
    ax.plot(xs, ys, "-", color=COLOR[arm], marker=MARKER[arm], markeredgecolor=COLOR[arm], markersize=6,
            markeredgewidth=1.4, lw=1.3, label=label if label is not None else LABEL[arm], zorder=z)


def note(ax, x, y, text, dx=0, dy=6, ha="center", va="bottom", color=TEXT2, size=8):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=size, color=color,
                zorder=5)


def bar(ax, x, h, width, color, z=3, faded=False):
    if h is None:
        return
    ax.add_patch(Rectangle((x - width / 2, 0), width, h, linewidth=0, facecolor=color, zorder=z,
                           alpha=0.35 if faded else 1.0))


def grouped_bars(ax, groups, bars, values, colors, labels, gap=0.06, label_fmt=None, label_sel=None, faded=()):
    """groups: x category names; bars: bar names; values[g][b] -> float or None. Bars <= 24 px: the group's slot
    is 0.8 wide and the bars share it with a surface gap."""
    n = len(bars)
    slot = 0.72
    w = min((slot - gap * (n - 1)) / n, 0.18)          # never a slab: a thin column leaves the slot's rest as air
    ymax = max(v for g in values.values() for v in g.values() if v is not None)
    ax.set_ylim(0, ymax * 1.18)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    for gi, g in enumerate(groups):
        for bi, b in enumerate(bars):
            v = values[g].get(b)
            x = gi - slot / 2 + w / 2 + bi * (w + gap)
            bar(ax, x, v, w, colors[b], faded=(g, b) in faded)
            if v is not None and label_fmt and (label_sel is None or label_sel(g, b)):
                note(ax, x, v, label_fmt(v), dy=3, size=7.5)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.xaxis.grid(False)
    for b in bars:
        ax.add_patch(Rectangle((0, 0), 0, 0, facecolor=colors[b], label=labels[b]))


def legend_below(fig, handles_labels=None, ncol=None, y=0.0, ax=None):
    if handles_labels is None:
        h, l = [], []
        for a in ([ax] if ax else []) + list(fig.axes):
            hh, ll = a.get_legend_handles_labels()
            h += hh
            l += ll
        handles_labels = (h, l)
    h, l = handles_labels
    seen, hh, ll = set(), [], []
    for a, b in zip(h, l):
        if b not in seen:
            seen.add(b)
            hh.append(a)
            ll.append(b)
    # the baselines last, FA-2 before SDPA
    def rank(b):
        if b.startswith("SDPA"):
            return 20
        if b.startswith("FlashAttention") or b.startswith("FA-2"):
            return 10
        for i, a in enumerate(SERIES):
            if b.startswith(LABEL[a]):
                return i
        return 5
    order = sorted(range(len(ll)), key=lambda i: (rank(ll[i]), i))
    hh, ll = [hh[i] for i in order], [ll[i] for i in order]
    fig.legend(hh, ll, loc="lower center", ncol=ncol or len(ll), bbox_to_anchor=(0.5, y), handlelength=2.2,
               columnspacing=1.6, handletextpad=0.6)


def finish(fig, path, title, subtitle=None, legend_h=0.07):
    H = fig.get_size_inches()[1]
    n_sub = subtitle.count("\n") + 1 if subtitle else 0
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT, y=1 - 0.12 / H, va="top")
    if subtitle:
        fig.text(0.01, 1 - 0.36 / H, subtitle, ha="left", va="top", fontsize=9, color=TEXT2)
    top = 1 - (0.5 + 0.16 * n_sub) / H
    fig.tight_layout(rect=(0, legend_h, 1, top))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


def ylim_pad(ax, lo=None, hi=None, pad=0.08):
    y0, y1 = ax.get_ylim()
    if lo is not None:
        y0 = min(y0, lo)
    if hi is not None:
        y1 = max(y1, hi)
    ax.set_ylim(y0, y1 + pad * (y1 - y0))


# ---------------------------------------------------------------------------------------------------------------
# the figures
# ---------------------------------------------------------------------------------------------------------------
def fig_headline(d, out):
    """E1 + E2 classic CUDA against FA-2: the two configurations the README leads with."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    Ns = sorted({r["N"] for r in d.rows("E1", node="sxm")})
    for row, (dr, dname) in enumerate(DIRS):
        at, am = axes[row]
        for exp, arm_label, filled, ls in (("E1", "Stream-CQSA, Q/K/V on the device", True, "-"),
                                           ("E2", "Stream-CQSA, Q/K/V + accumulator in host memory", False, "-")):
            rows = {r["N"]: r for r in d.rows(exp, arm="classic_CUDA", direction=dr, node="sxm")}
            xs = [n for n in Ns if n in rows]
            color_arm = "classic_CUDA" if exp == "E1" else "wave_CUDA"
            series_line(at, xs, [d.ratio(rows[n]) for n in xs], color_arm, ls, filled, label=arm_label)
            series_line(am, xs, [rows[n]["peak_dev_gib"] for n in xs], color_arm, ls, filled, label=arm_label)
            if exp == "E1":
                for n in xs:
                    if rows[n].get("itr_ran", 1) > 1:
                        note_arrow(at, n, d.ratio(rows[n]), f"depth {rows[n]['itr_ran']}", -30, -13)
        base = {r["N"]: r for r in d.rows("E1", arm="FA-2", direction=dr)}
        xs = [n for n in Ns if n in base]
        gray_line(am, xs, [base[n]["peak_dev_gib"] for n in xs], "FA-2")
        sd = {r["N"]: r for r in d.rows("E1", arm="SDPA", direction=dr)}
        gray_line(at, xs, [d.ratio(sd[n]) for n in xs if n in sd], "SDPA")
        n_axis(at, Ns)
        n_axis(am, Ns)
        ratio_axis(at, log=True)
        ratio_line_label(at, Ns[0])
        gib_axis(am, log=True)
        at.set_title(f"{dname}: wall time relative to FlashAttention-2")
        am.set_title(f"{dname}: peak device memory")
    legend_below(fig, ncol=2)
    finish(fig, os.path.join(out, "headline.png"),
           "Stream-CQSA against one FlashAttention-2 call, 128K to 4M tokens",
           "classic CUDA engine, c=7, the smallest depth that fits; A100-SXM4-80GB, fp16, B=1 H=8 D=64, causal (E1, E2)",
           legend_h=0.10)


def fig_scaling(d, out, exp, title, subtitle, fname, acc_note=False):
    """E1 / E2: the four series vs N, ratio to FA-2 and peak device memory, forward and forward+backward."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4))
    Ns = sorted({r["N"] for r in d.rows(exp, node="sxm")} | {r["N"] for r in d.rows("E1", node="sxm")})
    for row, (dr, dname) in enumerate(DIRS):
        at, am = axes[row]
        for arm in reversed(SERIES):
            rows = {r["N"]: r for r in d.rows(exp, arm=arm, direction=dr, node="sxm")}
            xs = [n for n in Ns if n in rows]
            series_line(at, xs, [d.ratio(rows[n]) for n in xs], arm)
            series_line(am, xs, [rows[n]["peak_dev_gib"] for n in xs], arm)
            if arm in ("classic_CUDA", "wave_CUDA"):
                for n in xs:
                    it = rows[n].get("itr_ran", 1)
                    if it and it > 1:
                        note_arrow(at, n, d.ratio(rows[n]), f"{LABEL[arm]}: depth {it}",
                                   -46, 26 if arm == "classic_CUDA" else 44)
        base = {r["N"]: r for r in d.rows("E1", arm="FA-2", direction=dr, node="sxm")}
        sd = {r["N"]: r for r in d.rows("E1", arm="SDPA", direction=dr, node="sxm")}
        xs = [n for n in Ns if n in base]
        gray_line(am, xs, [base[n]["peak_dev_gib"] for n in xs], "FA-2")
        gray_line(am, [n for n in xs if n in sd], [sd[n]["peak_dev_gib"] for n in xs if n in sd], "SDPA")
        gray_line(at, [n for n in xs if n in sd], [d.ratio(sd[n]) for n in xs if n in sd], "SDPA")
        n_axis(at, Ns)
        n_axis(am, Ns)
        ratio_axis(at, log=exp == "E2")
        ratio_line_label(at, Ns[0])
        gib_axis(am, log=True)
        at.set_title(f"{dname}: wall time relative to FlashAttention-2")
        am.set_title(f"{dname}: peak device memory")
        if exp != "E2":
            ylim_pad(at, lo=0.85)
    legend_below(fig, ncol=6)
    finish(fig, os.path.join(out, fname), title, subtitle, legend_h=0.08)


def fig_e2_pcie(d, out):
    """E2's 4M / 8M forward rows on the PCIe node, against that node's own baselines."""
    rows = d.rows("E2", node="pcie", direction="fwd")
    Ns = sorted({r["N"] for r in rows})
    groups = [nfmt(n) for n in Ns]
    bars = list(SERIES)
    vals_t = {nfmt(n): {} for n in Ns}
    vals_m = {nfmt(n): {} for n in Ns}
    for r in rows:
        if r["arm"] in SERIES:
            vals_t[nfmt(r["N"])][r["arm"]] = d.ratio(r)
            vals_m[nfmt(r["N"])][r["arm"]] = r["peak_dev_gib"]
    for r in rows:
        if r["arm"] in BASELINES:
            vals_m[nfmt(r["N"])][r["arm"]] = r["peak_dev_gib"]
    fig, (at, am) = plt.subplots(1, 2, figsize=(10, 4.2))
    grouped_bars(at, groups, bars, vals_t, COLOR, LABEL, label_fmt=lambda v: f"{v:.2f}x")
    ratio_axis(at)
    at.set_title("forward: wall time relative to FlashAttention-2 (same node)")
    at.set_xlabel("N (tokens)")
    grouped_bars(am, groups, bars + ["FA-2", "SDPA"], vals_m, COLOR, LABEL, label_fmt=lambda v: f"{v:.0f}")
    gib_axis(am)
    am.set_title("forward: peak device memory")
    am.set_xlabel("N (tokens)")
    legend_below(fig, ncol=6, ax=am)
    finish(fig, os.path.join(out, "E2_pcie.png"),
           "E2 at 4M and 8M tokens on the PCIe node: Q/K/V and the accumulator in host memory",
           "A100 80GB PCIe (gpu-short), fp16, B=1 H=8 D=64, causal, c=7, depth 1; its own FA-2 / SDPA rows "
           "(not comparable with the SXM figures)", legend_h=0.13)


def fig_c_sweep(d, out, exp, arms, title, subtitle, fname, fa2_mem=True):
    """E3 / E10: time ratio and peak device memory against c at 1M."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4))
    cs = sorted({r["c"] for r in d.rows(exp)})
    for row, (dr, dname) in enumerate(DIRS):
        at, am = axes[row]
        mem_vals = {}
        for arm in reversed(arms):
            rows = {r["c"]: r for r in d.rows(exp, arm=arm, direction=dr)}
            xs = [c for c in cs if c in rows]
            series_line(at, xs, [d.ratio(rows[c]) for c in xs], arm)
            series_line(am, xs, [rows[c]["peak_dev_gib"] for c in xs], arm)
            mem_vals[arm] = [rows[c]["peak_dev_gib"] for c in xs]
        coincide_note(am, mem_vals, " and ".join(LABEL[a] for a in arms) + " coincide")
        any_row = d.rows(exp, direction=dr)[0]
        b = d.baseline(any_row, "FA-2")
        sd = d.baseline(any_row, "SDPA")
        if b and fa2_mem:
            am.axhline(b["peak_dev_gib"], color=COLOR["FA-2"], lw=1.0, zorder=1)
            note(am, cs[-1], b["peak_dev_gib"], f"FA-2: {b['peak_dev_gib']:.1f} GiB", ha="right", dy=3, color=TEXT3)
        if sd and b:
            at.axhline(wall(sd) / wall(b), color=COLOR["SDPA"], lw=1.0, zorder=1)
            note(at, cs[0], wall(sd) / wall(b), f"SDPA {wall(sd) / wall(b):.2f}x", ha="left", dy=3, color=TEXT3)
        c_axis(at, cs)
        c_axis(am, cs)
        ratio_axis(at)
        ratio_line_label(at, cs[-1], ha="right")
        gib_axis(am)
        at.set_title(f"{dname}: wall time relative to FlashAttention-2")
        am.set_title(f"{dname}: peak device memory")
        ylim_pad(at, lo=0.85, hi=1.15)
        ylim_pad(am, lo=0)
    legend_below(fig, ncol=len(arms))
    finish(fig, os.path.join(out, fname), title, subtitle, legend_h=0.08)


def fig_e4(d, out):
    """E4: depth x accumulator placement at 1M, grouped bars per series."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4))
    cfgs = [(1, "gpu"), (1, "cpu"), (2, "gpu"), (2, "cpu")]
    groups = [f"depth {i}\n{'device' if a == 'gpu' else 'host'} acc." for i, a in cfgs]
    for row, (dr, dname) in enumerate(DIRS):
        at, am = axes[row]
        vt = {g: {} for g in groups}
        vm = {g: {} for g in groups}
        faded = set()
        for (i, a), g in zip(cfgs, groups):
            for r in d.rows("E4", direction=dr, itr=i, acc=a):
                vt[g][r["arm"]] = d.ratio(r)
                vm[g][r["arm"]] = r["peak_dev_gib"]
                if r.get("acc_ran") and r["acc_ran"] != r["acc"]:
                    faded.add((g, r["arm"]))        # the wave backward keeps its accumulator on the device
        grouped_bars(at, groups, list(SERIES), vt, COLOR, LABEL, label_fmt=lambda v: f"{v:.2f}",
                     label_sel=lambda g, b: b in ("classic_CUDA", "wave_CUDA"), faded=faded)
        ratio_axis(at)
        at.set_title(f"{dname}: wall time relative to FlashAttention-2")
        grouped_bars(am, groups, list(SERIES), vm, COLOR, LABEL, label_fmt=lambda v: f"{v:.1f}",
                     label_sel=lambda g, b: b in ("classic_CUDA", "wave_CUDA"), faded=faded)
        if faded:
            am.add_patch(Rectangle((0, 0), 0, 0, facecolor=COLOR["wave_CUDA"], alpha=0.35,
                                   label="faded: ran with the device accumulator (the wave backward keeps it there)"))
        b = d.baseline(d.rows("E4", direction=dr)[0], "FA-2")
        if b:
            am.axhline(b["peak_dev_gib"], color=COLOR["FA-2"], lw=1.0, zorder=1,
                       label=f"FlashAttention-2 ({b['peak_dev_gib']:.1f} GiB forward, {{bwd}} GiB forward+backward)"
                       if False else "FlashAttention-2 (one call)")
        gib_axis(am)
        am.set_title(f"{dname}: peak device memory")
    legend_below(fig, ncol=3)
    finish(fig, os.path.join(out, "E4.png"),
           "E4: the depth and where the accumulator (acc.) lives, at 1M tokens",
           "A100-SXM4-80GB, fp16, B=1 H=8 D=64, causal, c=7 (depth 2 = 49 subproblems), Q/K/V on the device; "
           "median of 3 runs", legend_h=0.08)


def fig_wave_budget(d, out, exp, title, subtitle, fname, directions=DIRS):
    """E5 / E12: the wave budget W (subproblems per launch) for the wave engine."""
    nrow = len(directions)
    fig, axes = plt.subplots(nrow, 2, figsize=(10, 3.9 * nrow + 0.4), squeeze=False)
    rows_all = d.rows(exp)
    ws = sorted({int(r["extra"][2:-1]) for r in rows_all})
    for row, (dr, dname) in enumerate(directions):
        at, am = axes[row]
        for arm in ("wave_CUDA", "wave_Triton"):
            rows = {int(r["extra"][2:-1]): r for r in d.rows(exp, arm=arm, direction=dr)}
            xs = [w for w in ws if w in rows]
            series_line(at, xs, [d.ratio(rows[w]) for w in xs], arm)
            series_line(am, xs, [rows[w]["peak_dev_gib"] for w in xs], arm)
            if arm == "wave_CUDA":
                for w, dy, va in ((xs[0], -8, "top"), (xs[-1], 7, "bottom")):
                    note(am, w, rows[w]["peak_dev_gib"], f"{rows[w]['peak_dev_gib']:.1f} GiB", dy=dy, va=va, size=8)
        waves = {int(r["extra"][2:-1]): r.get("n_waves") for r in d.rows(exp, arm="wave_CUDA", direction=dr)}
        for ax in (at, am):
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_locator(FixedLocator(ws))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v))}x\n({waves.get(int(round(v)), '?')} waves)"))
            ax.xaxis.set_minor_locator(FixedLocator([]))
            ax.set_xlabel("wave budget W (subproblems per launch; waves per call)")
        b = d.baseline(d.rows(exp, direction=dr)[0], "FA-2")
        if b:
            am.axhline(b["peak_dev_gib"], color=COLOR["FA-2"], lw=1.0, zorder=1)
            note(am, ws[-1], b["peak_dev_gib"], f"FA-2: {b['peak_dev_gib']:.1f} GiB", ha="right", dy=3, color=TEXT3)
        ratio_axis(at)
        ratio_line_label(at, ws[0])
        gib_axis(am)
        at.set_title(f"{dname}: wall time relative to FlashAttention-2")
        am.set_title(f"{dname}: peak device memory")
        ylim_pad(at, lo=0.9, hi=1.1)
        ylim_pad(am, lo=0)
    legend_below(fig, ncol=2)
    finish(fig, os.path.join(out, fname), title, subtitle, legend_h=0.08 if nrow == 2 else 0.15)


def fig_e6(d, out):
    """E6: precision x head dim at 1M; each bar against the FA-2 call of the same dtype and D."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4))
    cfgs = [("fp16", 64), ("bf16", 64), ("fp16", 128), ("bf16", 128)]
    groups = [f"{dt}, D={D}" for dt, D in cfgs]
    for row, (dr, dname) in enumerate(DIRS):
        at, am = axes[row]
        vt = {g: {} for g in groups}
        vm = {g: {} for g in groups}
        for (dt, D), g in zip(cfgs, groups):
            for r in d.rows("E6", direction=dr, dtype=dt, D=D):
                if r["arm"] in SERIES:
                    vt[g][r["arm"]] = d.ratio(r)
                vm[g][r["arm"]] = r["peak_dev_gib"]
        grouped_bars(at, groups, list(SERIES), vt, COLOR, LABEL, label_fmt=lambda v: f"{v:.2f}",
                     label_sel=lambda g, b: b in ("classic_CUDA", "wave_CUDA"))
        ratio_axis(at)
        at.set_title(f"{dname}: wall time / FA-2 of the same dtype and D")
        grouped_bars(am, groups, list(SERIES) + ["FA-2", "SDPA"], vm, COLOR, LABEL, label_fmt=lambda v: f"{v:.0f}",
                     label_sel=lambda g, b: b in ("classic_CUDA", "FA-2"))
        gib_axis(am)
        am.set_title(f"{dname}: peak device memory")
    legend_below(fig, ncol=6)
    finish(fig, os.path.join(out, "E6.png"),
           "E6: precision and head dimension at 1M tokens",
           "A100-SXM4-80GB, B=1 H=8, causal, c=7, depth 1, Q/K/V on the device; median of 3 runs; the Triton D=128 "
           "backward rows were excluded (not a finding)", legend_h=0.08)


def fig_e8(d, out):
    """E8 against E1: the host accumulator with device-resident Q/K/V, across N."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4))
    Ns = sorted({r["N"] for r in d.rows("E8")})
    for row, (dr, dname) in enumerate(DIRS):
        at, am = axes[row]
        for arm in ("classic_CUDA", "wave_CUDA"):
            for exp, filled, ls, tag in (("E1", True, "-", "device accumulator"), ("E8", False, "--", "host accumulator")):
                rows = {r["N"]: r for r in d.rows(exp, arm=arm, direction=dr, node="sxm")}
                xs = [n for n in Ns if n in rows]
                series_line(at, xs, [d.ratio(rows[n]) for n in xs], arm, ls, filled, label=f"{LABEL[arm]}, {tag}")
                series_line(am, xs, [rows[n]["peak_dev_gib"] for n in xs], arm, ls, filled, label=f"{LABEL[arm]}, {tag}")
                for n in xs:
                    it = rows[n].get("itr_ran", 1)
                    if it and it > 1:
                        note_arrow(at, n, d.ratio(rows[n]), f"depth {it}", -22, 18 if exp == "E1" else -18)
        base = {r["N"]: r for r in d.rows("E1", arm="FA-2", direction=dr, node="sxm")}
        xs = [n for n in Ns if n in base]
        gray_line(am, xs, [base[n]["peak_dev_gib"] for n in xs], "FA-2")
        n_axis(at, Ns)
        n_axis(am, Ns)
        ratio_axis(at)
        ratio_line_label(at, Ns[0])
        gib_axis(am, log=True)
        at.set_title(f"{dname}: wall time relative to FlashAttention-2")
        am.set_title(f"{dname}: peak device memory")
        ylim_pad(at, lo=0.85, hi=1.4)
    legend_below(fig, ncol=3)
    finish(fig, os.path.join(out, "E8.png"),
           "E8: the default's trade from 1M up, the accumulator in host memory with Q/K/V on the device",
           "A100-SXM4-80GB, fp16, B=1 H=8 D=64, causal, c=7, the smallest depth that fits; "
           "device-accumulator rows from E1", legend_h=0.10)


def fig_e9(d, out):
    """E9: the chunk pool of the host-resident forward: pool settings per N, at D=64 and D=128."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4))
    Ns = [256 * K, M, 4 * M]
    settings = [("classic_CUDA", "pool=flush"), ("classic_CUDA", "pool=3"), ("classic_CUDA", "pool=4"),
                ("classic_CUDA", "pool=7"), ("wave_CUDA", "pool=3"), ("wave_CUDA", "pool=7")]
    names = {("classic_CUDA", "pool=flush"): "classic: no pool", ("classic_CUDA", "pool=3"): "classic: 3 slots (default)",
             ("classic_CUDA", "pool=4"): "classic: 4 slots", ("classic_CUDA", "pool=7"): "classic: 7 slots = all chunks",
             ("wave_CUDA", "pool=3"): "wave: one wave's chunks (default)", ("wave_CUDA", "pool=7"): "wave: all chunks"}
    colors = dict(zip(settings, BLUES + ORANGES[1:]))
    keys = [f"{a}|{e}" for a, e in settings]
    colors_k = {f"{a}|{e}": colors[(a, e)] for a, e in settings}
    names_k = {f"{a}|{e}": names[(a, e)] for a, e in settings}
    for row, D in enumerate((64, 128)):
        at, am = axes[row]
        groups = [nfmt(n) for n in Ns]
        vt = {g: {} for g in groups}
        vm = {g: {} for g in groups}
        default = {r["N"]: wall(r) for r in d.rows("E9", arm="classic_CUDA", extra="pool=3", D=D)}
        for a, e in settings:
            for r in d.rows("E9", arm=a, extra=e, D=D):
                vt[nfmt(r["N"])][f"{a}|{e}"] = wall(r) / default[r["N"]]
                vm[nfmt(r["N"])][f"{a}|{e}"] = r["peak_dev_gib"]
        grouped_bars(at, groups, keys, vt, colors_k, names_k, label_fmt=lambda v: f"{v:.2f}",
                     label_sel=lambda g, b: b.endswith("pool=flush") or b == "wave_CUDA|pool=3")
        ratio_axis(at, label="wall time / the default (classic, 3 slots)")
        at.set_title(f"D={D}, forward: wall time relative to the default pool")
        at.set_xlabel("N (tokens)")
        grouped_bars(am, groups, keys, vm, colors_k, names_k, label_fmt=lambda v: f"{v:.1f}",
                     label_sel=lambda g, b: b in ("classic_CUDA|pool=3", "classic_CUDA|pool=7"))
        for n in Ns:
            rr = d.rows("E9", arm="classic_CUDA", extra="pool=3", D=D, N=n)
            if rr:
                b = d.baseline(rr[0], "FA-2")
                if b:
                    gi = groups.index(nfmt(n))
                    am.plot([gi - 0.42, gi + 0.42], [b["peak_dev_gib"]] * 2, color=COLOR["FA-2"], lw=1.2, zorder=4)
        am.plot([], [], color=COLOR["FA-2"], lw=1.2, label="FA-2, device-resident")
        gib_axis(am)
        am.set_title(f"D={D}, forward: peak device memory")
        am.set_xlabel("N (tokens)")
    legend_below(fig, ncol=4)
    finish(fig, os.path.join(out, "E9.png"),
           "E9: chunk reuse in the host-resident forward, the device chunk pool",
           "A100-SXM4-80GB, fp16, B=1 H=8, causal, c=7, depth 1, Q/K/V and the accumulator in host memory; "
           "1M rows are the median of 3 runs", legend_h=0.11)


def fig_e11(d, out):
    """E11: the streamed backward's scatter, synchronous vs asynchronous, at 1M and 2M."""
    rows = d.rows("E11")
    Ns = sorted({r["N"] for r in rows})
    groups = [nfmt(n) for n in Ns]
    keys = ["scatter=sync", "scatter=async"]
    names = {"scatter=sync": "synchronous scatter (CQSA_SCATTER_ASYNC=0)", "scatter=async": "asynchronous fold (default since E11)"}
    colors = {"scatter=sync": BLUES[0], "scatter=async": BLUES[2]}
    vt = {g: {} for g in groups}
    vm = {g: {} for g in groups}
    for r in rows:
        vt[nfmt(r["N"])][r["extra"]] = d.ratio(r)
        vm[nfmt(r["N"])][r["extra"]] = r["peak_dev_gib"]
    fig, (at, am) = plt.subplots(1, 2, figsize=(10, 4.2))
    grouped_bars(at, groups, keys, vt, colors, names, label_fmt=lambda v: f"{v:.2f}x")
    ratio_axis(at)
    at.set_title("forward + backward: wall time relative to FlashAttention-2")
    at.set_xlabel("N (tokens)")
    for g, n in zip(groups, Ns):
        s, a = vt[g].get("scatter=sync"), vt[g].get("scatter=async")
        if s and a:
            note(at, groups.index(g), max(s, a), f"async / sync = {a / s:.2f}", dy=16, color=TEXT2)
    grouped_bars(am, groups, keys, vm, colors, names, label_fmt=lambda v: f"{v:.1f}")
    gib_axis(am)
    am.set_title("forward + backward: peak device memory")
    am.set_xlabel("N (tokens)")
    ylim_pad(at, hi=max(v for g in vt.values() for v in g.values()) * 1.25)
    legend_below(fig, ncol=2, ax=at)
    finish(fig, os.path.join(out, "E11.png"),
           "E11: the host-streamed backward's gradient scatter",
           "classic CUDA, A100-SXM4-80GB, fp16, B=1 H=8 D=64, causal, c=7, depth 1, Q/K/V and the accumulator in host "
           "memory; median of 3 runs", legend_h=0.13)


def fig_e0(d, out):
    """E0 / E0b: the error against float64 relative to FA-2's own error, every series x c x depth x accumulator."""
    fig, (ao, ag) = plt.subplots(1, 2, figsize=(10, 4.9))
    cfgs = [(3, 1), (3, 2), (7, 1), (7, 2), (31, 1), (31, 2), (133, 1)]
    labels = [f"c={c}\ndepth {i}" for c, i in cfgs]
    off = {"classic_CUDA": -0.27, "wave_CUDA": -0.09, "classic_Triton": 0.09, "wave_Triton": 0.27}
    for ax, dr, field, ttl in ((ao, "fwd", "fp64_ratio_fa2", "output: error vs float64, relative to FA-2's own"),
                               (ag, "bwd", "fp64_dq_ratio_fa2", "gradients: worst of dQ/dK/dV vs float64, relative to FA-2's own")):
        for exp in ("E0", "E0b"):
            for r in d.rows(exp, direction=dr, dtype="fp16", D=64):
                if r["c"] is None or (r["c"], r["itr"]) not in cfgs or r.get(field) is None:
                    continue
                x = cfgs.index((r["c"], r["itr"])) + off[r["arm"]]
                ax.plot([x], [r[field]], MARKER[r["arm"]], color=COLOR[r["arm"]], markerfacecolor=COLOR[r["arm"]],
                        markeredgecolor=SURFACE, markersize=6.5, zorder=3)
        ax.axhline(1.0, color=COLOR["FA-2"], lw=1.0, zorder=1)
        note(ax, -0.4, 1.0, "FA-2's own error = 1.0", ha="left", dy=-3, va="top", color=TEXT3)
        ax.set_xticks(range(len(cfgs)))
        ax.set_xticklabels(labels)
        ax.xaxis.grid(False)
        ax.set_ylabel("error / FlashAttention-2's error")
        ax.set_title(ttl)
        ax.set_ylim(0.85, 1.3)
    handles = [plt.Line2D([], [], color=COLOR[a], marker=MARKER[a], ls="", markeredgecolor=SURFACE, label=LABEL[a]) for a in SERIES]
    legend_below(fig, (handles, [h.get_label() for h in handles]), ncol=4)
    finish(fig, os.path.join(out, "E0.png"),
           "E0 / E0b: every path is exact",
           "256K tokens, A100-SXM4-80GB, fp16, B=1 H=8 D=64, causal; the error of 128 sampled rows against a float64 "
           "reference, divided by\nFlashAttention-2's own error on the same rows. One point per row: device or host "
           "accumulator, device- or host-resident Q/K/V\n(E0: 112 rows at D=64 fp16; E0b: 14); the points of one "
           "series at one x lie on top of each other.", legend_h=0.15)


FIGS = {
    "headline": fig_headline,
    "E0": fig_e0,
    "E1": lambda d, out: fig_scaling(d, out, "E1", "E1: scaling with Q/K/V on the device",
                                     "A100-SXM4-80GB, fp16, B=1 H=8 D=64, causal, c=7, the smallest depth that fits "
                                     "(7 subproblems; 49 at depth 2); one timed run after a warm-up", "E1.png"),
    "E2": lambda d, out: fig_scaling(d, out, "E2", "E2: scaling with Q/K/V and the accumulator in host memory",
                                     "A100-SXM4-80GB, fp16, B=1 H=8 D=64, causal, c=7, depth 1; the baselines are E1's "
                                     "device-resident FA-2 / SDPA calls", "E2.png"),
    "E2_pcie": fig_e2_pcie,
    "E3": lambda d, out: fig_c_sweep(d, out, "E3", SERIES, "E3: the quorum size c at 1M tokens, Q/K/V on the device",
                                     "A100-SXM4-80GB, fp16, B=1 H=8 D=64, causal, depth 1 (c subproblems of 3..13 "
                                     "chunks), device accumulator; median of 3 runs", "E3.png"),
    "E4": fig_e4,
    "E5": lambda d, out: fig_wave_budget(d, out, "E5", "E5: the wave budget at 1M tokens, c=73, Q/K/V on the device",
                                         "wave engine, A100-SXM4-80GB, fp16, B=1 H=8 D=64, causal, depth 1 (73 "
                                         "subproblems), device accumulator; median of 3 runs", "E5.png"),
    "E6": fig_e6,
    "E8": fig_e8,
    "E9": fig_e9,
    "E10": lambda d, out: fig_c_sweep(d, out, "E10", ("classic_CUDA", "wave_CUDA"),
                                      "E10: the quorum size c at 1M tokens, Q/K/V and the accumulator in host memory",
                                      "A100-SXM4-80GB, fp16, B=1 H=8 D=64, causal, depth 1; the baseline is E1's "
                                      "device-resident FA-2 call; median of 3 runs.\nThe forward+backward wave rows "
                                      "ran the classic backward (the wave backward is device-resident only).", "E10.png"),
    "E11": fig_e11,
    "E12": lambda d, out: fig_wave_budget(d, out, "E12",
                                          "E12: the wave budget at 1M tokens, c=73, Q/K/V and the accumulator in host memory",
                                          "wave engine, A100-SXM4-80GB, fp16, B=1 H=8 D=64, causal, depth 1; the "
                                          "device chunk pool holds one wave's chunks; median of 3 runs", "E12.png",
                                          directions=DIRS[:1]),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(ROOT, "results", "v3"))
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--only", nargs="*", default=None, help="figure ids (default: all)")
    a = ap.parse_args()
    d = Data(a.results)
    for name, fn in FIGS.items():
        if a.only and name not in a.only:
            continue
        fn(d, a.out)


if __name__ == "__main__":
    main()
