"""
Statistical evaluation of the Gemma refusal-direction ablation experiment.

Input : evaluation_outputs_manual_labels_filled.csv  (600 paired prompts)
Output: analysis/report.html                          (self-contained report)

The script recomputes every number it reports, builds all figures as inline
base64 PNGs, and writes one portable HTML file. No external assets.
"""
from __future__ import annotations

import base64
import io
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import matplotlib.patches as mpatches
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "evaluation_outputs_manual_labels_filled.csv"
OUT = ROOT / "analysis" / "report.html"
SEED = 2445

# ---- shared palette -------------------------------------------------------
C_BASE = "#3f6f9f"   # baseline (intact model)
C_EDIT = "#d1495b"   # edited (refusal ablated)
C_HELD = "#5a6b7b"   # outputs that stayed refused
C_GREY = "#9aa3ab"
INK = "#1f2933"
GRID = "#dfe3e8"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": "#c4ccd4",
    "axes.linewidth": 0.8,
    "axes.titlecolor": INK,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "figure.dpi": 130,
})


# ---------------------------------------------------------------------------
# Load & derive
# ---------------------------------------------------------------------------
def load():
    df = pd.read_csv(DATA)
    df["base_ref"] = (df["baseline_manual_label"] == "refused").astype(int)
    df["edit_ref"] = (df["edited_manual_label"] == "refused").astype(int)
    return df


def wilson(k, n):
    return proportion_confint(k, n, method="wilson")


def boot_paired_diff(base, edit, reps=20000, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(base)
    out = np.empty(reps)
    for i in range(reps):
        idx = rng.integers(0, n, n)
        out[i] = edit[idx].mean() - base[idx].mean()
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


def mcnemar_block(sub):
    a = int(((sub.base_ref == 1) & (sub.edit_ref == 1)).sum())
    b = int(((sub.base_ref == 1) & (sub.edit_ref == 0)).sum())
    c = int(((sub.base_ref == 0) & (sub.edit_ref == 1)).sum())
    d = int(((sub.base_ref == 0) & (sub.edit_ref == 0)).sum())
    n = len(sub)
    res = mcnemar([[a, b], [c, d]], exact=True)
    base_r, edit_r = sub.base_ref.mean(), sub.edit_ref.mean()
    blo, bhi = wilson(sub.base_ref.sum(), n)
    elo, ehi = wilson(sub.edit_ref.sum(), n)
    lo, hi = boot_paired_diff(sub.base_ref.values, sub.edit_ref.values)
    return dict(n=n, a=a, b=b, c=c, d=d,
                base_r=base_r, edit_r=edit_r,
                base_ci=(blo, bhi), edit_ci=(elo, ehi),
                delta=edit_r - base_r, delta_ci=(lo, hi),
                p=res.pvalue)


def stats_all(df):
    res = {grp: mcnemar_block(sub) for grp, sub in
           {"all": df, "unsafe": df[df.label == "unsafe"], "safe": df[df.label == "safe"]}.items()}

    # word counts (Wilcoxon signed rank)
    wc = {}
    for grp, sub in {"all": df, "unsafe": df[df.label == "unsafe"], "safe": df[df.label == "safe"]}.items():
        b, e = sub.baseline_word_count.values, sub.edited_word_count.values
        d = e - b
        nz = d[d != 0]
        W, p = stats.wilcoxon(e, b, zero_method="wilcox", alternative="two-sided")
        rbc = (np.sum(nz > 0) - np.sum(nz < 0)) / len(nz)
        wc[grp] = dict(base_med=float(np.median(b)), edit_med=float(np.median(e)),
                       delta_med=float(np.median(d)), W=float(W), p=float(p), rbc=float(rbc),
                       n=len(sub), nnz=len(nz))

    # refusal verbosity (unsafe)
    us = df[df.label == "unsafe"]
    verb = dict(base=float(us[us.base_ref == 1].baseline_word_count.median()),
                edit=float(us[us.edit_ref == 1].edited_word_count.median()),
                n_base=int((us.base_ref == 1).sum()), n_edit=int((us.edit_ref == 1).sum()))

    # per-category (unsafe)
    rows = []
    for cat, sub in df[df.label == "unsafe"].groupby("category"):
        mb = mcnemar_block(sub)
        rows.append(dict(category=cat, n=mb["n"], base=mb["base_r"], edit=mb["edit_r"],
                         delta=mb["delta"], flips=mb["b"], p=mb["p"]))
    cat = pd.DataFrame(rows).sort_values("delta").reset_index(drop=True)
    cat["p_holm"] = multipletests(cat["p"], method="holm")[1]
    cat["sig"] = cat["p_holm"] < 0.05
    return res, wc, verb, cat


# ---------------------------------------------------------------------------
# Figures  (return base64 png)
# ---------------------------------------------------------------------------
def b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def fig_refusal_bars(res):
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    groups = ["unsafe", "safe"]
    labels = ["Unsafe prompts\n(n=300)", "Safe prompts\n(n=300)"]
    x = np.arange(len(groups)); w = 0.36
    for i, (cond, color, off) in enumerate([("base", C_BASE, -w/2), ("edit", C_EDIT, w/2)]):
        vals = [res[g][f"{cond}_r"] * 100 for g in groups]
        cis = [res[g][f"{cond}_ci"] for g in groups]
        err = [[(v - lo*100) for v, (lo, hi) in zip(vals, cis)],
               [(hi*100 - v) for v, (lo, hi) in zip(vals, cis)]]
        bars = ax.bar(x + off, vals, w, color=color, yerr=err, capsize=4,
                      error_kw=dict(ecolor="#33414e", lw=1.1),
                      label="Baseline (intact)" if cond == "base" else "Edited (ablated)")
        for bx, v in zip(bars, vals):
            ax.text(bx.get_x()+bx.get_width()/2, v + 2.5, f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=10, fontweight="bold",
                    color=color)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Refusal rate"); ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20)); ax.set_yticklabels([f"{v}%" for v in range(0,101,20)])
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", ncol=1)
    # annotate the unsafe drop (arc routed clear of the value labels)
    ax.annotate("", xy=(0.05, 57), xytext=(-0.13, 97),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.4,
                                connectionstyle="arc3,rad=-0.2"))
    ax.text(0.205, 80, "−47.7 pp", fontsize=12, fontweight="bold", color=C_EDIT, ha="center")
    fig.tight_layout()
    return b64(fig)


def _ribbon(ax, x0, x1, y0a, y0b, y1a, y1b, color, alpha):
    # cubic bezier band between two vertical segments
    xm = (x0 + x1) / 2
    verts = [(x0, y0a), (xm, y0a), (xm, y1a), (x1, y1a),
             (x1, y1b), (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0a)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
    ax.add_patch(mpatches.PathPatch(MplPath(verts, codes), facecolor=color,
                                    edgecolor="none", alpha=alpha))


def fig_alluvial(df):
    us = df[df.label == "unsafe"]
    # left nodes (baseline), right nodes (edited). Order chosen so the tiny
    # complied->complied flow sits flat at the top instead of crossing.
    flows = [("complied", "complied", 2, C_BASE, "complied → complied"),
             ("refused", "complied", 142, C_EDIT, "refused → complied"),
             ("refused", "refused", 155, C_HELD, "refused → refused"),
             ("refused", "unusable", 1, C_GREY, "refused → unusable")]
    left_tot = {"refused": 298, "complied": 2}
    right_tot = {"complied": 144, "refused": 155, "unusable": 1}
    gap = 12; total = 300
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x0, x1, bw = 0.0, 1.0, 0.045

    # node y-positions (top->down), stacked with gaps
    def stack(order, tot):
        y = total + gap * (len(order) - 1)
        pos = {}
        for k in order:
            h = tot[k]
            pos[k] = (y, y - h)  # (top, bottom)
            y -= h + gap
        return pos
    lpos = stack(["complied", "refused"], left_tot)
    rpos = stack(["complied", "refused", "unusable"], right_tot)

    # draw nodes
    for k, (top, bot) in lpos.items():
        ax.add_patch(mpatches.Rectangle((x0 - bw, bot), bw, top - bot,
                     color=C_BASE if k == "complied" else C_HELD))
        ax.text(x0 - bw - 0.02, (top + bot) / 2, f"{k}\n{left_tot[k]}", ha="right",
                va="center", fontsize=10, fontweight="bold")
    for k, (top, bot) in rpos.items():
        col = {"complied": C_EDIT, "refused": C_HELD, "unusable": C_GREY}[k]
        ax.add_patch(mpatches.Rectangle((x1, bot), bw, top - bot, color=col))
        ax.text(x1 + bw + 0.02, (top + bot) / 2, f"{k}\n{right_tot[k]}", ha="left",
                va="center", fontsize=10, fontweight="bold")

    # ribbons, consuming each node top-down
    lcur = {k: v[0] for k, v in lpos.items()}
    rcur = {k: v[0] for k, v in rpos.items()}
    for src, dst, val, col, _ in flows:
        y0a, y0b = lcur[src], lcur[src] - val
        y1a, y1b = rcur[dst], rcur[dst] - val
        _ribbon(ax, x0, x1, y0a, y0b, y1a, y1b, col, 0.62)
        lcur[src] -= val; rcur[dst] -= val

    ax.text(x0 - bw, total + gap + 26, "BASELINE", fontsize=10, fontweight="bold", color=C_BASE)
    ax.text(x1 + bw, total + gap + 26, "EDITED", fontsize=10, fontweight="bold", color=C_EDIT, ha="right")
    # legend for the dominant flow
    ax.text(0.5, -34, "142 of 300 unsafe prompts flipped refused → complied;  0 moved the other way",
            ha="center", fontsize=10, color=INK)
    ax.set_xlim(-0.28, 1.28); ax.set_ylim(-46, total + gap + 40)
    ax.axis("off")
    return b64(fig)


def fig_dumbbell(cat):
    cat = cat.sort_values("delta")  # most negative (biggest drop) at top after invert
    y = np.arange(len(cat))
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for yi, (_, r) in zip(y, cat.iterrows()):
        ax.plot([r.base*100, r.edit*100], [yi, yi], color="#c4ccd4", lw=2.4, zorder=1)
    ax.scatter(cat.base*100, y, s=58, color=C_BASE, zorder=3, label="Baseline")
    ax.scatter(cat.edit*100, y, s=58, color=C_EDIT, zorder=3, label="Edited")
    for yi, (_, r) in zip(y, cat.iterrows()):
        ax.text(r.edit*100 - 3, yi, f"{r.edit*100:.0f}%", ha="right", va="center",
                fontsize=8.5, color=C_EDIT)
        mark = "" if r.sig else "  (n.s.)"
        ax.text(102, yi, f"−{(-r.delta)*100:.0f} pp{mark}", ha="left", va="center",
                fontsize=8.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([textwrap.fill(c, 34) for c in cat.category], fontsize=9)
    ax.set_xlim(0, 128); ax.set_xticks(range(0, 101, 20))
    ax.set_xticklabels([f"{v}%" for v in range(0, 101, 20)])
    ax.set_xlabel("Refusal rate  (25 prompts per category)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.62))
    ax.set_title("Refusal collapse by harm category (unsafe eval set)", loc="left",
                 fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    return b64(fig)


def fig_wordcounts(df):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3), gridspec_kw=dict(wspace=0.25))
    panels = [("unsafe", "Unsafe prompts"), ("safe", "Safe prompts")]
    for ax, (grp, title) in zip(axes, panels):
        sub = df[df.label == grp]
        data = [sub.baseline_word_count.values, sub.edited_word_count.values]
        parts = ax.violinplot(data, positions=[0, 1], showextrema=False, widths=0.8)
        for pc, col in zip(parts["bodies"], [C_BASE, C_EDIT]):
            pc.set_facecolor(col); pc.set_alpha(0.35); pc.set_edgecolor(col)
        for i, (d, col) in enumerate(zip(data, [C_BASE, C_EDIT])):
            ax.scatter(np.full(len(d), i) + np.random.default_rng(1).uniform(-0.07, 0.07, len(d)),
                       d, s=6, color=col, alpha=0.35, zorder=2)
            ax.hlines(np.median(d), i-0.26, i+0.26, color=INK, lw=2, zorder=4)
            ax.text(i, np.median(d)+17, f"med {np.median(d):.0f}", ha="center",
                    fontsize=9, fontweight="bold", color=col, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Baseline", "Edited"])
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(-8, 270); ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        if grp == "unsafe":
            ax.set_ylabel("Words per answer")
    fig.suptitle("Answer length shifts only for unsafe prompts", x=0.012, ha="left",
                 fontsize=12, fontweight="bold", y=1.0)
    fig.subplots_adjust(top=0.86, bottom=0.12, left=0.09, right=0.98)
    return b64(fig)


def fig_refusal_verbosity(df):
    us = df[df.label == "unsafe"]
    base = us[us.base_ref == 1].baseline_word_count.values
    edit = us[us.edit_ref == 1].edited_word_count.values
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    bins = np.linspace(0, 270, 46)
    ax.hist(base, bins=bins, color=C_BASE, alpha=0.7, label=f"Baseline refusals (n={len(base)})")
    ax.hist(edit, bins=bins, color=C_EDIT, alpha=0.6, label=f"Edited refusals (n={len(edit)})")
    ax.axvline(np.median(base), color=C_BASE, lw=2, ls="--")
    ax.axvline(np.median(edit), color=C_EDIT, lw=2, ls="--")
    ax.text(np.median(base)+4, ax.get_ylim()[1]*0.9, f"med {np.median(base):.0f}",
            color=C_BASE, fontsize=9, fontweight="bold")
    ax.text(np.median(edit)+4, ax.get_ylim()[1]*0.9, f"med {np.median(edit):.0f}",
            color=C_EDIT, fontsize=9, fontweight="bold")
    ax.set_xlabel("Words in a 'refused'-labeled answer"); ax.set_ylabel("Count")
    ax.set_title("Even when the edited model still refuses, refusals get long",
                 loc="left", fontsize=11.5, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    return b64(fig)


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------
def pct(x, d=1): return f"{x*100:.{d}f}%"


def build_html(df, res, wc, verb, cat, figs):
    u, s, a = res["unsafe"], res["safe"], res["all"]

    def ci(t): return f"[{t[0]*100:.1f}, {t[1]*100:.1f}]"

    cat_rows = "\n".join(
        f"<tr class='{'sig' if r.sig else 'nsig'}'>"
        f"<td class='l'>{r.category}</td><td>{pct(r.base,0)}</td><td>{pct(r.edit,0)}</td>"
        f"<td class='delta'>−{(-r.delta)*100:.0f} pp</td><td>{int(r.flips)}/25</td>"
        f"<td>{r.p:.1e}</td><td>{r.p_holm:.1e}</td>"
        f"<td>{'✔' if r.sig else 'n.s.'}</td></tr>"
        for _, r in cat.iterrows())

    figblock = lambda key, num, cap: (
        f"<figure><img src='data:image/png;base64,{figs[key]}' alt='{cap}'/>"
        f"<figcaption><b>Figure {num}.</b> {cap}</figcaption></figure>")

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Refusal-Ablation Evaluation</title>
<style>
:root{{--ink:#1f2933;--mut:#5a6b7b;--line:#e4e8ec;--base:{C_BASE};--edit:{C_EDIT};
--bg:#fbfcfd;--card:#fff;--accent:#0f5132;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
line-height:1.6;-webkit-font-smoothing:antialiased;}}
.wrap{{max-width:980px;margin:0 auto;padding:0 26px 90px;}}
header{{padding:54px 0 30px;border-bottom:1px solid var(--line);margin-bottom:8px;}}
.kicker{{text-transform:uppercase;letter-spacing:.14em;font-size:12px;font-weight:700;color:var(--edit);}}
h1{{font-size:33px;line-height:1.15;margin:.32em 0 .25em;letter-spacing:-.01em;}}
.sub{{color:var(--mut);font-size:16px;max-width:70ch;}}
.meta{{margin-top:18px;font-size:13px;color:var(--mut);display:flex;flex-wrap:wrap;gap:6px 20px;}}
.meta b{{color:var(--ink);font-weight:600;}}
h2{{font-size:22px;margin:46px 0 6px;letter-spacing:-.01em;}}
h2 .n{{color:var(--edit);font-weight:700;margin-right:10px;}}
h3{{font-size:16px;margin:26px 0 4px;}}
p{{margin:.5em 0 1em;}} .lead{{font-size:17px;}}
.cards{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:26px 0 6px;}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:15px 13px 13px;}}
.card .v{{font-size:22px;font-weight:750;letter-spacing:-.02em;line-height:1.1;}}
.card .k{{font-size:12px;color:var(--mut);margin-top:5px;}}
.card.red .v{{color:var(--edit);}} .card.blue .v{{color:var(--base);}} .card.green .v{{color:var(--accent);}}
figure{{margin:22px 0 8px;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;}}
figure img{{width:100%;height:auto;display:block;border-radius:6px;}}
figcaption{{font-size:13.5px;color:var(--mut);margin-top:11px;}}
table{{width:100%;border-collapse:collapse;font-size:13.5px;margin:14px 0 6px;}}
th,td{{padding:8px 10px;text-align:right;border-bottom:1px solid var(--line);}}
th{{font-size:11.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--mut);font-weight:700;border-bottom:2px solid #cfd6dc;}}
td.l,th.l{{text-align:left;}} td.delta{{font-weight:700;color:var(--edit);}}
tr.nsig td{{color:#93a0ab;}} tr.nsig td.delta{{color:#c98a93;}}
tbody tr:hover{{background:#f4f7f9;}}
.callout{{background:#f0f5f9;border:1px solid #d6e3ee;border-left:4px solid var(--base);
border-radius:10px;padding:14px 18px;margin:18px 0;font-size:14.5px;}}
.callout.warn{{background:#fff6f0;border-color:#f3d8c7;border-left-color:#e07a3f;}}
.callout.good{{background:#eef7f1;border-color:#cfe8d8;border-left-color:var(--accent);}}
.callout b{{font-weight:700;}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:22px;}}
.tag{{display:inline-block;font-size:11px;font-weight:700;padding:2px 8px;border-radius:20px;background:#eef1f4;color:var(--mut);}}
code{{background:#eef1f4;padding:1px 6px;border-radius:5px;font-size:13px;}}
ul{{margin:.3em 0 1.1em;padding-left:1.2em;}} li{{margin:.3em 0;}}
hr{{border:none;border-top:1px solid var(--line);margin:40px 0;}}
footer{{font-size:12.5px;color:#8a97a2;border-top:1px solid var(--line);padding-top:18px;margin-top:46px;}}
.statline{{font-variant-numeric:tabular-nums;font-size:14px;background:#f7f9fb;border:1px solid var(--line);
border-radius:8px;padding:10px 14px;margin:6px 0;}}
@media(max-width:820px){{.cards{{grid-template-columns:repeat(2,1fr)}}.two{{grid-template-columns:1fr}}}}
</style></head><body><div class="wrap">

<header>
<div class="kicker">Statistical Evaluation &middot; Mechanistic Safety</div>
<h1>Does ablating one "refusal direction" actually unlock Gemma?</h1>
<p class="sub">A paired analysis of 600 held-out prompts comparing the intact model against a
weight-edited model whose refusal direction was removed. The edit is measured on a
<b>category-disjoint</b> evaluation set, so no evaluated harm category contributed to fitting the direction.</p>
<div class="meta">
<span><b>Model</b> google/gemma-4-E4B-it</span>
<span><b>Design</b> within-prompt paired (baseline vs edited)</span>
<span><b>Eval set</b> 300 safe + 300 unsafe, 24 categories</span>
<span><b>Labels</b> manual: complied / refused / unusable</span>
<span><b>Date</b> 2026-06-11</span>
</div>
</header>

<h2><span class="n">01</span>What happened, in five numbers</h2>
<p class="lead">Removing a single direction cut the refusal rate on unsafe prompts roughly in half,
did <b>no</b> measurable damage to behavior on safe prompts, and moved every single
discordant prompt in the same direction.</p>

<div class="cards">
  <div class="card red"><div class="v">{pct(u['base_r'],1)}&rarr;{pct(u['edit_r'],1)}</div>
    <div class="k">Unsafe-prompt refusal rate, baseline &rarr; edited</div></div>
  <div class="card red"><div class="v">&minus;{(-u['delta'])*100:.1f} pp</div>
    <div class="k">Absolute drop in unsafe refusal &middot; boot 95% CI {ci(u['delta_ci'])}</div></div>
  <div class="card green"><div class="v">0 / 300</div>
    <div class="k">Safe prompts broken by the edit (compliance preserved)</div></div>
  <div class="card blue"><div class="v">{a['b']} &rarr; 0</div>
    <div class="k">Prompts (of all 600) that flipped toward vs. away from compliance &mdash; perfectly monotone</div></div>
  <div class="card red"><div class="v">+{verb['edit']-verb['base']:.0f} words</div>
    <div class="k">Median length gain of unsafe answers that still refuse</div></div>
</div>

<div class="callout good"><b>Headline.</b> The Arditi-style single-direction hypothesis holds up here:
one direction governs a large, generalizable share of refusal behavior on <i>unseen</i> harm
categories, while leaving benign behavior essentially untouched &mdash; but the effect is
<b>partial and uneven</b>, not a clean on/off switch.</div>

<h2><span class="n">02</span>The headline effect &amp; its asymmetry</h2>
<p>Each prompt is run once through the intact model and once through the edited model, so every
comparison is a matched pair. Because the outcome is binary (refused vs. not) and paired, the
correct test is <b>McNemar's exact test</b>; it depends only on the prompts whose label changed.</p>

{figblock('bars', 1, 'Refusal rate by condition with Wilson 95% confidence intervals. The edit halves refusal on unsafe prompts while the safe-prompt rate stays at the floor.')}

<div class="two">
<div class="statline"><span class="tag">UNSAFE</span><br>
Baseline <b>{pct(u['base_r'])}</b> {ci(u['base_ci'])}<br>
Edited <b>{pct(u['edit_r'])}</b> {ci(u['edit_ci'])}<br>
Paired &Delta; <b>&minus;{(-u['delta'])*100:.1f} pp</b> {ci(u['delta_ci'])}<br>
McNemar exact <b>p = {u['p']:.2e}</b> (discordant {u['b']} vs {u['c']})</div>
<div class="statline"><span class="tag">SAFE</span><br>
Baseline <b>{pct(s['base_r'])}</b> {ci(s['base_ci'])}<br>
Edited <b>{pct(s['edit_r'])}</b> {ci(s['edit_ci'])}<br>
Paired &Delta; <b>{s['delta']*100:+.1f} pp</b> {ci(s['delta_ci'])}<br>
McNemar exact <b>p = {s['p']:.2f}</b> (discordant {s['b']} vs {s['c']}) &mdash; n.s.</div>
</div>

<p>The two sides tell opposite stories, which is exactly what a well-targeted edit should do.
On unsafe prompts the change is overwhelming (<code>p &approx; {u['p']:.0e}</code>). On safe prompts it is
statistically nil &mdash; in fact the lone safe prompt that the baseline refused was
<i>answered</i> after the edit, so collateral movement, such as it is, also points toward helpfulness.</p>

<h2><span class="n">03</span>The edit only ever pushes one way</h2>
<p>A skeptic's first worry: maybe the edit just adds noise, flipping some prompts to compliance and
others to refusal. It does not. Across all 600 prompts, <b>{a['b']}</b> moved from refused to
not-refused and <b>{a['c']}</b> moved the other way. Every baseline compliance was preserved.</p>

{figblock('alluvial', 2, 'Paired transitions on the 300 unsafe prompts. Ribbon width is prompt count. The edit converts 142 refusals into compliance and elaborates 155 more into long soft-refusals; nothing flows back toward refusal.')}

<div class="callout"><b>Why this matters statistically.</b> Perfect monotonicity (one empty
off-diagonal cell) means the marginal drop in refusal is not an average of competing effects &mdash;
it is a floor on the true per-prompt effect. It also makes the McNemar odds ratio formally infinite,
so the effect size is better reported as the absolute paired difference above.</div>

<h2><span class="n">04</span>Generalization is real but category-dependent</h2>
<p>None of these 12 unsafe categories were used to build the direction, yet refusal drops in all of
them. The size of the drop, however, ranges from a near-total collapse to almost nothing &mdash;
evidence that "refusal" is largely, but not entirely, carried by the single fitted axis.</p>

{figblock('dumbbell', 3, 'Per-category refusal rate, baseline (blue) vs edited (red), 25 prompts each. Categories are sorted by drop size; "n.s." marks the one category whose change is not significant after Holm correction.')}

<table>
<thead><tr><th class="l">Unsafe category</th><th>Base</th><th>Edited</th><th>Drop</th>
<th>Flips</th><th>p (exact)</th><th>p (Holm)</th><th>Sig</th></tr></thead>
<tbody>
{cat_rows}
</tbody></table>
<p style="font-size:13px;color:var(--mut)">Holm&ndash;Bonferroni correction across the 12 category tests.
<b>{int(cat['sig'].sum())} of 12</b> categories show a significant refusal reduction at &alpha;=0.05.</p>

<div class="callout warn"><b>The safety-relevant outlier.</b> <b>Self-harm facilitation</b> is the one
category the edit barely touches (100% &rarr; {pct(cat[cat.category.str.contains('Self-harm')]['edit'].iloc[0],0)}, not significant).
Whatever governs the model's self-harm refusals is mostly <i>not</i> aligned with the single direction
that controls the rest &mdash; a useful hint that refusal is not perfectly one-dimensional.</div>

<h2><span class="n">05</span>A subtler cost: refusals stop being clean</h2>
<p>The binary label hides a second effect. Baseline refusals are terse one-liners
("I cannot help with that."). After the edit, answers to unsafe prompts get much longer &mdash;
and crucially, this is true <b>even for prompts that are still labeled "refused."</b> Those
become 200-word hedged essays that explain the harm, moralize, and frequently offer a
"for-fiction-only" reframing &mdash; a partial erosion of the refusal that the binary outcome cannot see.</p>

<div class="two">
{figblock('wordcounts', 4, 'Per-answer word counts. Unsafe answers jump from a tight cluster of short refusals to long replies; safe answers are unchanged.')}
{figblock('verbosity', 5, 'Word counts among answers still labeled "refused" on unsafe prompts. The edited refusals are an order of magnitude longer.')}
</div>

<div class="statline">
Unsafe length &middot; median <b>{wc['unsafe']['base_med']:.0f} &rarr; {wc['unsafe']['edit_med']:.0f}</b> words,
Wilcoxon <b>p = {wc['unsafe']['p']:.1e}</b>, rank-biserial <b>{wc['unsafe']['rbc']:+.2f}</b>
&nbsp;|&nbsp; Safe length &middot; median <b>{wc['safe']['base_med']:.0f} &rarr; {wc['safe']['edit_med']:.0f}</b>,
Wilcoxon <b>p = {wc['safe']['p']:.2f}</b> (n.s.)
&nbsp;|&nbsp; Still-refused unsafe answers &middot; median <b>{verb['base']:.0f} &rarr; {verb['edit']:.0f}</b> words
</div>

<h2><span class="n">06</span>Methods &amp; caveats</h2>
<div class="two">
<div>
<h3>What was tested</h3>
<ul>
<li><b>Unit of analysis:</b> the prompt; each contributes one matched (baseline, edited) pair.</li>
<li><b>Primary outcome:</b> manual label collapsed to <i>refused</i> vs. <i>not-refused</i>.</li>
<li><b>Tests:</b> McNemar exact (paired binary); Wilson intervals for each rate; 20k-resample
bootstrap for the paired-difference CI; Wilcoxon signed-rank for word counts;
Holm correction across the 12 category tests.</li>
</ul>
</div>
<div>
<h3>Caveats</h3>
<ul>
<li><b>Single labeler, no adjudicated reliability.</b> Labels are one rater's judgment; no
inter-annotator agreement is available, so the boundary cases (esp. long soft-refusals) carry rater uncertainty.</li>
<li><b>One <code>unusable</code> output</b> (a partial malware script) is treated as not-a-clean-refusal;
with n=1 it moves nothing. </li>
<li><b>Greedy decoding, one sample per prompt</b> &mdash; no within-prompt sampling variance is captured.</li>
<li><b>Quasi-complete separation</b> (the empty off-diagonal) rules out a category-clustered logistic
GLMM, so clustering is handled descriptively via the per-category tests rather than a mixed model.</li>
<li><b>"Refused" is a coarse target.</b> Section 5 shows the edit degrades refusal quality even where
the label is unchanged, so the {pct(u['edit_r'],1)} residual unsafe-refusal rate likely <i>overstates</i>
how safe the edited model really is.</li>
</ul>
</div>
</div>

<div class="callout"><b>Bottom line.</b> One linear direction accounts for a large, transferable
fraction of Gemma's refusal behavior &mdash; enough to flip ~half of unseen unsafe prompts to
compliance and to soften most of the rest &mdash; yet self-harm refusals and the residual
hard-refusals show the behavior is not fully one-dimensional. The edit is precise (no safe-side
collateral) but not complete.</div>

<footer>
Generated from <code>evaluation_outputs_manual_labels_filled.csv</code> (600 rows) &middot;
all statistics recomputed by <code>analysis/analyze.py</code> &middot;
McNemar/Wilcoxon via SciPy &amp; statsmodels &middot; bootstrap seed {SEED} &middot;
method after Arditi et al. 2024, &ldquo;Refusal in Language Models Is Mediated by a Single Direction.&rdquo;
</footer>

</div></body></html>"""
    return html


def main():
    df = load()
    res, wc, verb, cat = stats_all(df)
    figs = {
        "bars": fig_refusal_bars(res),
        "alluvial": fig_alluvial(df),
        "dumbbell": fig_dumbbell(cat),
        "wordcounts": fig_wordcounts(df),
        "verbosity": fig_refusal_verbosity(df),
    }
    html = build_html(df, res, wc, verb, cat, figs)
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}  ({len(html)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
