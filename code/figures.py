#!/usr/bin/env python3
"""Figure: fingerspelling removed versus meaning kept, per system and vocabulary.

    python code/figures.py      # reads results/tables/main_results.csv

One panel per text type. x = content words fingerspelled (%), y = COMET (FLORES)
or mutual entailment with the input (AMI). Colour = method family (validated
categorical slots 1-3 of the reference palette; baseline in neutral grey),
marker = vocabulary; each point carries its system code as a direct label.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from adjustText import adjust_text  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"

SURFACE = "#fcfcfb"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FAMILY = {  # colour by method family
    "baseline": ("Text or translation as it is", "#52514e"),
    "word": ("Word replacement", "#2a78d6"),
    "llm": ("Language model, unrestricted", "#eb6834"),
    "restricted": ("Language model, restricted to the avatar's words", "#1baf7a"),
}
# result file stem (before "__") -> code used in the paper
PAPER_CODE = {"E0_orig": "E0", "E1_wordnet": "E1", "E2_embed": "E2", "E3_simplify": "E3", "E4_hints": "E4",
              "E5_hints_constrained": "E5", "E6_simplify_constrained": "E6",
              "E6nofilter_simplify_constrained": "E6c", "R5_E6_simplify_constrained": "E6r",
              "T0_nllb": "T0", "T1_eurollm": "T1", "T2_nllb_embed": "T2", "T6_nllb_simplify_constrained": "T3",
              "T4_eurollm_constrained": "T4"}
SYSTEM_FAMILY = {"E0": "baseline", "T0": "baseline", "T1": "baseline", "E1": "word", "E2": "word",
                 "T2": "word", "E3": "llm", "E4": "llm", "E5": "restricted", "E6": "restricted",
                 "E6c": "restricted", "E6r": "restricted", "T3": "restricted", "T4": "restricted"}
VOCAB = {"is": ("IS dictionary", "o"), "freq1000": ("Top 1,000 words", "^"),
         "freq5000": ("Top 5,000 words", "s")}
PANELS = [("flores_eng", "Written sentences, English", "comet", "COMET"),
          ("flores_ell_Grek", "Written sentences, Greek → English", "comet", "COMET"),
          ("flores_fra_Latn", "Written sentences, French → English", "comet", "COMET"),
          ("ami", "Meetings, English", "entail", "Mutual entailment with the original (%)")]


def draw(panels, stem: Path, figsize=(9.2, 7.4), exts=("png", "pdf"), surface=SURFACE,
         legend_rect=0.07, ncols=2, family_labels=None, legend_ncol=4, legend_fontsize=8.0):
    """Scatter panels of fingerspelling against meaning; one panel per (corpus, title, metric, label).

    family_labels optionally renames method families in the legend (for example "Input as is" when the
    figure shows English input only). With legend_ncol=2 the legend lists the method families in its first
    column and the dictionaries in its second, which suits a figure drawn at the printed width."""
    res = pd.read_csv(TABLES / "main_results.csv")
    res = res[res["vocab"].isin(VOCAB) & ~res["system"].str.contains("nofilter")].copy()
    res["code"] = res["system"].map(lambda s: PAPER_CODE.get(s.split("__")[0], s[:2]))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=surface)
    for ax, (corpus, title, metric, ylabel) in zip(np.ravel(axes), panels):
        ax.set_facecolor(surface)
        d = res[(res["corpus"] == corpus) & res[metric].notna()].copy()
        d["x"] = 100 * d["fs_rate"]
        d["y"] = d[metric] * (100 if metric == "entail" else 1)
        d["fam"] = d["code"].map(lambda c: SYSTEM_FAMILY.get(c, "baseline"))
        for _, r in d.iterrows():
            ax.scatter(r["x"], r["y"], s=46, marker=VOCAB[r["vocab"]][1], color=FAMILY[r["fam"]][1],
                       edgecolors=surface, linewidths=1.5, zorder=3)
        ax.set_xlim(-3, max(d["x"].max() * 1.08, 10))
        lo, hi = d["y"].min(), d["y"].max()
        ax.set_ylim(lo - 0.06 * (hi - lo), hi + 0.08 * (hi - lo))
        # one label per group of points that coincide (same family and vocabulary)
        fig.canvas.draw()
        xy = ax.transData.transform(d[["x", "y"]].to_numpy())
        d["px"], d["py"] = xy[:, 0], xy[:, 1]
        used, texts = set(), []
        for i, r in d.sort_values("code").iterrows():
            if i in used:
                continue
            near = d[(d["fam"] == r["fam"]) & (d["vocab"] == r["vocab"]) &
                     (np.hypot(d["px"] - r["px"], d["py"] - r["py"]) < 10)]
            used |= set(near.index)
            label = ", ".join(sorted(near["code"].unique()))
            texts.append(ax.text(near["x"].mean(), near["y"].mean(), label, fontsize=7, color=INK2, zorder=4))
        adjust_text(texts, x=d["x"].to_numpy(), y=d["y"].to_numpy(), ax=ax, expand=(1.3, 1.6),
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.7))
        ax.set_title(title, loc="left", fontsize=9.5, color=INK, pad=6)
        ax.set_xlabel("Content words fingerspelled (%)")
        ax.set_ylabel(ylabel)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    present = {SYSTEM_FAMILY.get(c, "baseline") for c in res[res["corpus"].isin([p[0] for p in panels])]["code"]}
    names = family_labels or {}
    fam = [Line2D([], [], marker="o", linestyle="", markersize=7, color=c, label=names.get(f, l))
           for f, (l, c) in FAMILY.items() if f in present]
    voc = [Line2D([], [], marker=m, linestyle="", markersize=7, color=INK2, markerfacecolor="none", label=l)
           for l, m in VOCAB.values()]
    blank = Line2D([], [], linestyle="", label=" ")
    if legend_ncol == 2:  # families in the first column, dictionaries in the second
        n = max(len(fam), len(voc))
        handles = fam + [blank] * (n - len(fam)) + voc + [blank] * (n - len(voc))
    else:  # keep the families in the first two legend columns and the dictionaries in the others
        handles = fam + ([blank] if len(fam) % 2 else []) + voc
    fig.legend(handles=handles, loc="lower center", ncol=legend_ncol, frameon=False, fontsize=legend_fontsize,
               bbox_to_anchor=(0.5, -0.005), columnspacing=2.5)
    fig.tight_layout(rect=(0, legend_rect, 1, 1))
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in exts:
        kw = {"pil_kwargs": {"quality": 95}} if ext == "jpg" else {}
        fig.savefig(stem.with_suffix("." + ext), dpi=300, facecolor=surface, **kw)
    plt.close(fig)
    return stem


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    # drawn at the printed width, so that its text keeps its size, the jpg is for the journal upload
    draw(PANELS, FIGS / "tradeoff", figsize=(6.5, 6.6), exts=("png", "pdf", "jpg"), surface="#ffffff",
         legend_rect=0.11, legend_ncol=2, legend_fontsize=7.5)
    print("wrote", FIGS / "tradeoff.png")


if __name__ == "__main__":
    main()
