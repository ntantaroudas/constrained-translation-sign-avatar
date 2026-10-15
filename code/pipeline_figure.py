#!/usr/bin/env python3
"""Figure: how a sentence reaches the avatar, and where each method changes it.

    python code/pipeline_figure.py      # -> results/figures/pipeline.{png,pdf,jpg}

A schematic, not a plot of results. The colours are the method families of the trade-off figure
(code/figures.py): grey for the text as it is, blue for word replacement, orange for unrestricted
rewriting by the language model and green for rewriting restricted to the avatar's words.
`draw(rows=...)` draws the English path, the translation path and the direct restricted translation,
or a subset of them for the two companion papers.

The figure is drawn at the width at which it is printed (6.5 inches by default), with the axes in
points, so that the text keeps its size (8 and 7.5 pt) in the article and the boxes are sized from
the number of lines they hold.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "results" / "figures"
INK, INK2, LINE = "#0b0b0b", "#52514e", "#8a8984"
FAMILY = {"base": ("#52514e", "#ecebe8"), "word": ("#2a78d6", "#e3eefb"),
          "llm": ("#eb6834", "#fdebe3"), "restricted": ("#1baf7a", "#ddf5ec")}
FS, FS_CHIP = 8.0, 7.5          # font sizes in points
LEAD = 1.3                      # line height as a multiple of the font size
PAD = 4.0                       # inner padding of a box, points
GAP_ROWS, GAP_CHIPS = 10.0, 3.0
TITLE_H = 15.0                  # space for the title of a container of options


def n_lines(text: str) -> int:
    return text.count("\n") + 1


def text_height(text: str, fs: float) -> float:
    return n_lines(text) * fs * LEAD


def box(ax, x, y, w, h, text, edge=LINE, face="#ffffff", fs=FS, bar=None, align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=4", linewidth=0.9,
                                edgecolor=edge, facecolor=face, zorder=2))
    if bar:  # coloured bar on the left edge marks the method family
        ax.add_patch(FancyBboxPatch((x, y), 4.5, h, boxstyle="round,pad=0,rounding_size=2", linewidth=0,
                                    facecolor=bar, zorder=3))
    tx = x + w / 2 if align == "center" else x + 10
    ax.text(tx, y + h / 2, text, ha=align, va="center", fontsize=fs, color=INK, zorder=4, linespacing=LEAD)


def arrow(ax, p, q, head=True):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>" if head else "-", mutation_scale=8, linewidth=0.9,
                                 color=INK2, shrinkA=0, shrinkB=0, zorder=1))


def chip_height(text: str) -> float:
    return text_height(text, FS_CHIP) + 2 * PAD


def options_height(items) -> float:
    return TITLE_H + sum(chip_height(t) for _, t in items) + GAP_CHIPS * (len(items) - 1) + PAD + 1


def options(ax, x, y_top, w, items, title):
    """A dashed container titled `title`, holding one coloured chip per option, drawn downwards from y_top.
    Returns the height of the container."""
    h = options_height(items)
    y = y_top - h
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=5", linewidth=0.9,
                                edgecolor=LINE, facecolor="#fafaf8", linestyle=(0, (3, 2)), zorder=1))
    ax.text(x + w / 2, y_top - TITLE_H / 2 - 1, title, ha="center", va="center", fontsize=FS_CHIP, color=INK2,
            zorder=4)
    cy = y_top - TITLE_H
    for fam, text in items:
        ch = chip_height(text)
        edge, face = FAMILY[fam]
        box(ax, x + 5, cy - ch, w - 10, ch, text, edge=face, face=face, fs=FS_CHIP, bar=edge, align="left")
        cy -= ch + GAP_CHIPS
    return h


ENG_ITEMS = [("base", "none, the text is signed as it is (E0)"),
             ("word", "word replacement, by WordNet (E1)\nor by word vectors (E2)"),
             ("llm", "rewriting by the language model,\nsimplification (E3) or with\nsuggested words (E4)"),
             ("restricted", "the same, restricted to the avatar's\nwords (E5, E6, E6c, E6r)")]
TR_ITEMS = [("base", "none, the translation is\nsigned as it is (T0, T1)"),
            ("word", "word replacement with\nword vectors (T2)"),
            ("restricted", "simplification,\nrestricted to the\navatar's words (T3)")]
SIGNING = ("signing by\ndictionary lookup\n\na sign for every word\nin the dictionary,\nfingerspelling\nfor the rest\n\n"
           "measured, the\nfingerspelling, the\nsigning time and the\nmeaning that is kept")


def draw(stem: Path, rows=("english", "translation", "direct"), exts=("png", "pdf", "jpg"), surface="#ffffff",
         width_in: float = 6.5):
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    W = width_in * 72
    # column positions, in points
    in_x, in_w = 2.0, 78.0
    tr_x, tr_w = 96.0, 82.0
    right = W - 136.0                     # right edge of the rewording column
    sign_x, sign_w = right + 16.0, W - right - 18.0
    direct_h = text_height("x\nx", FS_CHIP) + 2 * PAD + 2
    heights = []
    if "english" in rows:
        heights.append(options_height(ENG_ITEMS))
    if "translation" in rows:
        heights.append(options_height(TR_ITEMS))
    if "direct" in rows:
        heights.append(direct_h)
    sign_h = text_height(SIGNING, FS_CHIP) + 2 * PAD + 4
    H = max(sum(heights) + GAP_ROWS * (len(heights) - 1), sign_h) + 4.0
    fig = plt.figure(figsize=(W / 72, H / 72), facecolor=surface)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    y = H - 2.0 - (H - 4.0 - (sum(heights) + GAP_ROWS * (len(heights) - 1))) / 2
    targets, src_bottom = [], None
    if "english" in rows:
        h = options(ax, tr_x, y, right - tr_x, ENG_ITEMS, "rewording, optional")
        yc = y - h / 2
        bh = text_height("x\nx\nx", FS) + 2 * PAD + 4
        box(ax, in_x, yc - bh / 2, in_w, bh, "English text,\nmeetings and\nwritten text")
        arrow(ax, (in_x + in_w, yc), (tr_x, yc))
        targets.append(yc)
        y -= h + GAP_ROWS
    if "translation" in rows:
        ox = tr_x + tr_w + 16.0
        h = options(ax, ox, y, right - ox, TR_ITEMS, "rewording, optional")
        yc = y - h / 2
        bh = text_height("x\nx\nx\nx", FS) + 2 * PAD + 4
        box(ax, in_x, yc - bh / 2, in_w, bh, "Greek, French,\nSpanish or\nGerman\nwritten text")
        box(ax, tr_x, yc - bh / 2, tr_w, bh, "translation\ninto English\nby NLLB (T0)\nor EuroLLM (T1)", fs=FS_CHIP)
        arrow(ax, (in_x + in_w, yc), (tr_x, yc))
        arrow(ax, (tr_x + tr_w, yc), (ox, yc))
        targets.append(yc)
        src_bottom = yc - bh / 2
        y -= h + GAP_ROWS
    if "direct" in rows:
        yc = y - direct_h / 2
        if src_bottom is None:
            bh = text_height("x\nx", FS) + 2 * PAD + 2
            box(ax, in_x, yc - bh / 2, in_w, bh, "Greek or French\nwritten text")
        edge, face = FAMILY["restricted"]
        box(ax, tr_x, yc - direct_h / 2, right - tr_x, direct_h,
            "direct translation by EuroLLM, restricted\nto the avatar's words (T4)", edge=face, face=face, fs=FS_CHIP,
            bar=edge, align="left")
        if src_bottom is not None:
            xm = in_x + in_w / 2
            arrow(ax, (xm, src_bottom), (xm, yc), head=False)
            arrow(ax, (xm, yc), (tr_x, yc))
        else:
            arrow(ax, (in_x + in_w, yc), (tr_x, yc))
        targets.append(yc)
    # the signing box spans every row, so that each row enters it with a straight arrow
    lo = min(min(targets) - 14.0, H / 2 - sign_h / 2)
    hi = max(max(targets) + 14.0, H / 2 + sign_h / 2)
    lo, hi = max(lo, 2.0), min(hi, H - 2.0)
    box(ax, sign_x, lo, sign_w, hi - lo, SIGNING, edge=INK2, face="#ecebe8", fs=FS_CHIP)
    for ty in targets:
        arrow(ax, (right, ty), (sign_x, ty))
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in exts:
        kw = {"pil_kwargs": {"quality": 95}} if ext == "jpg" else {}
        fig.savefig(stem.with_suffix("." + ext), dpi=300, facecolor=surface, **kw)
    plt.close(fig)
    return stem


def main():
    draw(FIGS / "pipeline")
    print("wrote", FIGS / "pipeline.png")


if __name__ == "__main__":
    main()
