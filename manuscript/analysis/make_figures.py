"""Regenerate every figure in the open-world ITS manuscript from data/*.csv.

Run:  python3 analysis/make_figures.py
Every number drawn here is read from the CSVs, so the figures cannot drift
away from the tables (which analysis/make_tables.py builds from the same files).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import binom

from style import (use_style, save, panel_label, DATADIR, FIGDIR,
                   BLUE, RED, GREEN, YELLOW, PURPLE, GREY, DARK,
                   VIEW_COLOUR, VIEW_LABEL, VIEW_ORDER,
                   MODEL_ORDER, MODEL_SUBTITLE)

use_style()

ab = pd.read_csv(DATADIR / "development_ablation.csv")
final = pd.read_csv(DATADIR / "frozen_test_metrics.csv")
conf = pd.read_csv(DATADIR / "conformal_operating_points.csv")
hist = pd.read_csv(DATADIR / "historical_benchmark.csv")
mat = pd.read_csv(DATADIR / "m4_dev_retrieval_matrix.csv")
audit = pd.read_csv(DATADIR / "historical_leakage_audit.csv")

# Bootstrap intervals for the frozen TEST AUROCs (2,000 stratified resamples,
# seed 17) as reported by the locked evaluator.
TEST_CI = {"core": (0.739, 0.770), "its1": (0.699, 0.734), "its2": (0.748, 0.779)}


# ---------------------------------------------------------------- figure 1 ---
def box(ax, x, y, w, h, text, fc="white", ec=DARK, lw=0.8, fs=8, weight=None,
        style="round,pad=0.3,rounding_size=1.2", tc=DARK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
                                facecolor=fc, edgecolor=ec, linewidth=lw,
                                mutation_scale=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight, color=tc, linespacing=1.35)


def arrow(ax, p0, p1, ec=DARK, lw=0.8, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8,
                                 color=ec, linewidth=lw, linestyle=ls,
                                 shrinkA=1, shrinkB=1))


def figure1():
    """Two zones separated by the firewall: everything above it was iterated on,
    everything below it was opened once. Fill colour encodes the kind of object:
    grey = a fixed partition of the data, white = a process, yellow = a model,
    purple = a reported result.

    Layout notes. The axes fills the figure and the figure is saved without a
    tight bounding box, so the aspect ratio here is exactly the aspect ratio on
    the page and spacing is predictable. At 6.3 x 7.7 in rendered to \linewidth,
    one x unit is about 4.5 pt and one y unit about 5.5 pt, so vertical gaps buy
    more visual air per unit than horizontal ones. Text is plain: this is
    matplotlib, not LaTeX, so no backslash escapes.
    """
    DATA_FC, DATA_EC = "#ECEFF3", "#8A94A6"
    MODEL_FC, MODEL_EC = YELLOW + "3A", "#8F8230"
    OUT_FC, OUT_EC = PURPLE + "16", PURPLE
    FS = 7.3

    fig = plt.figure(figsize=(6.3, 7.7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def head(y, letter, text):
        ax.text(1, y, letter, fontsize=10, fontweight="bold", va="top")
        ax.text(4.6, y, text, fontsize=9.2, fontweight="bold", va="top")

    # ================= a. collection and barcode views =====================
    head(97.5, "a", "Collection and barcode views")
    box(ax, 1, 83.5, 21, 9.5, "UNITE 2025\ndynamic release\n102,137 records",
        fc=DATA_FC, ec=DATA_EC, fs=FS)
    box(ax, 26, 83.5, 21, 9.5, "ITSx extraction\n+ exact-hash\nconflict masking", fs=FS)
    box(ax, 51, 83.5, 21, 9.5, "named family\nand genus\n67,406 records",
        fc=DATA_FC, ec=DATA_EC, fs=FS)
    arrow(ax, (22, 88.2), (26, 88.2))
    arrow(ax, (47, 88.2), (51, 88.2))
    for i, (name, col) in enumerate(zip(["ITS-core", "ITS1", "ITS2"],
                                        [BLUE, RED, GREEN])):
        box(ax, 78, 90.4 - 3.7 * i, 21, 2.6, name, fc=col + "26", ec=col, fs=FS)
        arrow(ax, (72, 88.2), (78, 91.7 - 3.7 * i), ec=col)
    arrow(ax, (61.5, 83.5), (61.5, 80.4))
    ax.text(64, 81.2, "split into six genus-separated partitions",
            fontsize=6.9, color="#666666", va="center")

    # ================= b. development ======================================
    head(77.5, "b", "Development: only TRAIN_REF receives gradients")
    box(ax, 1, 63.5, 26, 8.5, "TRAIN_REF\n46,453 records", fc=DATA_FC, ec=DATA_EC, fs=FS)
    box(ax, 1, 50.5, 26, 8.5, "DEV_KNOWN 1,528\nDEV_NOVEL 8,698",
        fc=DATA_FC, ec=DATA_EC, fs=FS)
    arrow(ax, (27, 67.7), (43, 67.7))
    ax.text(35, 69.1, "gradients", fontsize=6.9, color=DARK, ha="center",
            style="italic")
    arrow(ax, (27, 54.7), (43, 54.7), ls=(0, (3, 2)))
    ax.text(35, 56.1, "selection only,\n4 integration targets", fontsize=6.9,
            color=DARK, ha="center", va="bottom", style="italic", linespacing=1.35)

    # --- the ladder as an objective-term matrix (branching is explicit) -----
    cols = [("proxy", 55.0), ("view", 66.5), ("taxonomy", 78.0), ("episodes", 89.5)]
    rows = [("M0", 67.7, [1, 0, 0, 0]),
            ("M1", 64.0, [1, 1, 0, 0]),
            ("M2", 60.3, [1, 1, 1, 0]),
            ("M3", 56.6, [1, 1, 0, 1]),
            ("M4", 52.5, [1, 1, 1, 1])]
    ax.add_patch(FancyBboxPatch((44.0, 50.7), 55.0, 3.6,
                                boxstyle="round,pad=0.2,rounding_size=1.0",
                                facecolor=MODEL_FC, edgecolor=MODEL_EC, lw=1.1))
    for label, x in cols:
        ax.text(x, 72.0, label, ha="center", va="center", fontsize=7.0, color=DARK)
    ax.plot([45.0, 96.5], [70.3, 70.3], color="#AAAAAA", lw=0.6)
    for label, y, dots in rows:
        ax.text(46.0, y, label, ha="left", va="center", fontsize=FS,
                fontweight="bold" if label == "M4" else "normal")
        for (_, x), on in zip(cols, dots):
            if on:
                ax.plot([x], [y], "o", ms=5.4, color=BLUE, mec="white", mew=0.7)
            else:
                ax.plot([x], [y], "o", ms=4.6, color="white", mec="#C9C9C9", mew=0.8)
    ax.text(97.5, 52.5, "frozen", ha="right", va="center", fontsize=7.0,
            fontweight="bold", color="#6F6524")
    ax.text(71.5, 47.6, "one objective term added at a time; M2 and M3 both branch from M1",
            ha="center", fontsize=6.9, color="#666666")

    # ================= the firewall ========================================
    ax.add_patch(FancyBboxPatch((1, 40.0), 98, 4.4,
                                boxstyle="round,pad=0.2,rounding_size=1.0",
                                facecolor=RED + "1C", edgecolor=RED,
                                lw=1.0, linestyle=(0, (5, 3))))
    ax.text(50, 42.2, "EVALUATION FIREWALL   \u00b7   sealed by SHA-256   \u00b7   "
                      "CAL and TEST opened once, after freezing",
            ha="center", va="center", fontsize=7.5, color=RED, fontweight="bold")

    # ================= c. primary evaluation ===============================
    head(37.0, "c", "Evaluation 1 (primary): frozen M4 against a matched alignment baseline")
    ax.text(1, 33.4, "the same queries are also scored by exhaustive VSEARCH",
            fontsize=6.9, color="#666666", va="center", style="italic")
    box(ax, 1, 22.5, 18, 8, "frozen M4,\nexact-length\ninference", fc=MODEL_FC, ec=MODEL_EC,
        lw=1.2, fs=FS, weight="bold")
    box(ax, 23, 22.5, 21, 8, "CAL_KNOWN 1,505\ncalibrates the\nanomaly score",
        fc=DATA_FC, ec=DATA_EC, fs=FS)
    box(ax, 48, 27.0, 23, 5.2, "TEST_KNOWN 1,484", fc=DATA_FC, ec=DATA_EC, fs=FS)
    box(ax, 48, 20.0, 23, 5.2, "TEST_NOVEL 7,738", fc=DATA_FC, ec=DATA_EC, fs=FS)
    box(ax, 75, 27.0, 24, 5.2, "false-novelty rate", fc=OUT_FC, ec=OUT_EC, fs=FS)
    box(ax, 75, 17.5, 24, 7.7, "detection rate,\nfamily / order / class\nplacement",
        fc=OUT_FC, ec=OUT_EC, fs=FS)
    arrow(ax, (19, 26.5), (23, 26.5))
    arrow(ax, (44, 27.5), (48, 29.6))
    arrow(ax, (44, 25.5), (48, 22.6))
    arrow(ax, (71, 29.6), (75, 29.6), ec=PURPLE)
    arrow(ax, (71, 22.6), (75, 21.4), ec=PURPLE)

    # ================= d. secondary evaluation =============================
    head(15.5, "d", "Evaluation 2 (secondary): historical ITS2 benchmark")
    ax.text(1, 11.6, "the frozen weights were never scored here; only the fixed "
                    "recipe was reused",
            fontsize=6.9, color=RED, va="center", style="italic")
    box(ax, 1, 0.8, 22, 9.0, "historical LSO/LGO\n3,264 / 4,444\nITS2 queries",
        fc=DATA_FC, ec=DATA_EC, fs=FS)
    box(ax, 26, 0.8, 24, 9.0, "leakage audit:\n71% of LGO queries\n"
                              "and 85% of LGO genera\nwere in TRAIN_REF",
        fc=RED + "16", ec=RED, fs=FS)
    box(ax, 53, 0.8, 22, 9.0, "retrain the recipe\non masked TRAIN_REF,\n41,344 records",
        fc=MODEL_FC, ec=MODEL_EC, fs=FS)
    box(ax, 78, 0.8, 21, 9.0, "identity vs cosine,\nidentical queries\nand references",
        fc=OUT_FC, ec=OUT_EC, fs=FS)
    arrow(ax, (23, 5.3), (26, 5.3))
    arrow(ax, (50, 5.3), (53, 5.3))
    arrow(ax, (75, 5.3), (78, 5.3), ec=PURPLE)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"fig1_workflow.{ext}", bbox_inches=None,
                    pad_inches=0)
    plt.close(fig)
    print("  wrote figures/fig1_workflow.pdf and .png")


# ---------------------------------------------------------------- figure 2 ---
def figure2():
    fig = plt.figure(figsize=(6.3, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.12], hspace=0.42, wspace=0.22)
    ax0, ax1 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])

    width = 0.26
    idx = np.arange(len(MODEL_ORDER))
    for k, view in enumerate(VIEW_ORDER):
        sub = ab[ab["view"] == view].set_index("model")
        auc = [sub.loc[m, "auroc"] for m in MODEL_ORDER]
        fam = [100 * sub.loc[m, "novel_family"] for m in MODEL_ORDER]
        off = (k - 1) * width
        ax0.bar(idx + off, np.array(auc) - 0.5, width, bottom=0.5,
                color=VIEW_COLOUR[view], label=VIEW_LABEL[view],
                edgecolor="white", linewidth=0.4)
        ax1.bar(idx + off, fam, width, color=VIEW_COLOUR[view],
                label=VIEW_LABEL[view], edgecolor="white", linewidth=0.4)

    ax0.axhline(0.5, color=DARK, ls=(0, (4, 3)), lw=0.9)
    ax0.text(-0.45, 0.503, "chance", fontsize=7.2, ha="left", va="bottom", color=DARK)
    ax0.set_ylim(0.5, 0.81)
    ax0.set_ylabel("novel-genus AUROC")
    ax1.set_ylim(0, 65)
    ax1.set_ylabel("novel-genus family placement (%)")
    for ax in (ax0, ax1):
        ax.set_xticks(idx)
        ax.set_xticklabels([f"{m}\n{MODEL_SUBTITLE[m]}" for m in MODEL_ORDER],
                           fontsize=7.6)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
    ax0.legend(ncol=3, loc="upper left", bbox_to_anchor=(-0.02, 1.22),
               columnspacing=1.1, handlelength=1.1)
    panel_label(ax0, "a", y=1.24)
    panel_label(ax1, "b", y=1.24)

    # panel c: the trade-off the ladder is really navigating
    MARK = {"M0": "o", "M1": "s", "M2": "^", "M3": "v", "M4": "*"}
    for view in VIEW_ORDER:
        sub = ab[ab["view"] == view].set_index("model")
        for path in (["M0", "M1", "M2", "M4"], ["M1", "M3", "M4"]):
            ax2.plot([sub.loc[m, "auroc"] for m in path],
                     [100 * sub.loc[m, "novel_family"] for m in path],
                     color=VIEW_COLOUR[view], lw=0.8, alpha=0.5, zorder=1)
        for m in MODEL_ORDER:
            big = m == "M4"
            ax2.scatter(sub.loc[m, "auroc"], 100 * sub.loc[m, "novel_family"],
                        s=170 if big else 40, marker=MARK[m],
                        color=VIEW_COLOUR[view],
                        edgecolor="white", linewidth=0.7, zorder=3)
    ax2.set_xlabel("novel-genus AUROC (novelty discrimination)")
    ax2.set_ylabel("family placement (%)")
    ax2.set_xlim(0.66, 0.795)
    ax2.set_ylim(15, 66)
    ax2.grid(True, zorder=0)
    ax2.set_axisbelow(True)
    ax2.annotate("better on\nboth axes", xy=(0.7895, 31), xytext=(0.766, 19.5),
                 fontsize=7.6, color="#777777", ha="center", va="bottom",
                 arrowprops=dict(arrowstyle="-|>", color="#999999", lw=0.9,
                                 connectionstyle="arc3,rad=-0.15"))
    view_handles = [plt.Line2D([], [], color=VIEW_COLOUR[v], marker="o", ls="-",
                               markersize=5, lw=1.0, label=VIEW_LABEL[v])
                    for v in VIEW_ORDER]
    model_handles = [plt.Line2D([], [], color="#777777", marker=MARK[m], ls="",
                                markersize=9 if m == "M4" else 5.5, label=m)
                     for m in MODEL_ORDER]
    leg1 = ax2.legend(handles=view_handles, ncol=3, loc="upper left",
                      columnspacing=1.1, handlelength=1.4)
    ax2.add_artist(leg1)
    ax2.legend(handles=model_handles, ncol=5, loc="upper left",
               bbox_to_anchor=(0.0, 0.90), columnspacing=0.9, handlelength=0.8,
               handletextpad=0.3)
    panel_label(ax2, "c", y=1.03)

    save(fig, "fig2_ladder")


# ---------------------------------------------------------------- figure 3 ---
def figure3():
    panels = [("novelty_auroc", "novel-genus AUROC", "%.3f", None),
              ("dev_known_nn_genus_accuracy", "known-genus NN accuracy (%)", "%.1f", 100),
              ("dev_novel_nn_family_accuracy", "novel-genus family NN (%)", "%.1f", 100)]
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.55))
    for ax, (col, title, fmt, scale), letter in zip(axes, panels, "abc"):
        M = np.zeros((3, 3))
        for i, q in enumerate(VIEW_ORDER):
            for j, r in enumerate(VIEW_ORDER):
                v = mat[(mat.query_view == q) & (mat.reference_view == r)][col].iloc[0]
                M[i, j] = v * scale if scale else v
        im = ax.imshow(M, cmap="Blues", vmin=M.min() * 0.92, vmax=M.max())
        for i in range(3):
            for j in range(3):
                rel = (M[i, j] - M.min()) / (M.max() - M.min() + 1e-9)
                ax.text(j, i, fmt % M[i, j], ha="center", va="center",
                        fontsize=8, color="white" if rel > 0.62 else DARK,
                        fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels([VIEW_LABEL[v] for v in VIEW_ORDER], fontsize=8)
        ax.set_yticklabels([VIEW_LABEL[v] for v in VIEW_ORDER], fontsize=8,
                           rotation=90, va="center")
        ax.set_title(title, fontsize=8.5, pad=6)
        ax.set_xlabel("reference view", fontsize=8)
        if letter == "a":
            ax.set_ylabel("query view", fontsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        panel_label(ax, letter, x=-0.28, y=1.12)
    fig.subplots_adjust(wspace=0.34)
    save(fig, "fig3_crossview_matrix")


# ---------------------------------------------------------------- figure 4 ---
def figure4():
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.8),
                             gridspec_kw={"width_ratios": [1, 1.35], "wspace": 0.34})
    ax0, ax1 = axes
    f = final.set_index("view")

    for i, v in enumerate(VIEW_ORDER):
        lo, hi = TEST_CI[v]
        auc = f.loc[v, "auroc"]
        ax0.bar(i, auc - 0.5, 0.55, bottom=0.5, color=VIEW_COLOUR[v],
                edgecolor="white", linewidth=0.4)
        ax0.errorbar(i, auc, yerr=[[auc - lo], [hi - auc]], fmt="none",
                     ecolor=DARK, elinewidth=0.9, capsize=3)
        ax0.text(i, hi + 0.004, f"{auc:.3f}", ha="center", fontsize=7.8)
    ax0.axhline(0.5, color=DARK, ls=(0, (4, 3)), lw=0.9)
    ax0.set_xticks(range(3))
    ax0.set_xticklabels([VIEW_LABEL[v] for v in VIEW_ORDER])
    ax0.set_ylim(0.5, 0.81)
    ax0.set_ylabel("novel-genus AUROC (TEST)")
    ax0.yaxis.grid(True)
    ax0.set_axisbelow(True)
    panel_label(ax0, "a", x=-0.22, y=1.26)

    # where does the nearest neighbour actually land?
    labels = ["correct family", "correct order,\nwrong family",
              "correct class,\nwrong order", "wrong class"]
    cols = [BLUE, CYANISH := "#7BAFD4", "#C6DBEF", "#EEEEEE"]
    for i, v in enumerate(VIEW_ORDER):
        fam = 100 * f.loc[v, "novel_family"]
        order = 100 * f.loc[v, "novel_order"]
        cls = 100 * f.loc[v, "novel_class"]
        parts = [fam, order - fam, cls - order, 100 - cls]
        bottom = 0
        for p, c in zip(parts, cols):
            ax1.barh(i, p, left=bottom, height=0.55, color=c,
                     edgecolor="white", linewidth=0.6)
            if p > 4:
                ax1.text(bottom + p / 2, i, f"{p:.1f}", ha="center", va="center",
                         fontsize=7.5, color="white" if c == BLUE else DARK)
            bottom += p
    ax1.set_yticks(range(3))
    ax1.set_yticklabels([VIEW_LABEL[v] for v in VIEW_ORDER])
    ax1.invert_yaxis()
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("TEST_NOVEL queries (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cols]
    ax1.legend(handles, labels, ncol=4, fontsize=6.8, loc="upper left",
               bbox_to_anchor=(-0.02, 1.26), columnspacing=0.8,
               handlelength=0.9, handleheight=0.9)
    ax1.xaxis.grid(True)
    ax1.set_axisbelow(True)
    panel_label(ax1, "b", x=-0.14, y=1.26)
    save(fig, "fig4_frozen_test")


# ---------------------------------------------------------------- figure 5 ---
def figure5():
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7), gridspec_kw={"wspace": 0.28})
    ax0, ax1 = axes
    alphas = np.array(sorted(conf["alpha"].unique()))

    ax0.plot([0, 0.22], [0, 0.22], color=DARK, ls=(0, (4, 3)), lw=0.9,
             label="nominal $\\alpha$")
    for v in VIEW_ORDER:
        s = conf[conf["view"] == v].sort_values("alpha")
        ax0.plot(s["alpha"], s["false_novelty"], "-o", ms=4, lw=1.2,
                 color=VIEW_COLOUR[v], label=VIEW_LABEL[v])
    ax0.set_xlabel("nominal significance level $\\alpha$")
    ax0.set_ylabel("observed false-novelty rate")
    ax0.set_xlim(0, 0.215)
    ax0.set_ylim(0, 0.215)
    ax0.set_aspect("equal")
    ax0.legend(fontsize=7.2, loc="upper left", handlelength=1.4)
    ax0.grid(True)
    ax0.set_axisbelow(True)
    panel_label(ax0, "a", x=-0.2, y=1.04)

    for v in VIEW_ORDER:
        s = conf[conf["view"] == v].sort_values("alpha")
        ax1.plot(s["alpha"], s["novel_detection"], "-o", ms=4, lw=1.2,
                 color=VIEW_COLOUR[v], label=f"{VIEW_LABEL[v]}, per query")
        ax1.plot(s["alpha"], s["macro_genus"], "--s", ms=3.4, lw=1.0,
                 color=VIEW_COLOUR[v], alpha=0.75,
                 label=f"{VIEW_LABEL[v]}, per genus")
    ax1.set_xlabel("nominal significance level $\\alpha$")
    ax1.set_ylabel("novel-genus detection rate")
    ax1.set_xlim(0, 0.215)
    ax1.set_ylim(0, 0.58)
    ax1.legend(fontsize=6.8, ncol=1, loc="upper left", handlelength=1.6)
    for ax in (ax0, ax1):
        ax.set_xticks([0.0, 0.05, 0.10, 0.15, 0.20])
    ax1.grid(True)
    ax1.set_axisbelow(True)
    panel_label(ax1, "b", x=-0.18, y=1.04)
    save(fig, "fig5_conformal")


# ---------------------------------------------------------------- figure 6 ---
def figure6():
    h = hist.set_index("metric")
    fig = plt.figure(figsize=(6.3, 2.65))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.72, 1.25, 0.95], wspace=0.42)
    ax0, ax1, ax2 = [fig.add_subplot(gs[i]) for i in range(3)]
    ID, NN = PURPLE, GREEN

    for i, (lab, col, key) in enumerate([("percent\nidentity", ID, "identity"),
                                         ("M4\ncosine", NN, "m4_cosine")]):
        v = h.loc["AUROC", key]
        ax0.bar(i, v - 0.5, 0.55, bottom=0.5, color=col, edgecolor="white")
        ax0.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=8)
    ax0.axhline(0.5, color=DARK, ls=(0, (4, 3)), lw=0.9)
    ax0.set_xticks([0, 1])
    ax0.set_xticklabels(["percent\nidentity", "M4\ncosine"], fontsize=8)
    ax0.set_ylim(0.5, 0.81)
    ax0.set_ylabel("novel-genus AUROC")
    ax0.yaxis.grid(True)
    ax0.set_axisbelow(True)
    panel_label(ax0, "a", x=-0.36, y=1.04)

    al = [0.01, 0.05, 0.10, 0.20]
    det_id = [h.loc[f"Novel detection @ alpha={a:.2f}", "identity"] for a in al]
    det_nn = [h.loc[f"Novel detection @ alpha={a:.2f}", "m4_cosine"] for a in al]
    ax1.plot(al, det_id, "-o", color=ID, ms=4.5, lw=1.3, label="percent identity")
    ax1.plot(al, det_nn, "-s", color=NN, ms=4.5, lw=1.3, label="M4 cosine")
    for a, d1, d2 in zip(al, det_id, det_nn):
        ax1.annotate("", xy=(a, d1), xytext=(a, d2),
                     arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.7))
    ax1.set_xlabel("nominal significance level $\\alpha$")
    ax1.set_ylabel("novel-genus detection rate")
    ax1.set_xlim(0, 0.215)
    ax1.set_ylim(0, 0.62)
    ax1.legend(fontsize=7.6, loc="upper left", handlelength=1.5)
    ax1.set_xticks([0.0, 0.05, 0.10, 0.15, 0.20])
    ax1.grid(True)
    ax1.set_axisbelow(True)
    panel_label(ax1, "b", x=-0.2, y=1.04)

    idx = np.arange(2)
    w = 0.34
    lso = [h.loc["LSO family recovery", "identity"], h.loc["LSO family recovery", "m4_cosine"]]
    lgo = [h.loc["LGO family recovery", "identity"], h.loc["LGO family recovery", "m4_cosine"]]
    ax2.bar(idx - w / 2, [100 * x for x in lso], w, color=[ID, NN],
            edgecolor="white", label="LSO")
    ax2.bar(idx + w / 2, [100 * x for x in lgo], w, color=[ID, NN],
            alpha=0.5, edgecolor="white", hatch="///", label="LGO")
    for x, v in zip(idx - w / 2, lso):
        ax2.text(x, 100 * v + 1.5, f"{100*v:.1f}", ha="center", fontsize=7.2)
    for x, v in zip(idx + w / 2, lgo):
        ax2.text(x, 100 * v + 1.5, f"{100*v:.1f}", ha="center", fontsize=7.2)
    ax2.set_xticks(idx)
    ax2.set_xticklabels(["percent\nidentity", "M4\ncosine"], fontsize=8)
    ax2.set_ylim(0, 108)
    ax2.set_ylabel("family recovery (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="#999999"),
               plt.Rectangle((0, 0), 1, 1, facecolor="#999999", alpha=0.5, hatch="///")]
    ax2.legend(handles, ["known species (LSO)", "novel genus (LGO)"],
               fontsize=7, loc="upper center", ncol=1, handlelength=1.0,
               bbox_to_anchor=(0.5, 1.14))
    ax2.yaxis.grid(True)
    ax2.set_axisbelow(True)
    panel_label(ax2, "c", x=-0.3, y=1.04)
    save(fig, "fig6_historical_identity_vs_cosine")


# --------------------------------------------------------------- figure S1 ---
def figureS1():
    a = audit.set_index("item")["value"]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5), gridspec_kw={"wspace": 0.42})
    ax0, ax1 = axes

    cats = ["LGO queries\n(exact IDs)", "LGO query\ngenera",
            "LSO queries\n(exact IDs)", "LSO query\nspecies"]
    totals = [a["Historical LGO queries"], a["Historical LGO query genera"],
              a["Historical LSO queries"], a["Historical LSO query species"]]
    leaked = [a["LGO exact query IDs in original TRAIN_REF"],
              a["LGO query genera in original TRAIN_REF"],
              a["LSO exact query IDs in original TRAIN_REF"],
              a["LSO query species in original TRAIN_REF"]]
    y = np.arange(4)
    ax0.barh(y, totals, 0.6, color="#E5E5E5", edgecolor="#999999", linewidth=0.6,
             label="held-out set")
    ax0.barh(y, leaked, 0.6, color=RED, edgecolor="white", linewidth=0.6,
             label="present in original TRAIN_REF")
    for i, (t, l) in enumerate(zip(totals, leaked)):
        ax0.text(t + max(totals) * 0.015, i, f"{l:,} / {t:,}  ({100*l/t:.0f}%)",
                 va="center", fontsize=7.2)
    ax0.set_yticks(y)
    ax0.set_yticklabels(cats, fontsize=7.6)
    ax0.invert_yaxis()
    ax0.set_xlim(0, max(totals) * 1.42)
    ax0.set_xlabel("count")
    ax0.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=2, handlelength=1.0, columnspacing=1.2)
    panel_label(ax0, "a", x=-0.32, y=1.12)

    before, after = a["Original TRAIN_REF records"], a["Leakage-safe TRAIN_REF records"]
    ax1.bar([0, 1], [before, after], 0.5, color=[GREY, GREEN], edgecolor="white")
    ax1.text(0, before * 1.01, f"{before:,}", ha="center", fontsize=8)
    ax1.text(1, after * 1.01, f"{after:,}", ha="center", fontsize=8)
    ax1.annotate(f"$-${a['Excluded records']:,} records\n(every leaked taxon removed)",
                 xy=(0.5, before * 0.55), fontsize=7.4, ha="center", color=DARK)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["original\nTRAIN_REF", "leakage-safe\nTRAIN_REF"], fontsize=8)
    ax1.set_ylim(0, before * 1.18)
    ax1.set_ylabel("training records")
    ax1.yaxis.grid(True)
    ax1.set_axisbelow(True)
    ax1.yaxis.set_major_formatter(lambda v, pos: f"{int(v):,}")
    panel_label(ax1, "b", x=-0.26, y=1.12)
    save(fig, "figS1_leakage_audit")


if __name__ == "__main__":
    print("Regenerating figures from data/ ...")
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure6()
    figureS1()
    print("done.")
