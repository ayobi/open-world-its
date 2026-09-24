"""Tables, figures and prose macros for the corrected primary evaluation (v3.6).

Run after make_tables.py and make_figures.py:
    python3 analysis/make_primary.py

Everything here reads data/*.csv. Nothing is typed by hand except labels.
The older generators still produce the sealed development ladder, cross-view
matrix and historical benchmark, which the manuscript now reports in the
appendix as sealed values pending the corrected development rescore.

numbers.tex holds a LaTeX macro for every number the prose quotes from these
results, so the text cannot drift from the tables beside it.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from style import (use_style, save, panel_label, DATADIR, BLUE, RED, GREEN,
                   PURPLE, GREY, DARK, VIEW_COLOUR, VIEW_LABEL, VIEW_ORDER)

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "tables"
use_style()

m4x = pd.read_csv(DATADIR / "m4_test_exact.csv").set_index("view")
m4s = pd.read_csv(DATADIR / "frozen_test_metrics.csv").set_index("view")
idt = pd.read_csv(DATADIR / "identity_test_cov08.csv").set_index("view")
m4c = pd.read_csv(DATADIR / "m4_conformal_exact.csv")
idc = pd.read_csv(DATADIR / "identity_conformal_cov08.csv")
unc = pd.read_csv(DATADIR / "paired_uncertainty_test.csv").set_index("view")
dev = pd.read_csv(DATADIR / "dev_identity_sameview.csv").set_index("view")
xv = pd.read_csv(DATADIR / "dev_crossview_identity.csv").set_index("query_view")
aud = pd.read_csv(DATADIR / "search_correction_audit.csv")
cov = pd.read_csv(DATADIR / "coverage_sweep.csv")
cfg = pd.read_csv(DATADIR / "identity_config_test.csv")
pad = pd.read_csv(DATADIR / "padding_diagnostics.csv")
abl = pd.read_csv(DATADIR / "development_ablation.csv")

# M4 development values come from the sealed (padded) ladder; they are used only
# for the development-to-TEST decomposition, where the padding confound runs in
# the conservative direction (see the Results).
M4DEV = {v: abl[(abl.model == "M4") & (abl.view == v)].iloc[0] for v in VIEW_ORDER}
ID = PURPLE
M4 = GREEN


def P(x, d=1):
    return f"{100*x:.{d}f}\\%"


def A(x):
    return f"{x:.3f}"


def write(name, body):
    (TAB / name).write_text(body.rstrip() + "\n", encoding="utf-8")
    print(f"  wrote tables/{name}")


def op(df, v, a, col):
    return float(df[(df.view == v) & (np.isclose(df.alpha, a))][col].iloc[0])


# ------------------------------------------------------------------ tables ---
def tab_headtohead():
    b = [r"\begin{table}[htbp]\centering", r"\begin{threeparttable}",
         r"\caption{Primary TEST evaluation: M4 against a matched alignment "
         r"baseline on identical queries and references. The sealed row is the "
         r"locked evaluation as run, with padded inference; the corrected row "
         r"recomputes the same checkpoint with exact-length batching. Identity is "
         r"exhaustive VSEARCH with query coverage $\geq0.8$. Best value per view "
         r"and column in bold.}",
         r"\label{tab:headtohead}", r"\small",
         r"\begin{tabular}{llccccc}", r"\toprule",
         r" & & & known & \multicolumn{3}{c}{novel-genus placement} \\",
         r"\cmidrule(l){5-7}",
         r"view & score & AUROC & genus NN & family & order & class \\", r"\midrule"]
    cols = [("auroc", A), ("known_genus", P), ("novel_family", P),
            ("novel_order", P), ("novel_class", P)]
    for k, v in enumerate(VIEW_ORDER):
        rows = [("M4, sealed (padded)", m4s.loc[v]), ("M4, corrected", m4x.loc[v]),
                ("percent identity", idt.loc[v])]
        best = {c: max(float(r[c]) for _, r in rows) for c, _ in cols}
        for i, (lab, r) in enumerate(rows):
            cells = []
            for c, f in cols:
                t = f(float(r[c]))
                if abs(float(r[c]) - best[c]) < 1e-12:
                    t = r"\textbf{" + t + "}"
                cells.append(t)
            name = VIEW_LABEL[v] if i == 0 else ""
            b.append(f"{name} & {lab} & " + " & ".join(cells) + r" \\")
        if k < 2:
            b.append(r"\addlinespace")
    b += [r"\bottomrule", r"\end{tabular}", r"\begin{tablenotes}\footnotesize",
          r"\item $n$ known / novel: " + "; ".join(
              f"{VIEW_LABEL[v]} {int(m4x.loc[v,'n_test_known']):,}/{int(m4x.loc[v,'n_test_novel']):,}"
              for v in VIEW_ORDER) + ". Identity censors queries with no hit at "
          r"identity $\geq0.5$ and coverage $\geq0.8$ at $49.999$ and counts them as "
          r"placement failures: " + "; ".join(
              f"{VIEW_LABEL[v]} {int(idt.loc[v,'censored_known'])} known, "
              f"{int(idt.loc[v,'censored_novel'])} novel" for v in VIEW_ORDER) + ".",
          r"\item Paired uncertainty for the AUROC and detection contrasts is in "
          r"Table~\ref{tab:uncertainty}.",
          r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
    write("tabP1_headtohead.tex", "\n".join(b))


def tab_uncertainty():
    b = [r"\begin{table}[htbp]\centering", r"\begin{threeparttable}",
         r"\caption{Paired uncertainty for corrected M4 against identity on TEST, "
         r"M4 minus identity. The query-level DeLong interval treats queries as "
         r"independent; the genus-cluster bootstrap resamples whole genera "
         r"separately within the known and novel classes, applies identical "
         r"weights to both scores, and uses 10,000 replicates.}",
         r"\label{tab:uncertainty}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lccccc}", r"\toprule",
         r"\multicolumn{6}{l}{\emph{(a) Novelty AUROC}} \\",
         r"view & $\Delta$AUROC & DeLong 95\% CI & $P$ & genus 95\% CI & Bonferroni, 3 views \\",
         r"\midrule"]
    for v in VIEW_ORDER:
        u = unc.loc[v]
        pv = (f"{u.q_p:.1e}".replace("e-05", r"\times10^{-5}").replace("e-04", r"\times10^{-4}")
              if u.q_p < 0.001 else f"{u.q_p:.3f}")
        b.append(f"{VIEW_LABEL[v]} & ${u['diff']:+.3f}$ & ${u.q_lo:+.3f}$ to ${u.q_hi:+.3f}$ & "
                 f"${pv}$ & ${u.lo:+.3f}$ to ${u.hi:+.3f}$ & ${u.bonf_lo:+.3f}$ to ${u.bonf_hi:+.3f}$ \\\\")
    b += [r"\midrule", r"\multicolumn{6}{l}{\emph{(b) Novel detection at nominal $\alpha=0.05$}} \\",
          r"view & M4 & identity & $\Delta$, pp & genus 95\% CI & false novelty, M4 / identity \\",
          r"\midrule"]
    for v in VIEW_ORDER:
        u = unc.loc[v]
        b.append(f"{VIEW_LABEL[v]} & {P(u.det_cosine)} & {P(u.det_identity)} & ${100*u.det_diff:+.1f}$ & "
                 f"${100*u.det_lo:+.1f}$ to ${100*u.det_hi:+.1f}$ & {P(u.fn_cosine)} / {P(u.fn_identity)} \\\\")
    b += [r"\bottomrule", r"\end{tabular}", r"\begin{tablenotes}\footnotesize",
          r"\item Novel genera per view: " + ", ".join(
              f"{VIEW_LABEL[v]} {int(unc.loc[v,'n_novel_genera'])}" for v in VIEW_ORDER)
          + r". Point estimates are query-weighted; the genus-cluster interval is "
          r"for a genus-resampled version of that statistic. Rate intervals hold "
          r"the CAL\_KNOWN scores fixed, so they omit calibration-set sampling "
          r"variability and understate total uncertainty.",
          r"\item An interval that includes zero means no demonstrated advantage, "
          r"not equivalence.",
          r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
    write("tabP2_uncertainty.tex", "\n".join(b))


def tab_operating():
    b = [r"\begin{table}[htbp]\centering", r"\begin{threeparttable}",
         r"\caption{Conformal operating points on TEST for corrected M4 and "
         r"identity, each calibrated on its own CAL\_KNOWN scores by the same "
         r"procedure. False novelty is the fraction of TEST\_KNOWN flagged; "
         r"detection the fraction of TEST\_NOVEL flagged.}",
         r"\label{tab:operating}", r"\small",
         r"\begin{tabular}{llcccc}", r"\toprule",
         r" & & \multicolumn{2}{c}{M4, corrected} & \multicolumn{2}{c}{percent identity} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"view & $\alpha$ & false novelty & detection & false novelty & detection \\",
         r"\midrule"]
    for k, v in enumerate(VIEW_ORDER):
        for i, a in enumerate([0.01, 0.05, 0.10, 0.20]):
            mf, md = op(m4c, v, a, "false_novelty"), op(m4c, v, a, "novel_detection")
            jf, jd = op(idc, v, a, "false_novelty"), op(idc, v, a, "novel_detection")
            dm = r"\textbf{" + P(md) + "}" if md > jd else P(md)
            dj = r"\textbf{" + P(jd) + "}" if jd > md else P(jd)
            name = VIEW_LABEL[v] if i == 0 else ""
            b.append(f"{name} & {a:.2f} & {P(mf)} & {dm} & {P(jf)} & {dj} \\\\")
        if k < 2:
            b.append(r"\addlinespace")
    b += [r"\bottomrule", r"\end{tabular}", r"\begin{tablenotes}\footnotesize",
          r"\item Higher detection per row in bold. At $\alpha\in\{0.05,0.10\}$ "
          r"identity detects more novel queries at a lower observed false-novelty "
          r"rate at every view. At $\alpha=0.01$ both scores detect almost "
          r"nothing, and ITS1 identity flags no query at all.",
          r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
    write("tabP3_operating.tex", "\n".join(b))


def tab_baseline():
    b = [r"\begin{table}[htbp]\centering", r"\begin{threeparttable}",
         r"\caption{How the alignment baseline depends on its configuration. "
         r"(a) Default termination heuristics against an exhaustive search on the "
         r"historical ITS2 benchmark. (b) Query-coverage filtering on TEST\_NOVEL: "
         r"the ITS-core rows sweep the threshold and the spacer rows are the "
         r"control, since ITS1 and ITS2 contain no conserved region.}",
         r"\label{tab:baseline}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lccccc}", r"\toprule",
         r"\multicolumn{6}{l}{\emph{(a) Default heuristics against exhaustive search, historical ITS2}} \\",
         r"query set & with a hit & lower-identity hit & target changed & rescued & mean gain \\", r"\midrule"]
    for _, r in aud.iterrows():
        b.append(f"{r['set']} & {int(r.shared):,} & {int(r.improved):,} ({100*r.improved/r.shared:.1f}\\%) & "
                 f"{int(r.changed):,} ({100*r.changed/r.shared:.1f}\\%) & {int(r.rescued)} & "
                 f"${r.mean_gain:+.2f}$ pp \\\\")
    b += [r"\midrule",
          r"\multicolumn{6}{l}{\emph{(b) Novel-genus placement by query-coverage filter, TEST}} \\",
          r"view & coverage filter & family & order & class & \\", r"\midrule"]
    for k, v in enumerate(VIEW_ORDER):
        sub = cov[cov.view == v]
        for i, (_, r) in enumerate(sub.iterrows()):
            name = VIEW_LABEL[v] if i == 0 else ""
            b.append(f"{name} & {r.setting} & {P(r.family)} & {P(r.order)} & {P(r.cls)} & \\\\")
        if k < 2:
            b.append(r"\addlinespace")
    b += [r"\bottomrule", r"\end{tabular}", r"\begin{tablenotes}\footnotesize",
          r"\item In (b), 44.5\% of ITS-core best hits changed between no filter "
          r"and coverage $0.8$, with a mean identity change of $-2.53$ points: "
          r"without the filter the search preferred higher-identity partial "
          r"alignments that were taxonomically wrong. Identity AUROC was "
          r"unchanged by the filter to three decimals at every view "
          r"(Table~\ref{tab:config}). In (a), a lower-identity hit is one an "
          r"exhaustive search improved on; a changed target also counts swaps "
          r"among exact ties. Panel (b) is an independent search run: because "
          r"multithreaded VSEARCH does not resolve exact ties deterministically, "
          r"its $0.8$ row can differ from Table~\ref{tab:headtohead} by up to "
          r"$0.1$ point.",
          r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
    write("tabP4_baseline.tex", "\n".join(b))

    c = [r"\begin{table}[htbp]\centering", r"\begin{threeparttable}",
         r"\caption{The coverage filter changes placement and leaves novelty "
         r"ranking untouched. Identity on TEST under exhaustive search, with and "
         r"without query coverage $\geq0.8$.}",
         r"\label{tab:config}", r"\small",
         r"\begin{tabular}{llccc}", r"\toprule",
         r"view & configuration & AUROC & known genus NN & novel family \\", r"\midrule"]
    for k, v in enumerate(VIEW_ORDER):
        sub = cfg[cfg.view == v]
        for i, (_, r) in enumerate(sub.iterrows()):
            name = VIEW_LABEL[v] if i == 0 else ""
            c.append(f"{name} & {r.config} & {r.auroc:.4f} & {P(r.known_genus)} & {P(r.novel_family)} \\\\")
        if k < 2:
            c.append(r"\addlinespace")
    c += [r"\bottomrule", r"\end{tabular}", r"\end{threeparttable}", r"\end{table}"]
    write("tabP4b_config.tex", "\n".join(c))


def tab_padding():
    b = [r"\begin{table}[htbp]\centering", r"\begin{threeparttable}",
         r"\caption{Dependence of M4 embeddings on inference batching, measured "
         r"as the change in each query's maximum cosine to TRAIN\_REF. Same "
         r"hash-verified checkpoint, same sequences, same device.}",
         r"\label{tab:padding}", r"\small",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"comparison & mean $|\Delta|$ & max $|\Delta|$ & signed mean \\", r"\midrule"]
    for _, r in pad.iterrows():
        mx = "---" if pd.isna(r.max_abs) else f"{r.max_abs:.3f}"
        sg = "---" if pd.isna(r.signed) else f"${r.signed:+.4f}$"
        b.append(f"{r.comparison.replace('_', chr(92)+'_')} & {r.mean_abs:.4f} & {mx} & {sg} \\\\")
    b += [r"\bottomrule", r"\end{tabular}", r"\begin{tablenotes}\footnotesize",
          r"\item Batches of 64 and 256 are both heavily padded and differ little "
          r"from each other; both differ from unpadded batch-1 inference by about "
          r"five times as much. Between unpadded and padded scoring, 35.2\% of "
          r"TEST\_KNOWN ITS-core queries changed their nearest reference and "
          r"16.6\% changed its genus. The magnitude of the shift correlates only "
          r"weakly with sequence length (Spearman $\rho=-0.11$).",
          r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
    write("tabP5_padding.tex", "\n".join(b))


def tab_devtest():
    b = [r"\begin{table}[htbp]\centering", r"\begin{threeparttable}",
         r"\caption{Development-to-TEST movement separated into partition "
         r"composition and selection. Identity is parameter-free and nothing was "
         r"selected on it, so its gap estimates composition alone. Same-view "
         r"retrieval throughout.}",
         r"\label{tab:devtest}", r"\small",
         r"\begin{tabular}{lcccccccc}", r"\toprule",
         r" & \multicolumn{3}{c}{identity AUROC} & \multicolumn{3}{c}{M4 AUROC} & "
         r"residual & \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"view & DEV & TEST & gap & DEV & TEST & gap & (M4 $-$ identity) & \\", r"\midrule"]
    for v in VIEW_ORDER:
        idg = dev.loc[v, "auroc"] - idt.loc[v, "auroc"]
        md = float(M4DEV[v]["auroc"])
        mg = md - m4x.loc[v, "auroc"]
        b.append(f"{VIEW_LABEL[v]} & {dev.loc[v,'auroc']:.3f} & {idt.loc[v,'auroc']:.3f} & "
                 f"${idg:+.3f}$ & {md:.3f} & {m4x.loc[v,'auroc']:.3f} & ${mg:+.3f}$ & "
                 f"${mg-idg:+.3f}$ & \\\\")
    b += [r"\addlinespace",
          r"\multicolumn{9}{l}{\emph{Novel-family placement, DEV minus TEST}} \\"]
    for v in VIEW_ORDER:
        idg = dev.loc[v, "novel_family"] - idt.loc[v, "novel_family"]
        mg = float(M4DEV[v]["novel_family"]) - m4x.loc[v, "novel_family"]
        b.append(f"{VIEW_LABEL[v]} & \\multicolumn{{8}}{{l}}{{identity "
                 f"${100*idg:+.1f}$ pp; M4 ${100*mg:+.1f}$ pp}} \\\\")
    b += [r"\bottomrule", r"\end{tabular}", r"\begin{tablenotes}\footnotesize",
          r"\item Both scores use corrected inference throughout: exact-length "
          r"batching for M4 and exhaustive, coverage-filtered search for identity. "
          r"No view shows a positive residual.",
          r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
    write("tabP6_devtest.tex", "\n".join(b))


def tab_crossview():
    b = [r"\begin{table}[htbp]\centering", r"\begin{threeparttable}",
         r"\caption{Cross-view retrieval through the ITS-core references on DEV. "
         r"Spacer queries searched against ITS-core references, by exhaustive "
         r"VSEARCH (query coverage $\geq0.8$) and by M4 cosine. Because ITS-core "
         r"contains both spacers, this is a homologous comparison that alignment "
         r"can make directly.}",
         r"\label{tab:crossview}", r"\small",
         r"\begin{tabular}{llccc}", r"\toprule",
         r"query view & score & AUROC & known genus NN & novel family \\", r"\midrule"]
    for k, q in enumerate(["its1", "its2"]):
        r = xv.loc[q]
        b.append(f"{VIEW_LABEL[q]} & percent identity & \\textbf{{{r.identity_auroc:.3f}}} & "
                 f"\\textbf{{{P(r.identity_known)}}} & \\textbf{{{P(r.identity_family)}}} \\\\")
        b.append(f" & M4 cosine & {r.m4_auroc:.3f} & {P(r.m4_known)} & {P(r.m4_family)} \\\\")
        if k == 0:
            b.append(r"\addlinespace")
    b += [r"\bottomrule", r"\end{tabular}", r"\begin{tablenotes}\footnotesize",
          r"\item M4 values use exact-length inference; under the sealed, padded "
          r"inference they were about 12 points higher in genus accuracy and 8 to 9 "
          r"in family placement (Section~\ref{sec:invariance}). M4 was selected on "
          r"DEV, so these cells if anything flatter it. Direct ITS1-to-ITS2 "
          r"retrieval, the only genuinely non-homologous pair, has M4 AUROC "
          r"$0.490$ and $0.471$ (Table~\ref{tab:m4matrix}).",
          r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
    write("tabP7_crossview.tex", "\n".join(b))


# ----------------------------------------------------------------- figures ---
def fig_primary():
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.7),
                             gridspec_kw={"width_ratios": [1.15, 1, 1], "wspace": 0.42})
    ax0, ax1, ax2 = axes
    y = np.arange(3)[::-1]
    for i, v in zip(y, VIEW_ORDER):
        u = unc.loc[v]
        ax0.plot([u.lo, u.hi], [i, i], color=GREY, lw=5, solid_capstyle="butt", zorder=1)
        ax0.plot([u.q_lo, u.q_hi], [i, i], color=DARK, lw=1.6, zorder=2)
        ax0.plot([u["diff"]], [i], "o", color=DARK, ms=5.5, zorder=3,
                 mec="white", mew=0.8)
    ax0.axvline(0, color=DARK, ls=(0, (4, 3)), lw=0.8)
    ax0.set_yticks(y)
    ax0.set_yticklabels([VIEW_LABEL[v] for v in VIEW_ORDER])
    ax0.set_xlabel("AUROC, M4 minus identity")
    ax0.set_xlim(-0.13, 0.13)
    ax0.text(0.02, 1.02, "grey: genus-cluster 95% CI\nblack: query-level DeLong",
             transform=ax0.transAxes, fontsize=6.6, va="bottom", color="#555555")
    panel_label(ax0, "a", x=-0.30, y=1.15)

    w = 0.36
    x = np.arange(3)
    for ax, col, lab, letter in [(ax1, "known_genus", "known-genus NN (%)", "b"),
                                 (ax2, "novel_family", "novel-genus family (%)", "c")]:
        mv = [100 * m4x.loc[v, col] for v in VIEW_ORDER]
        iv = [100 * idt.loc[v, col] for v in VIEW_ORDER]
        ax.bar(x - w / 2, mv, w, color=M4, label="M4, corrected", edgecolor="white")
        ax.bar(x + w / 2, iv, w, color=ID, label="percent identity", edgecolor="white")
        for xi, a, b_ in zip(x, mv, iv):
            ax.text(xi - w / 2, a + 1.2, f"{a:.0f}", ha="center", fontsize=6.8)
            ax.text(xi + w / 2, b_ + 1.2, f"{b_:.0f}", ha="center", fontsize=6.8)
        ax.set_xticks(x)
        ax.set_xticklabels([VIEW_LABEL[v] for v in VIEW_ORDER], fontsize=7.8)
        ax.set_ylim(0, 100)
        ax.set_ylabel(lab)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        panel_label(ax, letter, x=-0.30, y=1.15)
    ax1.legend(fontsize=6.8, loc="lower left", handlelength=1.0,
               bbox_to_anchor=(0.0, 1.0), ncol=1)
    save(fig, "fig4_primary")


def fig_operating():
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.35), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    for ax, v in zip(axes, VIEW_ORDER):
        for df, col, lab, mk in [(m4c, M4, "M4, corrected", "s"),
                                 (idc, ID, "percent identity", "o")]:
            s = df[df.view == v].sort_values("alpha")
            ax.plot(100 * s.false_novelty, 100 * s.novel_detection, "-" + mk,
                    color=col, ms=4, lw=1.3, label=lab)
            for _, r in s.iterrows():
                if r.alpha in (0.05, 0.10):
                    off = (-14, 5) if mk == "o" else (5, -9)
                    ax.annotate(f"{r.alpha:.2f}", (100 * r.false_novelty, 100 * r.novel_detection),
                                textcoords="offset points", xytext=off, fontsize=6.2,
                                color=col)
        ax.set_title(VIEW_LABEL[v], fontsize=8.5)
        ax.set_xlim(0, 19)
        ax.set_ylim(0, 50)
        ax.set_xlabel("observed false novelty (%)")
        ax.grid(True)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("novel-genus detection (%)")
    axes[0].legend(fontsize=6.8, loc="upper left", handlelength=1.6)
    save(fig, "fig5_operating")


def fig_baseline():
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.55),
                             gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.36})
    ax0, ax1 = axes
    core = cov[cov.view == "core"].copy()
    xs = [0, 1, 2, 3, 4]
    for rank, col, lab in [("family", "#1F3B73", "ITS-core family"),
                           ("order", "#4477AA", "ITS-core order"),
                           ("cls", "#9DB9DD", "ITS-core class")]:
        ax0.plot(xs, 100 * core[rank].to_numpy(), "-o", color=col, ms=4, lw=1.4, label=lab)
    for v, mk in [("its1", "^"), ("its2", "v")]:
        s = cov[(cov.view == v)]
        ax0.plot([0, 3], 100 * s.family.to_numpy(), ls=(0, (3, 2)), marker=mk,
                 color=VIEW_COLOUR[v], ms=4.5, lw=1.0, label=f"{VIEW_LABEL[v]} family (control)")
    ax0.set_xticks(xs)
    ax0.set_xticklabels(["none", "0.5", "0.7", "0.8", "0.9"])
    ax0.set_xlabel("query-coverage filter")
    ax0.set_ylabel("ITS-core novel placement (%)")
    ax0.set_ylim(45, 100)
    ax0.grid(True)
    ax0.set_axisbelow(True)
    ax0.legend(fontsize=6.4, loc="lower right", ncol=1, handlelength=1.4)
    panel_label(ax0, "a", x=-0.18, y=1.04)

    labs = ["LGO\n(novel genus)", "LSO\n(known genus)"]
    pc = [100 * r.changed / r.shared for _, r in aud.iterrows()]
    ax1.bar([0, 1], pc, 0.55, color=[RED, BLUE], edgecolor="white")
    for i, (p_, g) in enumerate(zip(pc, aud.mean_gain)):
        ax1.text(i, p_ + 1.5, f"{p_:.0f}%\n(+{g:.1f} pp)", ha="center", fontsize=7.2)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(labs, fontsize=7.8)
    ax1.set_ylim(0, 80)
    ax1.set_ylabel("best hit changed by\nexhaustive search (%)")
    ax1.yaxis.grid(True)
    ax1.set_axisbelow(True)
    panel_label(ax1, "b", x=-0.30, y=1.04)
    save(fig, "fig7_baseline_config")


# ----------------------------------------------------------------- numbers ---
def numbers():
    L = ["% generated by analysis/make_primary.py; do not edit by hand"]

    def m(name, val):
        L.append(f"\\newcommand{{\\{name}}}{{{val}}}")
    vk = {"core": "Core", "its1": "One", "its2": "Two"}
    for v in VIEW_ORDER:
        k = vk[v]
        m(f"MfAUC{k}", A(m4x.loc[v, "auroc"]))
        m(f"MsAUC{k}", A(m4s.loc[v, "auroc"]))
        m(f"IdAUC{k}", A(idt.loc[v, "auroc"]))
        m(f"MfKnown{k}", P(m4x.loc[v, "known_genus"]))
        m(f"IdKnown{k}", P(idt.loc[v, "known_genus"]))
        m(f"MfFam{k}", P(m4x.loc[v, "novel_family"]))
        m(f"IdFam{k}", P(idt.loc[v, "novel_family"]))
        m(f"MfOrd{k}", P(m4x.loc[v, "novel_order"]))
        m(f"IdOrd{k}", P(idt.loc[v, "novel_order"]))
        m(f"MfCls{k}", P(m4x.loc[v, "novel_class"]))
        m(f"IdCls{k}", P(idt.loc[v, "novel_class"]))
        u = unc.loc[v]
        m(f"DAUC{k}", f"{u['diff']:+.3f}")
        m(f"GbLo{k}", f"{u.lo:+.3f}")
        m(f"GbHi{k}", f"{u.hi:+.3f}")
        m(f"MfDet{k}", P(op(m4c, v, 0.05, "novel_detection")))
        m(f"IdDet{k}", P(op(idc, v, 0.05, "novel_detection")))
        m(f"MfFn{k}", P(op(m4c, v, 0.05, "false_novelty")))
        m(f"IdFn{k}", P(op(idc, v, 0.05, "false_novelty")))
        m(f"IdGap{k}", f"{dev.loc[v,'auroc']-idt.loc[v,'auroc']:+.3f}")
        m(f"MfGap{k}", f"{float(M4DEV[v]['auroc'])-m4x.loc[v,'auroc']:+.3f}")
        m(f"NGen{k}", f"{int(u.n_novel_genera)}")
    for q, k in [("its1", "One"), ("its2", "Two")]:
        r = xv.loc[q]
        m(f"XIdKnown{k}", P(r.identity_known)); m(f"XIdFam{k}", P(r.identity_family))
        m(f"XIdAUC{k}", A(r.identity_auroc)); m(f"XMKnown{k}", P(r.m4_known))
        m(f"XMFam{k}", P(r.m4_family)); m(f"XMAUC{k}", A(r.m4_auroc))
    lgo, lso = aud.iloc[0], aud.iloc[1]
    m("AudLgoPct", f"{100*lgo.changed/lgo.shared:.1f}\\%")
    m("AudLsoPct", f"{100*lso.changed/lso.shared:.1f}\\%")
    m("AudAllPct", f"{100*(lgo.changed+lso.changed)/(lgo.shared+lso.shared):.1f}\\%")
    m("AudAllImpPct", f"{100*(lgo.improved+lso.improved)/(lgo.shared+lso.shared):.1f}\\%")
    m("AudLgoGain", f"{lgo.mean_gain:.2f}"); m("AudLsoGain", f"{lso.mean_gain:.2f}")
    c = cov[cov.view == "core"].set_index("setting")
    m("CovNoneFam", P(c.loc["none", "family"])); m("CovNoneCls", P(c.loc["none", "cls"]))
    # prose quotes the authoritative head-to-head run, not the independent sweep run
    m("CovEightFam", P(idt.loc["core", "novel_family"])); m("CovEightCls", P(idt.loc["core", "novel_class"]))
    (ROOT / "numbers.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"  wrote numbers.tex ({len(L)-1} macros)")


if __name__ == "__main__":
    print("Building corrected primary-evaluation artefacts ...")
    tab_headtohead(); tab_uncertainty(); tab_operating(); tab_baseline()
    tab_padding(); tab_devtest(); tab_crossview()
    fig_primary(); fig_operating(); fig_baseline()
    numbers()
    print("done.")
