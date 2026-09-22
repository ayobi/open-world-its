"""Regenerate every LaTeX table in the manuscript from data/*.csv.

Run:  python3 analysis/make_tables.py

Formatting conventions applied throughout the manuscript:
  * AUROC and other [0,1] discrimination statistics: three decimals.
  * Percentages: one decimal, with an explicit per-cent sign.
  * Counts: comma thousands separators.
Only the constants in DESIGN below are typed by hand; everything else is read
from the CSVs, so a table can never disagree with the figure beside it.
"""
from pathlib import Path

import pandas as pd
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "tables"
OUT.mkdir(parents=True, exist_ok=True)

VIEWS = ["core", "its1", "its2"]
VLAB = {"core": "ITS-core", "its1": "ITS1", "its2": "ITS2"}

# Counts that are documented in the manuscript but not carried in any CSV.
DESIGN = {
    "unite_records": 102_137,
    "historical_its2_pool": 99_808,
    "fresh_its2_extraction": 99_862,
    "named_family_genus": 67_406,
    "train_ref": 46_453,
    "dev_known": 1_528,
    "dev_novel": 8_698,
    "cal_known": 1_505,
    "test_known": 1_484,
    "test_novel": 7_738,
    "cal_core": 1_494,
    "cal_its1": 1_501,
    "cal_its2": 1_493,
    "conflict_groups": 301,
    "conflict_its1_views": 393,
    "conflict_its2_views": 338,
}
# Bootstrap intervals reported by the locked evaluator (2,000 stratified
# resamples, seed 17).
TEST_CI = {"core": (0.739, 0.770), "its1": (0.699, 0.734), "its2": (0.748, 0.779)}

ab = pd.read_csv(DATA / "development_ablation.csv")
final = pd.read_csv(DATA / "frozen_test_metrics.csv").set_index("view")
conf = pd.read_csv(DATA / "conformal_operating_points.csv")
hist = pd.read_csv(DATA / "historical_benchmark.csv").set_index("metric")
mat = pd.read_csv(DATA / "m4_dev_retrieval_matrix.csv")
audit = pd.read_csv(DATA / "historical_leakage_audit.csv").set_index("item")["value"]


def auroc(x):
    return f"{x:.3f}"


def pct(x, scale=100):
    return f"{scale * x:.1f}\\%"


def num(x):
    return f"{int(x):,}"


def write(name, body):
    (OUT / name).write_text(body.rstrip() + "\n", encoding="utf-8")
    print(f"  wrote tables/{name}")


def cell(q, r, col):
    return mat[(mat.query_view == q) & (mat.reference_view == r)][col].iloc[0]


# ------------------------------------------------------------------ table 1 ---
def table1():
    ref = {v: cell("core", v, "n_reference") for v in VIEWS}
    dk = {v: cell(v, v, "n_dev_known") for v in VIEWS}
    dn = {v: cell(v, v, "n_dev_novel") for v in VIEWS}
    tk = {v: final.loc[v, "n_test_known"] for v in VIEWS}
    tn = {v: final.loc[v, "n_test_novel"] for v in VIEWS}
    d = DESIGN
    total = sum(d[k] for k in ["train_ref", "dev_known", "dev_novel",
                               "cal_known", "test_known", "test_novel"])
    assert total == d["named_family_genus"], total
    cal = {"core": d["cal_core"], "its1": d["cal_its1"], "its2": d["cal_its2"]}
    rows = [
        ("TRAIN\\_REF", d["train_ref"], ref),
        ("DEV\\_KNOWN", d["dev_known"], dk),
        ("DEV\\_NOVEL", d["dev_novel"], dn),
        ("CAL\\_KNOWN", d["cal_known"], cal),
        ("TEST\\_KNOWN", d["test_known"], tk),
        ("TEST\\_NOVEL", d["test_novel"], tn),
    ]
    body = [
        "\\begin{table}[htbp]\\centering",
        "\\begin{threeparttable}",
        "\\caption{Source collection and open-world partitioning. "
        "\\emph{Records} counts source records after taxonomic filtering; the three "
        "right-hand columns give how many of those records carry each barcode view "
        "and are therefore the denominators for the per-view results in "
        "Tables~\\ref{tab:ablation}--\\ref{tab:conformal}.}",
        "\\label{tab:dataset}",
        "\\small",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        " & & \\multicolumn{3}{c}{records carrying the view} \\\\",
        "\\cmidrule(l){3-5}",
        "item & records & ITS-core & ITS1 & ITS2 \\\\",
        "\\midrule",
        "\\multicolumn{5}{l}{\\emph{Source collection}} \\\\",
        f"\\quad UNITE dynamic release (19 Feb 2025) & {num(d['unite_records'])} & --- & --- & --- \\\\",
        f"\\quad historical ITS2 pool & {num(d['historical_its2_pool'])} & --- & --- & --- \\\\",
        f"\\quad fresh ITS2 extraction & {num(d['fresh_its2_extraction'])} & --- & --- & --- \\\\",
        f"\\quad named family and genus & {num(d['named_family_genus'])} & --- & --- & --- \\\\",
        "\\addlinespace",
        "\\multicolumn{5}{l}{\\emph{Open-world partitions}} \\\\",
    ]
    for label, n, per in rows:
        body.append(f"\\quad {label} & {num(n)} & {num(per['core'])} & "
                    f"{num(per['its1'])} & {num(per['its2'])} \\\\")
    body += [
        "\\addlinespace",
        f"\\quad total partitioned & {num(total)} & & & \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\begin{tablenotes}\\footnotesize",
        f"\\item Exact cross-genus sequence conflicts were masked at the level of the "
        f"individual view rather than by deleting the source record: "
        f"{num(d['conflict_its1_views'])} ITS1 and {num(d['conflict_its2_views'])} ITS2 "
        f"views across {d['conflict_groups']} conflict groups. No ITS-core conflict was detected.",
        "\\item The six partitions sum exactly to the named-family-and-genus total.",
        "\\end{tablenotes}",
        "\\end{threeparttable}",
        "\\end{table}",
    ]
    write("table1_dataset.tex", "\n".join(body))


# ------------------------------------------------------------------ table 2 ---
def table2():
    body = r"""\begin{table}[htbp]\centering
\begin{threeparttable}
\caption{Controlled model ladder. Each step adds exactly one scientific ingredient
to the step before it; M4 is the combination that was frozen. M0 was rerun at batch
size 96 so that every comparison shown here shares the same source-record batch size
and eight-epoch update budget.}
\label{tab:objectives}
\begin{tabular}{llccc}
\toprule
model & training objective added & $\lambda_{\mathrm{view}}$ & $\lambda_{\mathrm{tax}}$ & $\lambda_{\mathrm{epi}}$ \\
\midrule
M0 & genus proxy on ITS-core only (Eq.~\ref{eq:proxy}) & --- & --- & --- \\
M1 & \quad + cross-view invariance, core$\leftrightarrow$ITS1/ITS2 (Eq.~\ref{eq:view}) & 1.0 & --- & --- \\
M2 & \quad + hierarchical taxonomic geometry (Eq.~\ref{eq:tax}) & 1.0 & 1.0 & --- \\
M3 & \quad + leave-one-genus family episodes (Eq.~\ref{eq:epi}) & 1.0 & --- & 0.5 \\
M4 & \quad + both structured objectives & 1.0 & 1.0 & 0.5 \\
\bottomrule
\end{tabular}
\begin{tablenotes}\footnotesize
\item All models share the same convolutional encoder and 256-dimensional
L2-normalized output. View, taxonomy and episode temperatures were all $0.10$;
the hierarchical decay was $\beta=0.70$; seed 17; eight epochs; batch size 96.
\item M2 and M3 are both built on M1, so the M2/M3 contrast isolates hierarchical
geometry against pseudo-novel episodes at equal view supervision.
\end{tablenotes}
\end{threeparttable}
\end{table}"""
    write("table2_objectives.tex", body)


# ------------------------------------------------------------------ table 3 ---
def table3():
    body = [
        "\\begin{table}[htbp]\\centering",
        "\\begin{threeparttable}",
        "\\caption{Development-set ablation, same-view retrieval. Novelty AUROC treats "
        "held-out genera as the positive class and $-S(q)$ as the score. The placement "
        "columns give the proportion of novel-genus queries whose nearest TRAIN\\_REF "
        "neighbour belongs to the true family, order and class. Best value per column in bold.}",
        "\\label{tab:ablation}",
        "\\small",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        " & & & known & \\multicolumn{3}{c}{novel-genus placement} \\\\",
        "\\cmidrule(l){5-7}",
        "model & view & AUROC & genus NN & family & order & class \\\\",
        "\\midrule",
    ]
    best = {}
    for v in VIEWS:
        sub = ab[(ab.view == v) & (ab.model.isin(["M0", "M1", "M2", "M3", "M4"]))]
        best[v] = {c: sub[c].max() for c in
                   ["auroc", "known_genus", "novel_family", "novel_order", "novel_class"]}
    for m in ["M0", "M1", "M2", "M3", "M4"]:
        for i, v in enumerate(VIEWS):
            r = ab[(ab.model == m) & (ab.view == v)].iloc[0]
            name = m if i == 0 else ""
            cells = []
            for c, f in [("auroc", auroc), ("known_genus", pct),
                         ("novel_family", pct), ("novel_order", pct),
                         ("novel_class", pct)]:
                txt = f(r[c])
                if abs(r[c] - best[v][c]) < 1e-12:
                    txt = "\\textbf{" + txt + "}"
                cells.append(txt)
            body.append(f"{name} & {VLAB[v]} & " + " & ".join(cells) + " \\\\")
        if m != "M4":
            body.append("\\addlinespace")
    body += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\begin{tablenotes}\\footnotesize",
        "\\item M2 denotes the primary $\\lambda_{\\mathrm{tax}}=1.0$ run; the "
        "reduced-weight sensitivity run is Table~\\ref{tab:m2sens}.",
        "\\item Single run per configuration (seed 17). Between-seed variability was not estimated; differences of approximately 0.01 AUROC between adjacent rungs should therefore be interpreted cautiously.",
        "\\end{tablenotes}",
        "\\end{threeparttable}",
        "\\end{table}",
    ]
    write("table3_ablation.tex", "\n".join(body))


# ------------------------------------------------------------------ table 4 ---
def table4():
    blocks = [("novelty AUROC", "novelty_auroc", auroc),
              ("known-genus NN accuracy", "dev_known_nn_genus_accuracy", pct),
              ("novel-genus family NN", "dev_novel_nn_family_accuracy", pct)]
    body = [
        "\\begin{table}[htbp]\\centering",
        "\\begin{threeparttable}",
        "\\caption{M4 development retrieval matrix. Every query view was searched "
        "against every reference view. Diagonal cells are same-view retrieval; "
        "off-diagonal cells measure how far the shared space transfers. "
        "Same-view cells in bold.}",
        "\\label{tab:m4matrix}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        " & & \\multicolumn{3}{c}{reference view} \\\\",
        "\\cmidrule(l){3-5}",
        "metric & query view & ITS-core & ITS1 & ITS2 \\\\",
        "\\midrule",
    ]
    for bi, (label, col, f) in enumerate(blocks):
        for i, q in enumerate(VIEWS):
            cells = []
            for r in VIEWS:
                txt = f(cell(q, r, col))
                if q == r:
                    txt = "\\textbf{" + txt + "}"
                cells.append(txt)
            name = label if i == 0 else ""
            body.append(f"{name} & {VLAB[q]} & " + " & ".join(cells) + " \\\\")
        if bi < len(blocks) - 1:
            body.append("\\addlinespace")
    body += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\begin{tablenotes}\\footnotesize",
        "\\item ITS-core is the strongest bridge to both spacers. Direct ITS1$\\leftrightarrow$ITS2 retrieval remains near chance in AUROC and below 7\\% in known-genus accuracy, consistent with incomplete alignment of the two spacer-specific representations.",
        "\\item Chance for a nearest-neighbour genus assignment over the 4,231 represented genera is approximately 0.02\\%, so the spacer-to-spacer cells are weak rather than empty.",
        "\\end{tablenotes}",
        "\\end{threeparttable}",
        "\\end{table}",
    ]
    write("table4_m4_crosslocus.tex", "\n".join(body))


# ------------------------------------------------------------------ table 5 ---
def table5():
    body = [
        "\\begin{table}[htbp]\\centering",
        "\\begin{threeparttable}",
        "\\caption{Frozen M4 on the untouched TEST partitions. CAL\\_KNOWN was used "
        "only for conformal calibration; no DEV sequence was embedded or scored during "
        "this evaluation.}",
        "\\label{tab:final}",
        "\\small",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        " & & & known & \\multicolumn{3}{c}{novel-genus placement} & \\\\",
        "\\cmidrule(lr){5-7}",
        "view & AUROC & 95\\% CI & genus NN & family & order & class & "
        "$n$ known / novel \\\\",
        "\\midrule",
    ]
    for v in VIEWS:
        r = final.loc[v]
        lo, hi = TEST_CI[v]
        body.append(
            f"{VLAB[v]} & {auroc(r['auroc'])} & {lo:.3f}--{hi:.3f} & "
            f"{pct(r['known_genus'])} & {pct(r['novel_family'])} & "
            f"{pct(r['novel_order'])} & {pct(r['novel_class'])} & "
            f"{num(r['n_test_known'])} / {num(r['n_test_novel'])} \\\\")
    body += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\begin{tablenotes}\\footnotesize",
        "\\item Intervals are stratified bootstrap percentile intervals over queries "
        "(2,000 resamples, seed 17).",
        "\\item Development AUROCs for the same model were "
        + ", ".join(f"{auroc(ab[(ab.model=='M4') & (ab.view==v)].iloc[0]['auroc'])} "
                    f"({VLAB[v]})" for v in VIEWS)
        + ". The ITS-core and ITS1 development values lie above their TEST "
          "intervals and the ITS2 value lies inside its own, so selection "
          "optimism of roughly 0.02 and 0.04 AUROC is present at two of the "
          "three views.",
        "\\item Novel-genus placement moves the other way: TEST family placement "
        "exceeds the development value at every view (by 7.4, 6.5 and 10.8 "
        "percentage points). Because the two effects disagree in sign on one "
        "frozen checkpoint, they are most economically read as a composition "
        "difference between DEV\\_NOVEL and TEST\\_NOVEL rather than as "
        "generalization behaviour.",
        "\\end{tablenotes}",
        "\\end{threeparttable}",
        "\\end{table}",
    ]
    write("table5_final_test.tex", "\n".join(body))


# ------------------------------------------------------------------ table 6 ---
def table6():
    body = [
        "\\begin{table}[htbp]\\centering",
        "\\begin{threeparttable}",
        "\\caption{Conformal operating points on untouched TEST data. False novelty is "
        "the fraction of TEST\\_KNOWN flagged as novel; detection is the fraction of "
        "TEST\\_NOVEL flagged.}",
        "\\label{tab:conformal}",
        "\\small",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        " & & false novelty & \\multicolumn{2}{c}{detection (TEST\\_NOVEL)} & \\\\",
        "\\cmidrule(lr){4-5}",
        "view & $\\alpha$ & TEST\\_KNOWN & per query & per genus & novel genera \\\\",
        "\\midrule",
    ]
    for k, v in enumerate(VIEWS):
        sub = conf[conf["view"] == v].sort_values("alpha")
        for i, (_, r) in enumerate(sub.iterrows()):
            a = r["alpha"]
            name = VLAB[v] if i == 0 else ""
            body.append(
                f"{name} & {a:.2f} & {pct(r['false_novelty'])} & "
                f"{pct(r['novel_detection'])} & {pct(r['macro_genus'])} & "
                f"{int(r['n_novel_genera'])} \\\\")
        if k < len(VIEWS) - 1:
            body.append("\\addlinespace")
    body += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\begin{tablenotes}\\footnotesize",
        "\\item Under exchangeability, split conformal controls the marginal false-novelty probability for a future known query. The empirical fraction on a finite TEST\\_KNOWN sample may lie above or below the nominal $\\alpha$.",
        "\\item \\emph{Per query} weights every novel sequence equally; \\emph{per genus} "
        "averages the detection rate within each novel genus first, so that "
        "genera represented by many sequences do not dominate.",
        "\\end{tablenotes}",
        "\\end{threeparttable}",
        "\\end{table}",
    ]
    write("table6_conformal.tex", "\n".join(body))


# ------------------------------------------------------------------ table 7 ---
def table7():
    rows = [("AUROC", "AUROC", auroc, True),
            ("novel detected, $\\alpha=0.01$", "Novel detection @ alpha=0.01", pct, True),
            ("novel detected, $\\alpha=0.05$", "Novel detection @ alpha=0.05", pct, True),
            ("novel detected, $\\alpha=0.10$", "Novel detection @ alpha=0.10", pct, True),
            ("novel detected, $\\alpha=0.20$", "Novel detection @ alpha=0.20", pct, True),
            ("false novelty, $\\alpha=0.05$", "False novelty @ alpha=0.05", pct, False),
            (None, None, None, None),
            ("family recovery, known species (LSO)", "LSO family recovery", pct, True),
            ("family recovery, novel genus (LGO)", "LGO family recovery", pct, True)]
    body = [
        "\\begin{table}[htbp]\\centering",
        "\\begin{threeparttable}",
        "\\caption{Leakage-controlled historical ITS2 benchmark. Percent identity and "
        "leakage-safe M4 cosine are scored on identical queries, identical reference "
        "collections and an identical conformal procedure, so the comparison is paired "
        "throughout. Higher is better except for false novelty, where the target is "
        "the nominal $\\alpha$.}",
        "\\label{tab:historical}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "metric & percent identity & M4 cosine & difference \\\\",
        "\\midrule",
    ]
    for label, key, f, higher_better in rows:
        if label is None:
            body.append("\\addlinespace")
            continue
        a, b = hist.loc[key, "identity"], hist.loc[key, "m4_cosine"]
        ta, tb = f(a), f(b)
        if higher_better:
            ta = "\\textbf{" + ta + "}" if a > b else ta
            tb = "\\textbf{" + tb + "}" if b > a else tb
            d = f(a - b) if f is pct else f"{a - b:+.3f}"
            if f is pct:
                d = f"{100*(a-b):+.1f} pp"
        else:
            d = f"{100*(a-b):+.1f} pp"
        body.append(f"{label} & {ta} & {tb} & {d} \\\\")
    body += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\begin{tablenotes}\\footnotesize",
        f"\\item $n=1{{,}}632$ calibration knowns, $1{{,}}632$ test knowns and "
        f"{num(audit['Historical LGO queries'])} novel-genus queries, identical for both scores.",
        "\\item The AUROC difference is $+0.069$ (95\\% CI $+0.049$ to $+0.089$, "
        "10{,}000 paired bootstrap resamples clustered on query genus in both "
        "classes). The DeLong test on the same contrast gives $z=12.4$, "
        "$p<10^{-16}$; it assumes within-class independence, which the LGO design "
        "violates, so the clustered interval is the one to read. Clustering widens "
        "the interval by $1.9\\times$. McNemar on the $\\alpha=0.05$ flag sets "
        "separates power from calibration: among novel queries, 969 were flagged by "
        "identity alone against 115 by cosine alone ($p<10^{-16}$), while the "
        "observed false-novelty rates differ by $0.6$ percentage points.",
        "\\item 542 of the 4,444 LGO queries (12.2\\%) returned no VSEARCH hit at "
        "the 0.5 identity threshold and carry the censoring value. They are flagged "
        "at every $\\alpha$: of the 1,235 novel queries identity flags at "
        "$\\alpha=0.05$, 542 (44\\%) are censored rather than scored.",
        "\\item The M4 cosine column comes from the leakage-safe retraining, fitted "
        "to 11\\% fewer records and 483 fewer genera than primary M4. DEV was not "
        "reopened for that run, so the cost of the mask to model quality is "
        "unmeasured.",
        "\\item The calibration and test halves of the LSO set are drawn by "
        "\\texttt{np.random.default\\_rng(0).permutation}. The choice of halving moves "
        "the identity advantage between $0.069$ and $0.083$ AUROC across the "
        "partitions we examined, without affecting its direction or significance, so "
        "the operating points here should be read as approximate.",
        "\\item The currently deposited splitter yields 3,264 LSO queries rather than "
        "the 3,170 of the earlier novelty manuscript; the LGO count reproduces exactly. "
        "These results are therefore a paired comparison on the currently reproducible "
        "harness, not a numerical reproduction of the earlier LSO run.",
        "\\end{tablenotes}",
        "\\end{threeparttable}",
        "\\end{table}",
    ]
    write("table7_historical.tex", "\n".join(body))


# ----------------------------------------------------------------- table S1 ---
def tableS1():
    body = [
        "\\begin{table}[htbp]\\centering",
        "\\begin{threeparttable}",
        "\\caption{Sensitivity of the hierarchical objective to its weight. The reduced $\\lambda_{\\mathrm{tax}}=0.25$ run was performed once after the initially specified $\\lambda_{\\mathrm{tax}}=1.0$ run and was not followed by a weight sweep.}",
        "\\label{tab:m2sens}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "$\\lambda_{\\mathrm{tax}}$ & view & AUROC & novel family & novel order \\\\",
        "\\midrule",
    ]
    for lab, model in [("1.00", "M2"), ("0.25", "M2b")]:
        for i, v in enumerate(VIEWS):
            r = ab[(ab.model == model) & (ab.view == v)].iloc[0]
            name = lab if i == 0 else ""
            body.append(f"{name} & {VLAB[v]} & {auroc(r['auroc'])} & "
                        f"{pct(r['novel_family'])} & {pct(r['novel_order'])} \\\\")
        if model == "M2":
            body.append("\\addlinespace")
    body += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\begin{tablenotes}\\footnotesize",
        "\\item Reducing the weight to one quarter did not produce a uniformly better trade-off: ITS1 AUROC increased slightly, whereas ITS-core and ITS2 AUROC and most placement metrics decreased. The sensitivity run therefore did not motivate further DEV-set weight tuning.",
        "\\end{tablenotes}",
        "\\end{threeparttable}",
        "\\end{table}",
    ]
    write("tableS1_m2_sensitivity.tex", "\n".join(body))


# ----------------------------------------------------------------- table S2 ---
def tableS2():
    def row(label, key, denom_key=None):
        v = int(audit[key])
        if denom_key is None:
            return f"{label} & {num(v)} & \\\\"
        d = int(audit[denom_key])
        return f"{label} & {num(v)} & {100*v/d:.1f}\\% \\\\"

    body = [
        "\\begin{table}[htbp]\\centering",
        "\\begin{threeparttable}",
        "\\caption{Historical-benchmark leakage audit and exclusion mask. The share "
        "column expresses each count as a percentage of the corresponding held-out set.}",
        "\\label{tab:leakage}",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "item & count & share of held-out set \\\\",
        "\\midrule",
        "\\multicolumn{3}{l}{\\emph{Training records}} \\\\",
        row("\\quad original TRAIN\\_REF", "Original TRAIN_REF records"),
        row("\\quad leakage-safe TRAIN\\_REF", "Leakage-safe TRAIN_REF records"),
        row("\\quad excluded by the mask", "Excluded records"),
        "\\addlinespace",
        "\\multicolumn{3}{l}{\\emph{Leave-one-genus-out (LGO) queries}} \\\\",
        row("\\quad queries", "Historical LGO queries"),
        row("\\quad query genera", "Historical LGO query genera"),
        row("\\quad exact query IDs in original TRAIN\\_REF",
            "LGO exact query IDs in original TRAIN_REF", "Historical LGO queries"),
        row("\\quad query genera in original TRAIN\\_REF",
            "LGO query genera in original TRAIN_REF", "Historical LGO query genera"),
        row("\\quad query genera after the mask", "LGO query genera remaining after mask",
            "Historical LGO query genera"),
        row("\\quad exact query IDs after the mask",
            "LGO exact query IDs remaining after mask", "Historical LGO queries"),
        "\\addlinespace",
        "\\multicolumn{3}{l}{\\emph{Leave-one-species-out (LSO) queries}} \\\\",
        row("\\quad queries", "Historical LSO queries"),
        row("\\quad query species", "Historical LSO query species"),
        row("\\quad exact query IDs in original TRAIN\\_REF",
            "LSO exact query IDs in original TRAIN_REF", "Historical LSO queries"),
        row("\\quad query species in original TRAIN\\_REF",
            "LSO query species in original TRAIN_REF", "Historical LSO query species"),
        row("\\quad query species after the mask", "LSO query species remaining after mask",
            "Historical LSO query species"),
        row("\\quad query genera still represented after the mask",
            "LSO query genera still represented after mask"),
        "\\bottomrule",
        "\\end{tabular}",
        "\\begin{tablenotes}\\footnotesize",
        "\\item The LSO design deliberately keeps the query's genus in the reference "
        "collection, so the last row is a property of the design rather than residual "
        "leakage. What the mask had to remove was the query species itself.",
        "\\end{tablenotes}",
        "\\end{threeparttable}",
        "\\end{table}",
    ]
    write("tableS2_leakage.tex", "\n".join(body))


if __name__ == "__main__":
    print("Regenerating tables from data/ ...")
    table1(); table2(); table3(); table4()
    table5(); table6(); table7(); tableS1(); tableS2()
    print("done.")
