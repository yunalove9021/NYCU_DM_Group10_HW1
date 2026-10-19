"""Export the pipeline outline (with tables T1-T15 and figures E1-E6, F1-F14) to results/report_outline.docx.

Numbers are read from the result files, so rerun this after `run_all.py`. Tables are parsed from
results/report_summary.md, whose table numbers are the T-codes used here.
"""
import json
import re

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from common import OUT

FIG_CAPTIONS = {
    "E1": "Target distribution: raw SalePrice and log1p(SalePrice), train vs. test.",
    "E2": "Top 25 features by |Pearson r| with log price (train), coloured by feature type.",
    "E3": "Six strongest non-dummy features vs. log price with linear fit.",
    "E4": "Correlation among the 15 strongest features (collinear blocks).",
    "E5": "Train → test drift: largest standardised mean differences.",
    "E6": "Log price by neighbourhood (train).",
    "F1": "Best time-series CV RMSE at each correlation threshold, per model family.",
    "F2": "Number of selected features vs. threshold.",
    "F3": "Effect of one-hot handling and outlier removal.",
    "F4": "CV RMSE along the regularisation range (dotted = chosen α).",
    "F5": "Best configuration per model family (error bar = fold SD).",
    "F6": "Top 30 standardised coefficients with bootstrap 95% CI, p-value stars and VIF.",
    "F7": "Residual diagnostics of the final model on the test set.",
    "F8": "Feature ablation: CV MAE per set and test MAE by house age.",
    "F9": "Top-k sets: CV MAE vs. k.",
    "F10": "Pairwise test MAE differences between feature sets (95% paired-bootstrap CI).",
    "F11": "Test residuals; houses outside ±2 SD circled, five largest numbered.",
    "F12": "Sale-condition mix and feature differences of flagged houses.",
    "F13": "Five largest errors on key features (hollow = predicted, filled = actual).",
    "F14": "Percentile profile of the five largest errors among training houses.",
    "F15": "Pipeline: from raw data to a prediction, with each stage's input and output.",
    "F16": "Main results: CV MAE (± fold SD) and test MAE of every baseline and our model, with the decision threshold.",
}


def fig_path(code):
    return next((OUT / "figures").glob(f"{code}_*.png"))


def parse_tables(md_text):
    """'*Table N: caption*' below a markdown table -> {T{N}: (caption, header, rows)}."""
    out, lines = {}, md_text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"\*Table (\d+): (.*)\*$", line.strip())
        if not m:
            continue
        j = i - 1
        while j >= 0 and not lines[j].startswith("|"):
            j -= 1
        block = []
        while j >= 0 and lines[j].startswith("|"):
            block.insert(0, lines[j])
            j -= 1
        cells = [[c.strip() for c in row.strip("|").split("|")] for row in block]
        out[f"T{m.group(1)}"] = (m.group(2), cells[0], cells[2:])
    return out


# ------------------------------------------------------------------ docx helpers
def add_runs(p, text, size=None, italic=False):
    """Add text with **bold** segments."""
    for k, part in enumerate(re.split(r"(\*\*[^*]+\*\*)", text)):
        if not part:
            continue
        bold = part.startswith("**") and part.endswith("**")
        r = p.add_run(part[2:-2] if bold else part)
        r.bold, r.italic = bold or None, italic or None
        if size:
            r.font.size = Pt(size)
    return p


def add_hyperlink(p, url, text=None):
    rel = p.part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    h = OxmlElement("w:hyperlink")
    h.set(qn("r:id"), rel)
    r = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    for tag, val in (("w:color", "2A78D6"), ("w:u", "single")):
        e = OxmlElement(tag)
        e.set(qn("w:val"), val)
        rpr.append(e)
    r.append(rpr)
    t = OxmlElement("w:t")
    t.text = text or url
    t.set(qn("xml:space"), "preserve")
    r.append(t)
    h.append(r)
    p._p.append(h)


def set_border(cell, **sides):
    tcpr = cell._tc.get_or_add_tcPr()
    b = tcpr.find(qn("w:tcBorders"))
    if b is None:
        b = OxmlElement("w:tcBorders")
        tcpr.append(b)
    for side, sz in sides.items():
        e = OxmlElement(f"w:{side}")
        e.set(qn("w:val"), "single" if sz else "nil")
        if sz:
            e.set(qn("w:sz"), str(sz))
            e.set(qn("w:color"), "000000")
        b.append(e)


class Builder:
    def __init__(self, tables):
        self.doc = Document()
        self.tables = tables
        sec = self.doc.sections[0]
        sec.left_margin = sec.right_margin = Inches(0.8)
        sec.top_margin = sec.bottom_margin = Inches(0.8)
        st = self.doc.styles["Normal"]
        st.font.name = "Calibri"
        st.font.size = Pt(10.5)
        self.width = sec.page_width - sec.left_margin - sec.right_margin

    def h(self, text, level):
        self.doc.add_heading(text, level=level)

    def b(self, text, level=1):
        style = "List Bullet" if level == 1 else f"List Bullet {level}"
        p = self.doc.add_paragraph(style=style)
        add_runs(p, text)
        p.paragraph_format.space_after = Pt(2)

    def ref(self, url, level=1):
        style = "List Bullet" if level == 1 else f"List Bullet {level}"
        p = self.doc.add_paragraph(style=style)
        r = p.add_run("Ref: ")
        r.italic = True
        add_hyperlink(p, url)
        p.paragraph_format.space_after = Pt(2)

    def table(self, code):
        cap, header, rows = self.tables[code]
        ncol = len(header)
        t = self.doc.add_table(rows=1 + len(rows), cols=ncol)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        # column width proportional to the longest entry (header words may wrap), with a floor
        need = [max([len(w) for w in header[c].split()] + [len(r[c].replace("**", "")) for r in rows if c < len(r)]) for c in range(ncol)]
        need = [min(max(n + 2, 5), 32) for n in need]
        widths = [int(self.width * n / sum(need)) for n in need]
        size = 7.5 if ncol >= 8 else 8.5
        numeric = [c > 0 and all(re.match(r"^\**[\s+\-−<\[0-9]", r[c] or "0") or r[c] in ("—", "inf", "") for r in rows) for c in range(ncol)]
        for i, row in enumerate([header] + rows):
            for c in range(ncol):
                cell = t.cell(i, c)
                cell.width = widths[c]
                p = cell.paragraphs[0]
                add_runs(p, row[c] if c < len(row) else "", size=size)
                if i == 0:
                    for r in p.runs:
                        r.bold = True
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if numeric[c] else WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.space_after = Pt(0)
                set_border(cell, left=0, right=0,
                           top=12 if i == 0 else 0,
                           bottom=6 if i == 0 else (12 if i == len(rows) else 0))
        p = self.doc.add_paragraph()
        add_runs(p, f"Table {code[1:]}: {cap}", size=9)
        p.paragraph_format.space_after = Pt(10)

    def figure(self, code):
        self.doc.add_picture(str(fig_path(code)), width=self.width)
        self.doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        p = self.doc.add_paragraph()
        add_runs(p, f"Figure {code}: {FIG_CAPTIONS[code]}", size=9)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(10)

    def attach(self, *codes):
        for c in codes:
            (self.table if c.startswith("T") else self.figure)(c)


def main():
    J = lambda n: json.loads((OUT / f"{n}.json").read_text(encoding="utf-8"))
    ed, ts, best, tm = J("eda_summary"), J("tuning_summary"), J("best_config"), J("test_metrics")
    cs, er, ab_s, ms, codes = J("coef_summary"), J("error_summary"), J("ablation_summary"), J("main_summary"), J("table_codes")
    R = lambda n: pd.read_csv(OUT / n)
    bf, corr, drift, coef, gv = R("tuning_best_per_family.csv").set_index("model"), R("eda_correlations.csv"), R("eda_drift.csv"), R("coefficients.csv"), R("gvif.csv")
    ab, pw, ep = R("ablation.csv").set_index("set"), R("ablation_pairwise.csv").set_index("comparison"), R("error_points.csv")
    ec, res, mr, mf, sl = R("error_compare.csv").set_index("feature"), R("tuning_results.csv"), R("main_results.csv"), R("main_results_folds.csv"), R("ablation_slices.csv")
    tables = parse_tables((OUT / "report_summary.md").read_text(encoding="utf-8"))
    C = lambda label: codes[label]
    import report as rp  # short labels for causes / verdicts; p-value wording

    sig = coef[coef["p"] < 0.05]
    prac = sig[sig["practical"] == "yes"]
    snp = sig[sig["practical"] == "negligible"]["feature"].tolist()
    hyp = coef[coef["expected"].notna()]
    rev = coef[coef["hypothesis_check"].str.startswith("reversal")]
    eff = ed["effect_counts"]
    n_eng10 = int((corr.head(10)["kind"] == "engineered").sum())
    ys = drift.set_index("feature").at["YrSold", "SMD"]
    k = lambda v: f"${v / 1e3:,.1f}k"
    sd = lambda v: f"{'+' if v >= 0 else '−'}${abs(v):,.0f}"
    pAB, pBF = pw.loc["A vs B"], pw.loc["B vs F"]
    verd = ep["comps_verdict"].map(rp.short_verdict).value_counts()
    drop_tab = res.groupby(["drop_outliers", "model"])["cv_rmse"].min().unstack()
    drop_helps = drop_tab.loc[True] < drop_tab.loc[False]
    drop_txt = ("this lowered CV error for every model" if drop_helps.all() else
                "this raised CV error for every model, so the best configuration keeps them" if not drop_helps.any() else
                f"this helped {', '.join(drop_helps[drop_helps].index)} only")
    thr_groups = {}
    for m in bf.index:
        thr_groups.setdefault(f"{bf.at[m, 'threshold']:g}", []).append(m)
    thr_txt = ("; ".join(f"{t} for {', '.join(v)}" for t, v in thr_groups.items()) if len(thr_groups) > 1
               else f"{next(iter(thr_groups))} for all five model families")
    sig_word = lambda r: "significant" if min(r["p_test_bootstrap_holm"], r["p_test_wilcoxon_holm"]) < 0.05 else "not significant"
    e_sig = all(min(pw.loc[c, "p_test_bootstrap_holm"], pw.loc[c, "p_test_wilcoxon_holm"]) < 0.05 for c in pw.index if "E" in c.split(" vs "))
    sl["gain"] = sl["MAE_usd A"] - sl["MAE_usd B"]
    best_sl, worse_sl = sl.loc[sl["gain"].idxmax()], sl[sl["gain"] < 0]["slice"].tolist()
    same_ac = abs(ab.at["A", "test_mae_usd"] - ab.at["C", "test_mae_usd"]) < 1
    same_bd = abs(ab.at["B", "test_mae_usd"] - ab.at["D", "test_mae_usd"]) < 1
    dollars = lambda r: f"{'+' if r['effect_usd'] >= 0 else '−'}${abs(r['effect_usd']) / 1e3:.1f}k"
    mz = coef.set_index("feature").loc["MSZoning_C"] if "MSZoning_C" in set(sig["feature"]) else None
    gv_over = gv[gv["GVIF^(1/(2df))"] > 10 ** 0.5]
    thr = ms["threshold_usd"]
    ours, rival = mr.iloc[-1], mr.set_index("method").loc[ms["rival"]]
    meets = mr[mr["test_mae"] <= thr]["method"].tolist()
    fold_gap = (mf[mf["method"] == ms["rival"]].sort_values("fold")["mae"].to_numpy()
                - mf[mf["method"] == ms["ours"]].sort_values("fold")["mae"].to_numpy())
    claim = ("a clear win" if ms["cv_gap"] > 0 and ms["test_gap"] > 0 and ms["p_cv"] < 0.05 and not (ms["test_ci"][0] <= 0 <= ms["test_ci"][1]) else
             "a consistent but small gain" if ms["cv_gap"] > 0 and ms["test_gap"] > 0 else "performance on par with the strong baseline, not a win")
    st, l1 = best["setting"], best["key"][2]
    curve = res[(res["model"] == best["family"]) & (res["threshold"] == st["threshold"]) & (res["cat_mode"] == st["cat_mode"]) &
                (res["drop_outliers"] == st["drop_outliers"])]
    curve = curve[np.isclose(curve["extra"], l1)] if l1 is not None else curve
    ib = curve["cv_rmse"].idxmin()
    flat = curve[curve["cv_rmse"] <= curve.at[ib, "cv_rmse"] + curve.at[ib, "cv_rmse_sd"] / np.sqrt(5)]["alpha"]
    n_vif10 = int((coef.loc[~coef["is_reference"], "VIF"] >= 10).sum())
    hyp_rows = tables[C("hypotheses")][2]
    check_rows = tables[C("hypothesis_check")][2]

    B = Builder(tables)
    B.doc.add_heading("Linear Regression on Ames House Prices: Pipeline and Findings", 0)
    p = B.doc.add_paragraph()
    add_runs(p, "Codes: E = EDA figures, F = model figures, T = tables (attached under each step; same numbers as report.md). "
                "Generated by model/export_docx.py from model/results/.", size=9, italic=True)

    # ---------------- 1 data
    B.h("1. Data and setup", 1)
    B.b(f"Chronological split from the team's preprocessing: train = {ed['n_train']:,} sales ({tm['train_period']}), test = {ed['n_test']} sales ({tm['test_period']}) ({C('split')}).")
    B.b("Target is log1p(SalePrice); dollar metrics use expm1 of the prediction.")
    fc = {r[0]: r[1] for r in tables[C("features")][2]}
    B.b(f"Features: {fc['Total']} in total = {fc['Original (non-dummy)']} original non-dummy + {fc['One-hot dummies']} one-hot + {fc['Engineered']} engineered ({C('features')}).")
    B.b("Log transform decided per fit: the team's rule (log1p if skewness > 0.75) is re-applied to raw values on each fold's training rows only, then applied unchanged to its validation rows (and to the test set for the final model).")
    B.b("The team's preprocessing picks this list once on the whole training set (mapping/skewed_features.json); re-deciding it per fold keeps the choice from seeing the CV validation folds.")
    B.b("Outlier removal, |r| feature ranking and the scaler are also fitted on each training fold only and applied unchanged to its validation fold.")
    B.ref("https://doi.org/10.1145/2382577.2382579", 2)
    B.attach(C("split"), C("features"), C("folds"))

    # ---------------- 2 EDA (§4)
    B.h("2. Exploratory feature analysis (§4)", 1)
    B.b("**Evidence (training set only)**", 1)
    B.b(f"Raw price is right-skewed (skew {ed['skew_raw']:.2f}); log1p reduces it to {ed['skew_log']:.2f}, so the model predicts log price (E1).", 2)
    B.b(f"By Cohen's bands, {eff.get('large', 0)} features have large |r| (≥ 0.5), {eff.get('medium', 0)} medium, {eff.get('small', 0)} small and {eff.get('negligible', 0)} negligible (E2, {C('eda_corr')}).", 2)
    B.ref("https://www.taylorfrancis.com/books/mono/10.4324/9780203771587/statistical-power-analysis-behavioral-sciences-jacob-cohen", 3)
    B.b("Strongest: " + ", ".join(f"{r.feature} (r = {r.r:.2f})" for r in corr.head(3).itertuples()) + f"; {n_eng10} of the top 10 are engineered (E2).", 2)
    B.b("Strong features relate roughly linearly to log price (E3); area, quality, garage and age features form correlated blocks (E4).", 2)
    B.b(f"{ed['n_drift']} of {ed['n_nondummy']} non-dummy features drift by |SMD| > 0.1 from train to test; apart from YrSold ({ys:+.2f}, by construction) shifts are small (E5, {C('eda_drift')}).", 2)
    B.b("Neighbourhood separates price levels strongly (E6).", 2)
    B.b("**Keep / transform / drop**", 1)
    B.b("Keep strong and medium features and the location dummies (E2, E6).", 2)
    B.b("Transform the target and skewed areas to log scale (E1, E3), calendar years to ages, and month to sin/cos.", 2)
    B.b("Combine correlated parts into aggregates and add a quality × size interaction (E4, E2).", 2)
    B.b("Drop weak features with a CV-tuned correlation threshold (§5.2); handle redundancy with the penalty instead of deleting features.", 2)
    B.b(f"**Hypotheses** ({C('hypotheses')}), tested in §7.2 and §8.1:", 1)
    for r in hyp_rows:
        B.b(f"{r[0]}: {r[1]} — {r[2]}.", 2)
    B.attach(C("eda_corr"), C("eda_drift"), C("hypotheses"), "E1", "E2", "E3", "E4", "E5", "E6")

    # ---------------- 3 pipeline (§5.1)
    B.h("3. Pipeline and feature engineering (§5.1)", 1)
    B.b("Raw data → preprocessing → feature engineering → feature selection → linear model → prediction & evaluation; every data-dependent step is fitted on training rows only (F15).")
    for r in tables[C("feature_groups")][2]:
        B.b(f"**{r[0]}** ({r[1]}): {r[2]} [{r[3]}].", 1)
    B.b("Whether the groups help as a whole is tested in the ablation (§7.2, Set B vs A).")
    B.attach("F15", C("feature_groups"))

    # ---------------- 4 model & tuning (§5.2)
    B.h("4. Model choice and hyper-parameter tuning (§5.2)", 1)
    B.b("**Model choice**", 1)
    B.b(f"{n_vif10} features have VIF ≥ 10 and there are {fc['Total']} candidate features for {ed['n_train']:,} sales, mostly weak one-hot dummies, so we use {best['family']}: it shrinks correlated coefficients together and drops useless ones.", 2)
    B.ref("https://doi.org/10.1111/j.1467-9868.2005.00503.x", 3)
    B.b(f"Confirmed by CV: every penalised model beats OLS (CV RMSE {bf.at['OLS', 'cv_rmse']:.4f} vs. {best['cv_rmse']:.4f}); Ridge, Lasso and Huber are compared too.", 2)
    B.b("**Feature selection**", 1)
    B.b("Correlation coefficient (r) threshold: [0, 0.1, 0.3, 0.5], where 0 keeps all features; ranking recomputed on each training fold.", 2)
    B.b(f"Best threshold: {thr_txt} (F1).", 3)
    B.ref("https://www.taylorfrancis.com/books/mono/10.4324/9780203771587/statistical-power-analysis-behavioral-sciences-jacob-cohen", 3)
    B.b("Cohen considers r = 0.1 a small effect, 0.3 medium and 0.5 large.", 3)
    B.b("Collinearity threshold: |r| > 0.7, applied in preprocessing (greedy selection), not re-tuned.", 2)
    B.ref("https://nsojournals.onlinelibrary.wiley.com/doi/10.1111/j.1600-0587.2012.07348.x", 3)
    B.b(f"Outliers: dropping houses with GrLivArea > 4000 from training folds only was tried; {drop_txt} (F3).", 2)
    B.ref("https://doi.org/10.1080/10691898.2011.11889627", 3)
    B.b("**Hyper-parameter ranges**", 1)
    B.b("Lasso / ElasticNet α: 100 log-spaced values from α_max (all coefficients zero) down to 1e-4·α_max (F4).", 2)
    B.ref("https://www.jstatsoft.org/article/view/v033i01", 3)
    B.b(f"Ridge α: 30 values spanning effective degrees of freedom from about the rank down to 1 (α = {ts['ridge_range'][0]:.2g} … {ts['ridge_range'][1]:.2g}).", 2)
    B.ref("https://doi.org/10.1007/978-0-387-84858-7", 3)
    B.b("ElasticNet l1_ratio: [0.01, 0.05, 0.1, 0.5, 0.7, 0.9, 0.95, 0.99]; 0.01 and 0.05 were added because the optimum sat on the lower edge.", 2)
    B.b("Huber ε: [1.345, 2.0]; 1.345 gives 95% efficiency under normal errors.", 2)
    B.ref("https://doi.org/10.1002/0471725250", 3)
    B.b("All grids use log spacing.", 2)
    B.ref("https://www.jmlr.org/papers/v13/bergstra12a.html", 3)
    B.b("**Time-series CV: k = 5 (training period only)**", 1)
    B.b(f"Selected: {best['family']} ({best['params']}), threshold {st['threshold']:g}, dummies = {st['cat_mode']}, outliers {'dropped' if st['drop_outliers'] else 'kept'}; CV RMSE {best['cv_rmse']:.4f} ± {best['cv_rmse_sd']:.4f} ({C('top10')}, F5).", 2)
    B.b(f"Tuning curve ({C('tuning_curve')}): CV RMSE stays within one standard error of the optimum for α ≈ {flat.min():.3g}–{flat.max():.3g}; larger α over-shrinks (F4).", 2)
    B.ref("https://www.sciencedirect.com/science/article/abs/pii/S0020025511006773", 2)
    B.attach(C("tuning_curve"), C("settings"), C("top10"), "F1", "F2", "F3", "F4", "F5")

    # ---------------- 5 metrics & baselines (§6)
    B.h("5. Metrics and baselines (§6)", 1)
    B.b("**Metrics**", 1)
    B.b("MAE in dollars is primary: a buyer, seller or lender loses money in proportion to the dollar error.", 2)
    B.b("RMSE on log price is the training objective and penalises large relative misses; MAPE, R² and the share within ±10% support it.", 2)
    B.b(f"**Decision threshold: MAE ≤ ${thr:,.0f}** (10% of the median training price, ${ms['median_price']:,.0f}): below it an estimate is close enough to set a listing price or an offer; above it, the decision would change. This band is our assumption about the user.", 1)
    B.b("**Baselines** (same folds and test set as our model; tiers from the team's baseline2.py)", 1)
    for r in tables[C("baselines")][2]:
        B.b(f"{r[0]}: {r[1]} — {r[2]}" + (f"; {r[3]}" if r[3] != "—" else "") + ".", 2)
    B.attach(C("metrics"), C("baselines"))

    # ---------------- 6 main results (§7.1)
    B.h("6. Main results (§7.1)", 1)
    B.b(f"On the test set {', '.join(meets) or 'no method'} {'meets' if len(meets) == 1 else 'meet'} the ${thr / 1e3:.1f}k threshold; trivial and single-feature baselines miss it by a wide margin ({C('main_results')}, F16).")
    B.b(f"Ours: CV MAE {k(ours['cv_mae'])}, test MAE {k(ours['test_mae'])}, test R² {ours['test_r2']:.3f}; strongest baseline ({ms['rival']}): CV {k(rival['cv_mae'])}, test {k(rival['test_mae'])}.")
    B.b(f"Ours vs. {ms['rival']}: {'better' if ms['cv_gap'] > 0 else 'worse'} in CV by ${abs(ms['cv_gap']):,.0f} (corrected t-test over folds, {rp.pv(ms['p_cv'])}); test MAE {'lower' if ms['test_gap'] > 0 else 'higher'} by ${abs(ms['test_gap']):,.0f} (paired bootstrap 95% CI [{sd(ms['test_ci'][0])}, {sd(ms['test_ci'][1])}], Wilcoxon {rp.pv(ms['p_wilcoxon'])}) → {claim}.")
    B.ref("https://doi.org/10.1023/A:1024068626366", 2)
    B.b(f"The linear model's CV advantage is largest with the least training data (gap per fold: {', '.join(k(g) for g in fold_gap)}) and fades with all {ed['n_train']:,} sales: the engineered features give it the shape of the relationships, while the trees must learn every curve from data.")
    B.b(f"Diagnostics ({C('test_details')}, F7): mean residual {tm['mean_resid']:+.4f} ({rp.pv(tm['mean_resid_p'])}, no level shift); Breusch-Pagan {rp.pv(tm['bp_p'])} (variance grows with price, so bootstrap intervals); Durbin-Watson {tm['dw']:.2f}.")
    B.attach(C("main_results"), "F16", C("test_details"), "F7")

    # ---------------- 7 ablation (§7.2)
    B.h("7. Ablation (§7.2)", 1)
    B.b("**Setting**", 1)
    B.b("Same model (ElasticNet), split, folds, tuning procedure and outlier rule for every set; only the features change.", 2)
    for code, text in [("A", "original features only + select all features"), ("B", "original & engineered features + select all features"),
                       ("C", "original features only + simple top-k (|r| ranked inside each training fold; k tuned by CV)"),
                       ("D", "original & engineered features + simple top-k"), ("E", "original features only + greedy feature selection"),
                       ("F", "original & engineered features + greedy feature selection")]:
        B.b(f"Set {code}: {text}", 2)
    B.b(f"Significance: corrected t-test on CV folds; paired bootstrap and Wilcoxon on test errors; Holm correction over the 15 pairs ({C('ablation_pairwise')}, F10).", 2)
    B.ref("https://doi.org/10.1023/A:1024068626366", 3)
    B.ref("https://doi.org/10.2307/3001968", 3)
    B.ref("https://www.jstor.org/stable/4615733", 3)
    B.b("**Findings**", 1)
    B.b(f"Engineered features (H6): B vs A, test MAE {sd(pAB['test_diff_usd'])} per house in favour of {'B' if pAB['test_diff_usd'] > 0 else 'A'} [{sd(pAB['test_ci_lo_usd'])}, {sd(pAB['test_ci_hi_usd'])}], {sig_word(pAB)} after Holm ({C('ablation')}, F8).", 2)
    B.b("Top-k adds nothing: the tuned k keeps nearly every feature, so C = A and D = B; ElasticNet already zeroes what a correlation filter would drop (F9)." if same_ac and same_bd else "Top-k vs. all features: see the ablation table and F9.", 2)
    B.b(f"Greedy selection needs engineered features: E is the worst set ({k(ab.at['E', 'test_mae_usd'])} test MAE)" + (" and significantly worse than every other set on the test set." if e_sig else "."), 2)
    B.b(f"F vs B: {int(ab.at['F', 'n_feats'])} vs. {int(ab.at['B', 'n_feats'])} features, test MAE difference {sd(-pBF['test_diff_usd'])} [{sd(-pBF['test_ci_hi_usd'])}, {sd(-pBF['test_ci_lo_usd'])}] (F − B), {sig_word(pBF)}" + (": the greedy set matches B with a third of the features." if sig_word(pBF) == "not significant" else "."), 2)
    B.b("E and F were selected on the whole training set, so their CV scores are optimistic; compare them on the test set.", 2)
    B.ref("https://doi.org/10.1073/pnas.102102699", 3)
    B.b(f"By slice (B vs A): engineered features help most for {best_sl['slice']} ({sd(best_sl['gain'])} MAE)" + (f" and hurt for {', '.join(worse_sl)}" if worse_sl else "") + f" ({C('ablation_slices')}).", 2)
    B.attach(C("ablation"), C("ablation_pairwise"), C("ablation_slices"), "F8", "F9", "F10")

    # ---------------- 8 coefficients (§8.1)
    B.h("8. Coefficient and statistical significance (§8.1)", 1)
    B.b("**Setting**", 1)
    B.b("Coefficients are standardised: β = change in log price per +1 SD of the feature.", 2)
    B.b("95% CI and p-value from 1,000 bootstrap resamples, with the log choice, feature selection and fit repeated in each resample (F6).", 2)
    B.ref("https://doi.org/10.1201/9780429246593", 3)
    B.b("Selection frequency = share of resamples with a non-zero coefficient (stability).", 2)
    B.ref("https://doi.org/10.1111/j.1467-9868.2010.00740.x", 3)
    B.b("VIF on a reference-coded design (one level per categorical dropped); categoricals judged as a group by GVIF^(1/(2df)) against √10 ≈ 3.16.", 2)
    B.ref("https://doi.org/10.1080/01621459.1992.10475190", 3)
    B.ref("https://doi.org/10.1007/s11135-006-9018-6", 3)
    B.b(f"Practical significance: an effect below {cs['practical']:.0%} of the median price (${cs['median_price'] * cs['practical']:,.0f}) is negligible.", 2)
    B.ref("https://doi.org/10.1080/00031305.2016.1154108", 3)
    B.b("**Findings**", 1)
    B.b(f"{len(sig)} of {len(coef)} features are significant (p < 0.05); {len(prac)} are also practically relevant ({C('coefficients')}).", 2)
    B.b("Strongest effects per +1 SD: " + ", ".join(f"{r['feature']} {dollars(r)}" for _, r in prac.head(4).iterrows()) + f" ({C('coefficients')}, F6).", 2)
    if mz is not None:
        B.b(f"Commercial zoning (MSZoning_C) lowers price by {abs(mz['effect_pct']):.0f}% ({dollars(mz)}).", 2)
    B.b(f"Significant but practically negligible: {', '.join(snp) or 'none'}.", 2)
    B.b(f"No sign reversals: all {int((hyp['hypothesis_check'] == 'agrees').sum())} features with a significant effect have the expected sign." if not len(rev)
        else f"{len(rev)} sign reversal(s): " + ", ".join(rev["feature"]) + ".", 2)
    B.b(f"{n_vif10} features have VIF ≥ 10, so the area, quality and age blocks must be read as groups.", 2)
    if len(gv_over):
        B.b("Above the GVIF threshold: " + ", ".join(f"{r.categorical} ({r._4:.2f})" for r in gv_over.itertuples()) + f" ({C('gvif')}).", 2)
    B.b(f"**Hypotheses revisited** ({C('hypothesis_check')})", 1)
    for r in check_rows:
        B.b(f"{r[0]}: {r[2]} — {r[1]}.", 2)
    B.attach(C("coefficients"), C("hypothesis_check"), C("gvif"), "F6")

    # ---------------- 9 error analysis (§8.2)
    B.h("9. Error analysis (§8.2)", 1)
    B.b("**Setting**", 1)
    B.b(f"A house is flagged if its residual (actual − predicted, log scale) is outside ±2 SD ≈ −{er['usd_band'][1]:.0f}% / +{er['usd_band'][0]:.0f}% of the price (F11).", 2)
    B.b("Each flagged house is compared with the median price of its 10 most similar training houses in the same neighbourhood.", 2)
    B.ref("https://doi.org/10.1108/14635780310483656", 3)
    B.b("A rule-based cause is added: beyond the training range, non-Normal sale, or poor condition.", 2)
    B.b(f"The {er['n_cases']} largest errors are analysed one by one ({C('error_cases')}, F13, F14).", 2)
    B.b("**Findings**", 1)
    B.b(f"{er['n_flag']} of {er['n_test']} test houses are flagged (≈ {er['expected_flag']:.0f} expected); {er['n_over']} are over-predicted.", 2)
    B.b("Comparison with similar houses: " + ", ".join(f"{v} {kk}" for kk, v in verd.items()) + ".", 2)
    B.b(f"Abnormal or partial sales make up {er['nonnormal_flag']:.0%} of flagged houses vs. {er['nonnormal_all']:.0%} of all test houses (F12).", 2)
    B.b(f"Flagged houses are older (SMD {ec.at['YearBuilt', 'SMD']:+.2f}), in worse condition ({ec.at['OverallCond', 'SMD']:+.2f}), lower quality ({ec.at['OverallQual', 'SMD']:+.2f}) and cheaper ({ec.at['Actual', 'SMD']:+.2f}) ({C('error_compare')}).", 2)
    for _, c in ep.head(3).iterrows():
        B.b(f"{c['#']} (Id {c['Id']}, {c['SaleCondition']}): sold {k(c['Actual'])} vs. {k(c['Pred'])} predicted; similar houses sold for {k(c['comps_median'])}, verdict: {rp.short_verdict(c['comps_verdict'])}; rule-based cause: {rp.short_cause(c['Likely cause'])}.", 2)
    B.b("First fix: model non-Normal sales separately, or add sale-condition × size/quality interactions.", 2)
    B.attach(C("error_cases"), C("error_compare"), "F11", "F12", "F13", "F14")

    path = OUT / "report_outline.docx"
    B.doc.save(path)
    print(f"docx -> {path}")


if __name__ == "__main__":
    main()
