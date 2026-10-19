"""Step 5: assemble results/report.md (full) and results/report_summary.md (key sections) from the saved outputs.

Tables follow ACL-paper conventions: numbered caption below the table ("Table N: ..."), units in the header,
best values in bold, no vertical rules. Every table is also written as a booktabs LaTeX file to results/tables/.
"""
import re

import numpy as np
import pandas as pd

from common import (ALPHA_MIN_RATIO, CAT_MODES, DROP_OUTLIERS, HUBER_EPS, L1_RATIOS, N_ALPHAS, N_BOOT, N_HUBER,
                    N_RIDGE, N_SPLITS, OUT, THRESHOLDS, TOPK, load_data, md, read_json, write_json)

REFERENCES = """- Ambroise, C., & McLachlan, G. J. (2002). Selection bias in gene extraction on the basis of microarray gene-expression data. *PNAS*, 99(10), 6562-6566.
- Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*, 191, 192-213.
- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13, 281-305.
- Breusch, T. S., & Pagan, A. R. (1979). A simple test for heteroscedasticity and random coefficient variation. *Econometrica*, 47(5), 1287-1294.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Erlbaum.
- De Cock, D. (2011). Ames, Iowa: Alternative to the Boston housing data as an end of semester regression project. *Journal of Statistics Education*, 19(3).
- Durbin, J., & Watson, G. S. (1950). Testing for serial correlation in least squares regression I. *Biometrika*, 37(3/4), 409-428.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Fox, J., & Monette, G. (1992). Generalized collinearity diagnostics. *Journal of the American Statistical Association*, 87(417), 178-183.
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software*, 33(1), 1-22.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55-67.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65-70.
- Huber, P. J. (1981). *Robust Statistics*. Wiley.
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
- Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: formulation, detection, and avoidance. *ACM TKDD*, 6(4), 15.
- Meinshausen, N., & Bühlmann, P. (2010). Stability selection. *Journal of the Royal Statistical Society B*, 72(4), 417-473.
- Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning*, 52(3), 239-281.
- O'Brien, R. M. (2007). A caution regarding rules of thumb for variance inflation factors. *Quality & Quantity*, 41, 673-690.
- Pagourtzi, E., Assimakopoulos, V., Hatzichristos, T., & French, N. (2003). Real estate appraisal: a review of valuation methods. *Journal of Property Investment & Finance*, 21(4), 383-401.
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society B*, 58(1), 267-288.
- Wasserstein, R. L., & Lazar, N. A. (2016). The ASA statement on p-values: context, process, and purpose. *The American Statistician*, 70(2), 129-133.
- Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin*, 1(6), 80-83.
- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society B*, 67(2), 301-320."""

TEX_MAP = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
           "~": r"\textasciitilde{}", "^": r"\^{}", "±": r"$\pm$", "−": r"$-$", "→": r"$\rightarrow$", "≥": r"$\geq$",
           "≤": r"$\leq$", "×": r"$\times$", "β": r"$\beta$", "α": r"$\alpha$", "ε": r"$\epsilon$", "Δ": r"$\Delta$", "|": r"$|$"}


def tex_escape(s):
    return "".join(TEX_MAP.get(ch, ch) for ch in str(s))


def fp(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def pv(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def fk(v):  # dollars in thousands
    return f"{v / 1e3:,.1f}"


TABLE_ORDER = ["split", "features", "folds",                       # 1 data
               "eda_corr", "eda_drift", "hypotheses",              # 2 EDA (§4)
               "feature_groups",                                    # 3 pipeline (§5.1)
               "tuning_curve", "settings", "top10",                 # 4 model & tuning (§5.2)
               "metrics", "baselines",                              # 5 metrics & baselines (§6)
               "main_results", "test_details",                      # 6 main results (§7.1)
               "ablation", "ablation_pairwise", "ablation_slices",  # 7 ablation (§7.2)
               "coefficients", "hypothesis_check", "gvif",          # 8 coefficients (§8.1)
               "error_cases", "error_compare"]                      # 9 error analysis (§8.2)


class Tables:
    """ACL-style tables: numbered caption below, bold best values; also saved as booktabs .tex.

    Main tables get codes T1, T2, ... from TABLE_ORDER (so text can refer to a table before it appears);
    any other label is supplementary (S1, S2, ...). Codes go to results/table_codes.json for the docx export."""

    def __init__(self):
        self.codes = {lab: f"T{i + 1}" for i, lab in enumerate(TABLE_ORDER)}
        self.s = 0
        (OUT / "tables").mkdir(parents=True, exist_ok=True)
        for f in (OUT / "tables").glob("*.tex"):
            f.unlink()

    def __call__(self, df, caption, label, bold=None):
        """df: display-ready strings. bold: list of (row, col) cells to bold."""
        if label not in self.codes:
            self.s += 1
            self.codes[label] = f"S{self.s}"
        code = self.codes[label]
        bold = set(bold or [])
        cols = list(df.columns)
        numeric = [c != cols[0] and all(str(v).strip()[:1] in "0123456789+-−<[" or str(v).strip() in ("—", "inf", "") for v in df[c]) for c in cols]
        cell = lambda i, c: f"**{df.iat[i, c]}**" if (i, c) in bold else str(df.iat[i, c])
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---:" if n else ":---" for n in numeric) + "|"]
        lines += ["| " + " | ".join(cell(i, c).replace("|", "/") for c in range(len(cols))) + " |" for i in range(len(df))]
        md_txt = "\n".join(lines) + f"\n\n*Table {code[1:] if code[0] == 'T' else code}: {caption}*\n"
        tcell = lambda i, c: (r"\textbf{" + tex_escape(df.iat[i, c]) + "}") if (i, c) in bold else tex_escape(df.iat[i, c])
        tex = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{" + "".join("r" if n else "l" for n in numeric) + "}",
               r"\toprule", " & ".join(tex_escape(c) for c in cols) + r" \\", r"\midrule"]
        tex += [" & ".join(tcell(i, c) for c in range(len(cols))) + r" \\" for i in range(len(df))]
        tex += [r"\bottomrule", r"\end{tabular}", r"\caption{" + tex_escape(caption) + "}", r"\label{tab:" + label + "}", r"\end{table}"]
        (OUT / "tables" / f"{code}_{label}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
        return md_txt

    def __getitem__(self, label):
        return self.codes[label]


def short_cause(text):
    out = []
    for c in str(text).split("; "):
        if c.startswith("extrapolation: price"):
            out.append("price outside train range")
        elif c.startswith("extrapolation:"):
            out.append(c.split(":")[1].split("=")[0].strip() + " beyond train range")
        elif "abnormal sale" in c:
            out.append("abnormal sale")
        elif "relatives" in c:
            out.append("family sale")
        elif "before completion" in c:
            out.append("unfinished new home")
        elif "allocation" in c:
            out.append("allocation sale")
        elif "poor condition" in c:
            out.append("poor condition")
        elif "commercial" in c:
            out.append("commercial zone")
        elif "adjoining" in c:
            out.append("adjoining land")
        else:
            out.append("none found")
    return ", ".join(dict.fromkeys(out))


VERDICT_SHORT = {"house sold far": "missing information", "similar houses agree": "model error",
                 "both differ": "atypical house", "close to similar": "price noise"}


def short_verdict(v):
    return next((s for k, s in VERDICT_SHORT.items() if str(v).startswith(k)), v)


def main():
    D = load_data()
    T = Tables()
    J = read_json
    best, ts, tm, cs = J("best_config.json"), J("tuning_summary.json"), J("test_metrics.json"), J("coef_summary.json")
    ab_s, er, ed, ms = J("ablation_summary.json"), J("error_summary.json"), J("eda_summary.json"), J("main_summary.json")
    R = lambda n: pd.read_csv(OUT / n)
    bf, res, coef, gv = R("tuning_best_per_family.csv"), R("tuning_results.csv"), R("coefficients.csv"), R("gvif.csv")
    ab, pw, sl = R("ablation.csv"), R("ablation_pairwise.csv"), R("ablation_slices.csv")
    ep, ec, ecorr, edrift, mr = R("error_points.csv"), R("error_compare.csv"), R("eda_correlations.csv"), R("eda_drift.csv"), R("main_results.csv")
    st = best["setting"]
    rtr, rte = D["raw_tr"], D["raw_te"]
    r_of = ecorr.set_index("feature")["r"]
    cf = coef.set_index("feature")
    S = {}

    # =========================================================== 1. Data and setup
    t_split = pd.DataFrame({"Split": ["Train", "Test"], "Period": [tm["train_period"], tm["test_period"]],
                            "Houses": [len(D["ytr"]), len(D["yte"])],
                            "Median price ($k)": [fk(rtr["SalePrice"].median()), fk(rte["SalePrice"].median())]})
    n_dummy = len(D["dummy"])
    t_feat = pd.DataFrame({"Feature group": ["Original (non-dummy)", "One-hot dummies", "Engineered", "Greedy list (models/)", "Total"],
                           "Count": [len(D["original"]) - n_dummy, n_dummy, len(D["engineered"]), ab_s["n_greedy"], len(D["feats"])],
                           "Examples": ["GrLivArea, OverallQual, YearBuilt", "Neighborhood_…, SaleCondition_…",
                                        "TotalArea, HouseAge, Qual_x_Area", f"{ab_s['n_greedy_orig']} original + {ab_s['n_greedy'] - ab_s['n_greedy_orig']} engineered", ""]})
    folds = pd.DataFrame({"Fold": range(1, N_SPLITS + 1), "Train rows": [a for a, _ in ts["fold_sizes"]],
                          "Validation rows": [b for _, b in ts["fold_sizes"]], "Validation period": ts["fold_periods"]})
    rep = D["repaired"]
    S["data"] = f"""## 1. Data and setup

- **Data:** the team's pre-processed files `process_dataset/one_hot_train.csv` / `one_hot_test.csv`, split **chronologically** (README §0). The test set is used once, after every modelling choice is made.
- **Target:** log1p(SalePrice); dollar metrics use expm1 of the prediction.
- **Log transform decided inside every fit:** the team's rule log1p-transforms a numeric feature when its skewness exceeds 0.75. The pre-processing chooses the list once on the whole training set and reuses it for test (`mapping/skewed_features.json`), so the choice has seen every future CV validation fold. Here the raw values of the 39 skew-checked columns are rebuilt from `Dataset/train.csv` with the team's formulas, and the same rule is re-applied **on the training rows of each fit only** (each CV fold, each bootstrap resample, the final model); the chosen transform is then applied unchanged to that fit's validation or test rows. On the full training set this reproduces the team's log-transformed columns exactly.{(' ' + str(len(rep)) + ' test columns had a transform different from train and were corrected: ' + ', '.join(r['column'] for r in rep) + '.') if rep else ''}

{T(t_split, "Chronological train / test split.", "split")}
{T(t_feat, "Feature groups after pre-processing.", "features")}
{T(folds, "Time-series cross-validation folds (expanding window) inside the training set.", "folds")}

### 1.1 Scaling inside cross-validation

Every data-dependent step is **decided on the training part of each fold only and applied unchanged to the validation part** (and, for the final model, to the test set): the outlier rule, the log1p choice, the |r| ranking for feature selection, and the scaler. The validation fold stands in for future data. Standardising it with its own mean and SD would use information unavailable at prediction time (Kaufman et al. 2012; Hastie et al. 2009 §7.10.2), and would give the same house different z-scores in different sets. Re-standardising inside each fold also puts the ordinal, binary and time columns (left unscaled by pre-processing) on the same scale, which penalised models require (Tibshirani 1996).
"""

    # =========================================================== 2. EDA (§4)
    top = ecorr.head(12)
    t_corr = pd.DataFrame({"Feature": top["feature"], "Type": top["kind"], "r": top["r"].map(lambda v: f"{v:+.3f}"),
                           "Effect size (Cohen)": top["effect_size"]})
    dtop = edrift.head(8)
    t_drift = pd.DataFrame({"Feature": dtop["feature"], "SMD (test − train)": dtop["SMD"].map(lambda v: f"{v:+.2f}"),
                            "KS p": dtop["KS_p"].map(fp)})
    eff = ed["effect_counts"]
    n_eng_top = int((ecorr.head(10)["kind"] == "engineered").sum())
    nb_med = rtr.groupby("Neighborhood")["SalePrice"].median()
    rr = lambda f: f"{r_of[f]:+.2f}" if f in r_of else "n/a"
    HYP = [
        ("H1", "Size: larger houses cost more; log price is roughly linear in log size",
         f"r(TotalArea) = {rr('TotalArea')}, r(GrLivArea) = {rr('GrLivArea')}; linear trend in E3", "§8.1 (β > 0)"),
        ("H2", "Quality: overall quality is among the strongest drivers of price",
         f"r(OverallQual) = {rr('OverallQual')}, the strongest original feature (E2)", "§8.1 (β > 0)"),
        ("H3", "Age: older houses are cheaper even at the same size and quality",
         f"r(HouseAge) = {rr('HouseAge')}, but age correlates with quality (E4)", "§8.1 (β < 0)"),
        ("H4", "Location: neighbourhood and zoning shift the price level",
         f"median price ranges ${nb_med.min() / 1e3:.0f}k–${nb_med.max() / 1e3:.0f}k across neighbourhoods (E6)", "§8.1 (dummies, GVIF)"),
        ("H5", "Luxury premium: quality matters more in bigger houses (quality × size)",
         f"r(Qual_x_Area) = {rr('Qual_x_Area')}, the highest of all features (E2)", "§8.1 (β > 0)"),
        ("H6", "Engineered features add predictive value beyond the raw columns",
         f"{n_eng_top} of the 10 strongest features are engineered (E2)", "§7.2 (Set B vs A)"),
        ("H7", "Collinear blocks make a penalised model better than plain OLS",
         "area, quality, garage and age features form correlated blocks (E4)", "§5.2 / §7.1 (CV, OLS vs penalised)"),
    ]
    t_hyp = pd.DataFrame(HYP, columns=["ID", "Hypothesis", "Evidence (training set)", "Tested in"])
    S["eda"] = f"""## 2. Exploratory feature analysis (§4)

All statistics use the training set only; the test set is touched only to measure train → test drift.

- **Target:** raw SalePrice is right-skewed (skew {ed['skew_raw']:.2f}); log1p reduces this to {ed['skew_log']:.2f} (E1), so the model predicts log price.
- **Association:** by Cohen's |r| bands, {eff.get('large', 0)} features are large (≥ 0.5), {eff.get('medium', 0)} medium, {eff.get('small', 0)} small and {eff.get('negligible', 0)} negligible (E2, {T['eda_corr']}).
- **Shape:** the strongest features relate to log price roughly linearly (E3).
- **Redundancy:** area, quality, garage and age features form tightly correlated blocks (E4).
- **Location:** median price differs strongly by neighbourhood (E6).
- **Drift:** {ed['n_drift']} of {ed['n_nondummy']} non-dummy features shift by |SMD| > 0.1 between the periods; apart from YrSold (by construction) the shifts are modest (E5, {T['eda_drift']}).

**How the evidence shaped the features.**
- **Keep:** strong and medium features, and the location dummies (E2, E6).
- **Transform:** the target and skewed areas to log scale (E1, E3); calendar years to ages; month to sin/cos (seasonality has no "December > January" order).
- **Combine:** correlated parts into aggregates (TotalArea, TotalBaths) and add a quality × size interaction (E4, E2).
- **Drop:** weak features are dropped by a correlation threshold tuned by CV (§5.2) rather than by hand; redundancy is handled by the penalty rather than by deleting features.

The hypotheses below are tested in §7.2 and §8.1 ({T['hypothesis_check']}).

{T(t_corr, "Twelve features most correlated with log price on the training set (Pearson r; Cohen's bands 0.1 / 0.3 / 0.5).", "eda_corr")}
{T(t_drift, "Largest train → test shifts among non-dummy features (standardised mean difference; Kolmogorov-Smirnov p).", "eda_drift")}
{T(t_hyp, "Hypotheses derived from the exploratory analysis and where they are tested.", "hypotheses")}

![E1](figures/E1_target.png)
![E2](figures/E2_top_correlations.png)
![E3](figures/E3_top_scatter.png)
![E4](figures/E4_feature_correlation.png)
![E5](figures/E5_drift.png)
![E6](figures/E6_neighbourhood.png)
"""

    # =========================================================== 3. Pipeline & feature engineering (§5.1)
    t_groups = pd.DataFrame([
        ("Transforms", "log1p of skewed numerics (skew > 0.75, decided per fit); sin/cos of MoSold",
         "makes area–price relations linear in logs; month becomes cyclic so Dec and Jan are neighbours", "H1"),
        ("Interactions", "Qual_x_Area, TotalQualCond",
         "lets the quality effect grow with size (luxury premium); one combined condition score", "H2, H5"),
        ("Aggregates", "TotalArea, TotalBaths, TotalPorchArea",
         "buyers value total usable space; one strong feature replaces several correlated parts", "H1, H7"),
        ("Domain-derived", "HouseAge, RemodAge, GarageAge; AvgRoomSize, LotUtilization",
         "age at sale is what a buyer sees, not the calendar year; room size and lot use measure spaciousness", "H1, H3"),
        ("Presence flags", "HasBasement, HasFireplace, HasPool, HasGarage, IsRemod",
         "capture the price step of having an amenity, separate from its size", "H6"),
    ], columns=["Group", "Features", "Why it should help", "Hypothesis"])
    S["pipeline"] = f"""## 3. Pipeline and feature engineering (§5.1)

F15 shows each stage with its input and output. Every data-dependent step (log choice, |r| selection, scaling, outlier rule) is fitted on training rows only.

![F15](figures/F15_pipeline.png)

{T(t_groups, "Engineered feature groups, the rationale for each and the hypothesis it addresses (§4).", "feature_groups")}
Whether these groups help as a whole is tested in the ablation (§7.2, Set B vs A).
"""

    # =========================================================== 4. Model choice & tuning (§5.2)
    fam, l1 = best["family"], best["key"][2]
    curve = res[(res["model"] == fam) & (res["threshold"] == st["threshold"]) & (res["cat_mode"] == st["cat_mode"]) &
                (res["drop_outliers"] == st["drop_outliers"])]
    if l1 is not None:
        curve = curve[np.isclose(curve["extra"], l1)]
    curve = curve.sort_values("alpha").reset_index(drop=True)
    ib = int(curve["cv_rmse"].idxmin())
    pick = sorted(set(list(range(0, len(curve), max(1, len(curve) // 8))) + [ib, len(curve) - 1]))
    cv_sub = curve.loc[pick].reset_index(drop=True)
    t_curve = pd.DataFrame({"α": cv_sub["alpha"].map(lambda v: f"{v:.3g}"), "CV MAE ($k)": cv_sub["cv_mae_usd"].map(fk),
                            "CV RMSE": cv_sub["cv_rmse"].map(lambda v: f"{v:.4f}")})
    ib_sub = int(np.flatnonzero(np.isclose(cv_sub["alpha"], curve.at[ib, "alpha"]))[0])
    se = curve.at[ib, "cv_rmse_sd"] / np.sqrt(N_SPLITS)
    flat = curve[curve["cv_rmse"] <= curve.at[ib, "cv_rmse"] + se]["alpha"]
    vif_max = coef.loc[~coef["is_reference"] & np.isfinite(coef["VIF"]), "VIF"].max()
    n_vif10 = int((coef.loc[~coef["is_reference"], "VIF"] >= 10).sum())
    ols_rmse = bf.set_index("model").at["OLS", "cv_rmse"]
    t_src = pd.DataFrame([
        ("Correlation threshold", ", ".join(f"{t:g}" for t in THRESHOLDS), "Cohen (1988)"),
        ("One-hot dummies", ", ".join(CAT_MODES), "—"),
        ("Drop GrLivArea > 4000 (train folds)", ", ".join(map(str, DROP_OUTLIERS)), "De Cock (2011)"),
        ("Lasso / ElasticNet α", f"{N_ALPHAS} values, α_max → {ALPHA_MIN_RATIO:g}·α_max (log)", "Friedman et al. (2010)"),
        ("ElasticNet l1_ratio", ", ".join(f"{x:g}" for x in L1_RATIOS[:-1]), "Zou & Hastie (2005)"),
        ("Ridge α", f"{N_RIDGE} values, df 1 → 0.95·rank ({ts['ridge_range'][0]:.3g} … {ts['ridge_range'][1]:.3g})", "Hastie et al. (2009) §3.4.1"),
        ("Huber ε", ", ".join(f"{e:g}" for e in HUBER_EPS), "Huber (1981)"),
        ("Huber α", f"{N_HUBER} values from the ridge range", "—"),
        ("Grid spacing", "logarithmic", "Bergstra & Bengio (2012)"),
        ("Cross-validation", f"TimeSeriesSplit, {N_SPLITS} folds", "Bergmeir & Benítez (2012)"),
        ("Top-k (ablation)", ", ".join(map(str, TOPK)), "—"),
        ("Bootstrap resamples", f"B = {N_BOOT}", "Efron & Tibshirani (1993)"),
    ], columns=["Setting", "Values searched", "Source"])
    bfd = pd.DataFrame({"Model": bf["model"], "Hyperparameters": bf["params"], "r threshold": bf["threshold"].map(lambda v: f"{v:g}"),
                        "Dummies": bf["cat_mode"], "Features": bf["n_feats"].map(lambda v: f"{v:.0f}"),
                        "CV RMSE ± SD": [f"{a:.4f} ± {b:.4f}" for a, b in zip(bf["cv_rmse"], bf["cv_rmse_sd"])],
                        "CV MAE ($k)": bf["cv_mae_usd"].map(fk)})
    i_best = int(bf["cv_rmse"].idxmin())
    top10 = res.head(10)
    t10 = pd.DataFrame({"Rank": range(1, 11), "Model": top10["model"], "Hyperparameters": top10["params"],
                        "r threshold": top10["threshold"].map(lambda v: f"{v:g}"), "Dummies": top10["cat_mode"],
                        "Drop outliers": top10["drop_outliers"].map(lambda v: "yes" if v else "no"),
                        "CV RMSE ± SD": [f"{a:.6f} ± {b:.4f}" for a, b in zip(top10["cv_rmse"], top10["cv_rmse_sd"])],
                        "CV MAE ($k)": top10["cv_mae_usd"].map(fk)})
    S["model"] = f"""## 4. Model choice and hyperparameter tuning (§5.2)

**Model.** Our features are strongly correlated ({n_vif10} features have VIF ≥ 10, several effectively infinite because of exact identities such as HouseAge = YrSold − YearBuilt; E4, §8.1), and there are {len(D['feats'])} candidate features for {len(D['ytr']):,} training sales, most of them weak one-hot dummies (E2). We therefore use **{fam}** (Zou & Hastie 2005), which shrinks correlated coefficients together like Ridge and drops useless ones like Lasso. The choice is confirmed by CV: all penalised models beat plain OLS (CV RMSE {ols_rmse:.4f} vs. {best['cv_rmse']:.4f}; H7, S1), and {fam} is the best of the five.
Features are standardised inside each fold. We tune the |r| threshold, one-hot handling, outlier rule, α and l1_ratio by 5-fold TimeSeriesSplit on the **training period only**, minimising CV RMSE on log price. The selected values are threshold {st['threshold']:g}, one-hot = {st['cat_mode']}, outliers {'dropped' if st['drop_outliers'] else 'kept'}, **α = {best['key'][1]:.3g}, l1_ratio = {l1}** ({tm['n_nonzero']} non-zero coefficients).

**Tuning curve.** {T['tuning_curve']} shows CV error along the α path at the selected setting. CV RMSE stays within one standard error of the optimum for α ≈ {flat.min():.3g}–{flat.max():.3g}, so the exact choice matters little there; larger α over-shrinks and the error rises (F4).

{T(t_curve, f"Tuning curve of {fam} (l1_ratio = {l1}, threshold {st['threshold']:g}): time-series CV error along the α path. Bold = selected α.", "tuning_curve", bold=[(ib_sub, 0), (ib_sub, 1), (ib_sub, 2)])}

### 4.1 How the search ranges were decided

A fixed decade grid (1e-4 … 1e2) is a convention: the effect of a given α depends on the number of rows, the feature scales and how the library writes the loss. Ranges are therefore derived from the data.

- **Lasso / ElasticNet:** α_max = max |xⱼᵀy| / (n · l1_ratio) is the smallest α that zeroes every coefficient ({ts['enet_alpha_max']['1.0']:.3g} for Lasso); the path runs down to {ALPHA_MIN_RATIO:g} · α_max (glmnet default for n > p).
- **Ridge:** never zeroes coefficients, so its range spans effective degrees of freedom df(α) = Σ dⱼ²/(dⱼ² + α) from about the rank (no shrinkage) to 1.
- **Edge rule:** a value chosen on the edge of its grid means the grid must be widened (done for l1_ratio and top-k).

{T(t_src, "Tunable settings, the values searched and their sources.", "settings")}
Search: {ts['n_settings']} selection settings × {ts['n_keys']} model hyperparameter values, each scored on {N_SPLITS} time-series folds.

{T(bfd, "Best configuration per model family (time-series CV on the training set; lower is better).", "best_family", bold=[(i_best, 5), (i_best, 6)])}
![F1](figures/F1_threshold_vs_rmse.png)
![F4](figures/F4_alpha_curves.png)
![F5](figures/F5_best_per_family.png)

Top 10 configurations (all in `tuning_results.csv`):

{T(t10, "Top 10 of all tuned configurations, ranked by time-series CV RMSE.", "top10", bold=[(0, 6), (0, 7)])}
![F2](figures/F2_threshold_vs_n_features.png)
![F3](figures/F3_selection_settings.png)
"""

    # =========================================================== 5. Metrics & baselines (§6)
    thr_usd = ms["threshold_usd"]
    t_met = pd.DataFrame([
        ("MAE", "$", "typical error size", "weights all errors linearly", "the user's cost: dollars a buyer or seller would misjudge; carries the decision threshold"),
        ("RMSE", "log price", "large (relative) errors", "sensitive to outliers", "the training objective and Kaggle metric; ≈ relative error"),
        ("MAPE", "%", "relative error", "unstable when y ≈ 0", "comparable across cheap and expensive houses (prices are never near 0)"),
        ("R²", "none", "variance explained", "can be negative on test data", "shows how much better than predicting the mean"),
        ("Within ±10%", "% of houses", "share of usable estimates", "ignores the size of misses", "how often an estimate is close enough to act on"),
    ], columns=["Metric", "Unit", "Emphasises", "Caution", "Why we use it"])
    t_base = pd.DataFrame([
        ("Trivial", "Training mean", "none", "—"),
        ("Trivial", "Persistence (last 3 months)", "median price of the last 3 months of the training part", "—"),
        ("Simple", "Single-feature OLS (OverallQual)", "OverallQual (strongest original feature, E2)", "—"),
        ("Simple", "Single-feature OLS (log GrLivArea)", "log living area (most intuitive size feature)", "—"),
        ("Strong", ms["strong_name"], "all raw columns (one-hot coded)",
         f"best of {ms['n_strong_cfgs']} random-forest / gradient-boosting settings by the same CV ({ms['strong_params'].strip('{}').replace(chr(39), '')})"),
    ], columns=["Tier", "Baseline", "Inputs", "Tuning"])
    S["metrics"] = f"""## 5. Metrics and baselines (§6)

**Metrics.** We report MAE in dollars as the primary metric, because a buyer, seller or lender loses money in proportion to the dollar error. We add RMSE on log price, the training objective, which penalises large relative misses, and three supporting metrics ({T['metrics']}).

**Decision threshold: MAE ≤ ${thr_usd:,.0f}** (10% of the median training price, ${ms['median_price']:,.0f}). Below this error an automated estimate is close enough to set a listing price or an offer without a manual appraisal. Above it, the estimate would change that decision. §7 compares every method against this threshold. The 10% band is our assumption about the user; adjust it if your problem statement defines a different user.

**Baselines** of increasing strength ({T['baselines']}) are scored with exactly the same protocol as our model: same time-series folds, then one fit on the full training set evaluated on the test set. They follow the team's `model/baseline2.py` (trivial / simple / strong tiers), plus a persistence baseline for the temporal setting.

{T(t_met, "Evaluation metrics and the reason for each.", "metrics")}
{T(t_base, "Baselines of increasing strength.", "baselines")}
"""

    # =========================================================== 6. Main results (§7.1)
    mrd = pd.DataFrame({"Method": mr["method"],
                        "CV MAE ($k)": [f"{fk(a)} ± {fk(b)}" for a, b in zip(mr["cv_mae"], mr["cv_mae_sd"])],
                        "CV RMSE": [f"{a:.4f} ± {b:.4f}" for a, b in zip(mr["cv_rmse"], mr["cv_rmse_sd"])],
                        "Test MAE ($k)": mr["test_mae"].map(fk), "Test RMSE": mr["test_rmse"].map(lambda v: f"{v:.4f}"),
                        "Test R²": mr["test_r2"].map(lambda v: f"{v:.3f}"), "Within ±10% (%)": mr["test_ppe10"].map(lambda v: f"{v:.1f}")})
    mr_bold = [(int(mr["cv_mae"].idxmin()), 1), (int(mr["cv_rmse"].idxmin()), 2), (int(mr["test_mae"].idxmin()), 3),
               (int(mr["test_rmse"].idxmin()), 4), (int(mr["test_r2"].idxmax()), 5), (int(mr["test_ppe10"].idxmax()), 6)]
    ours, rival = mr.iloc[-1], mr.set_index("method").loc[ms["rival"]]
    mf = R("main_results_folds.csv")
    fold_gap = (mf[mf["method"] == ms["rival"]].sort_values("fold")["mae"].to_numpy()
                - mf[mf["method"] == ms["ours"]].sort_values("fold")["mae"].to_numpy())
    meets = mr[mr["test_mae"] <= thr_usd]["method"].tolist()
    cv_meet = mr[mr["cv_mae"] <= thr_usd]["method"].tolist()
    cv_word = "better" if ms["cv_gap"] > 0 else "worse"
    test_word = "lower" if ms["test_gap"] > 0 else "higher"
    sig_cv = ms["p_cv"] < 0.05
    sig_test = min(ms["p_boot"], ms["p_wilcoxon"]) < 0.05 and not (ms["test_ci"][0] <= 0 <= ms["test_ci"][1])
    claim = ("a clear win" if ms["cv_gap"] > 0 and ms["test_gap"] > 0 and sig_cv and sig_test else
             "a consistent but small gain" if ms["cv_gap"] > 0 and ms["test_gap"] > 0 else
             "performance on par with the strong baseline, not a win")
    t_test = pd.DataFrame([
        ("RMSE (log price)", f"{tm['rmse']:.4f}", f"[{tm['rmse_ci'][0]:.4f}, {tm['rmse_ci'][1]:.4f}]"),
        ("MAE ($)", f"{tm['mae']:,.0f}", f"[{tm['mae_ci'][0]:,.0f}, {tm['mae_ci'][1]:,.0f}]"),
        ("Median absolute error ($)", f"{tm['medae']:,.0f}", "—"),
        ("MAPE (%)", f"{tm['mape']:.1f}", "—"),
        ("R² (log price)", f"{tm['r2']:.3f}", "—"),
        ("Mean residual (bias)", f"{tm['mean_resid']:+.4f}", f"t-test {pv(tm['mean_resid_p'])}"),
        ("Breusch-Pagan", "—", f"{pv(tm['bp_p'])}"),
        ("Durbin-Watson", f"{tm['dw']:.2f}", "—"),
    ], columns=["Metric", "Value", "95% CI / test"])
    S["main"] = f"""## 6. Main results (§7.1)

{T(mrd, f"Main results. CV: mean ± SD over {N_SPLITS} TimeSeriesSplit folds; test: chronological test set ({len(D['yte'])} sales). Bold = best in each column. Decision threshold: MAE ≤ ${thr_usd / 1e3:.1f}k.", "main_results", bold=mr_bold)}
![F16](figures/F16_main_results.png)

**Experimental results.**
- **Against the threshold:** on the test set {", ".join(meets) if meets else "no method"} {'meets' if len(meets) == 1 else 'meet'} the ${thr_usd / 1e3:.1f}k threshold. In CV, {", ".join(cv_meet) if cv_meet else "no method"} {'does' if len(cv_meet) == 1 else 'do'}. The trivial and single-feature baselines miss it by a wide margin.
- **Against the strongest baseline ({ms['rival']}):** ours is {cv_word} in CV by ${abs(ms['cv_gap']):,.0f} MAE (corrected resampled t-test over folds, Nadeau & Bengio 2003: {pv(ms['p_cv'])}). On the test set our MAE is {test_word} by ${abs(ms['test_gap']):,.0f} (paired bootstrap 95% CI of the gap [{ms['test_ci'][0]:+,.0f}, {ms['test_ci'][1]:+,.0f}] $, Wilcoxon {pv(ms['p_wilcoxon'])}). We therefore claim **{claim}**.
- **Why the linear model keeps up with boosting:** per fold, the baseline − ours MAE gap is {", ".join(f"${g:,.0f}" for g in fold_gap)} for training sizes {", ".join(str(a) for a, _ in ts['fold_sizes'])}; with all {len(D['ytr']):,} training sales the gap on the test set is {ms['test_gap']:+,.0f} $. The linear model gets the shape of the relationships for free from the log transforms, ages and aggregates (§5.1), while the trees must learn every curve from the data, so the linear model's advantage is largest when training data are scarce and fades as data grow.
- Gap definitions: CV gap = baseline − ours (positive = ours better); test gap = baseline MAE − our MAE per house.

**Final model diagnostics** ({T['test_details']}, F7): typical error ≈ {tm['rmse'] * 100:.0f}% of the price; mean residual {tm['mean_resid']:+.4f} ({pv(tm['mean_resid_p'])}, {'no' if tm['mean_resid_p'] >= 0.05 else 'a'} level shift between periods); Breusch-Pagan {pv(tm['bp_p'])} ({'error variance grows with price, hence bootstrap intervals' if tm['bp_p'] < 0.05 else 'constant variance'}); Durbin-Watson {tm['dw']:.2f} (no serial correlation).

{T(t_test, f"Final model ({best['family']}, {tm['n_feats']} features, {tm['n_nonzero']} non-zero) on the chronological test set; CIs from {N_BOOT} bootstrap resamples.", "test_details")}
![F7](figures/F7_residual_diagnostics.png)
"""

    # =========================================================== 7. Ablation (§7.2)
    a = ab.copy()
    t_ab = pd.DataFrame({"Set": a["set"], "Feature set": a["definition"], "Features": a["n_feats"],
                         "CV MAE ($k) ± SD": [f"{fk(m)} ± {fk(s)}" for m, s in zip(a["cv_mae_usd"], a["cv_mae_usd_sd"])],
                         "Test MAE ($k)": a["test_mae_usd"].map(fk), "Test RMSE": a["test_rmse"].map(lambda v: f"{v:.4f}")})
    ab_bold = [(int(a["cv_mae_usd"].idxmin()), 3), (int(a["test_mae_usd"].idxmin()), 4), (int(a["test_rmse"].idxmin()), 5)]
    t_pw = pd.DataFrame({"Comparison": pw["comparison"], "Δ CV MAE ($k)": pw["cv_diff_usd"].map(lambda v: f"{v / 1e3:+.2f}"),
                         "p (CV, t)": pw["p_cv_corrected_t_holm"].map(fp),
                         "Δ test MAE ($k) [95% CI]": [f"{d / 1e3:+.2f} [{l / 1e3:+.2f}, {h / 1e3:+.2f}]" for d, l, h in zip(pw["test_diff_usd"], pw["test_ci_lo_usd"], pw["test_ci_hi_usd"])],
                         "p (boot.)": pw["p_test_bootstrap_holm"].map(fp), "p (Wilcoxon)": pw["p_test_wilcoxon_holm"].map(fp)})
    pw_bold = [(i, c) for i in range(len(pw)) for c, col in [(2, "p_cv_corrected_t_holm"), (4, "p_test_bootstrap_holm"), (5, "p_test_wilcoxon_holm")] if pw.at[i, col] < 0.05]
    sets = list("ABCDEF")
    t_sl = pd.DataFrame({"Slice": sl["slice"], "n": sl["n"], **{k: sl[f"MAE_usd {k}"].map(fk) for k in sets}})
    sl_bold = [(i, 2 + int(np.argmin([sl.at[i, f"MAE_usd {k}"] for k in sets]))) for i in range(len(sl))]

    def pair(x, y):
        r = pw[pw["comparison"] == f"{x} vs {y}"].iloc[0]
        s = "significant" if min(r["p_test_bootstrap_holm"], r["p_test_wilcoxon_holm"]) < 0.05 else "not significant"
        return f"test MAE {x} − {y} = {r['test_diff_usd']:+,.0f} $ [{r['test_ci_lo_usd']:+,.0f}, {r['test_ci_hi_usd']:+,.0f}], {s} after Holm"

    ab_i = ab.set_index("set")
    same_ac = np.isclose(ab_i.at["A", "test_mae_usd"], ab_i.at["C", "test_mae_usd"])
    same_bd = np.isclose(ab_i.at["B", "test_mae_usd"], ab_i.at["D", "test_mae_usd"])
    S["ablation"] = f"""## 7. Feature ablation (§7.2)

Same model family (**{ab_s['family']}**), chronological split, time-series folds, tuning procedure and outlier rule for every set; only the feature set changes.

{T(t_ab, "Feature ablation. CV = time-series 5-fold on the training set (± fold SD); test = chronological test set. Bold = best.", "ablation", bold=ab_bold)}
![F8](figures/F8_ablation.png)

**Significance.** CV folds: corrected resampled t-test (Nadeau & Bengio 2003; variance × (1/k + n_val/n_train)). Test set: paired bootstrap of the mean absolute-error difference and the Wilcoxon signed-rank test (Wilcoxon 1945). All p-values are Holm-adjusted over the 15 pairs (Holm 1979). Δ = first set − second set, so positive = the second set is better.

{T(t_pw, "Pairwise comparison of the feature sets (Holm-adjusted p; bold p < 0.05).", "ablation_pairwise", bold=pw_bold)}
![F10](figures/F10_ablation_pairwise.png)

{T(t_sl, "Test MAE ($k) by slice of the test set (bold = best set in the row).", "ablation_slices", bold=sl_bold)}
![F9](figures/F9_topk_curves.png)

**Findings.**
- **Engineered features (B vs A, H6):** {pair('A', 'B')}.
- **Correlation ranking (C, D vs A, B):** the tuned k keeps {"essentially every feature, so C gives the same predictions as A" if same_ac else "most features"}{", and D the same as B" if same_bd else ""}. Features a |r| filter would drop are already set to zero by ElasticNet.
- **Greedy original only (E):** worse than every other set ({pair('E', 'F')}).
- **Greedy original + engineered (F) vs B:** {pair('B', 'F')}, with {int(ab_i.at['F', 'n_feats'])} instead of {int(ab_i.at['B', 'n_feats'])} features. Compare E and F on the test set only: their list was selected on the whole training set, so their CV scores are optimistic (Ambroise & McLachlan 2002).
"""
    S["ablation_valid"] = """### 7.1 Is the ablation design correct?

Yes: the six sets share the model family, split, folds, tuning procedure and outlier rule, and the top-k sets rank features inside each training fold. The one caveat is that the greedy sets (E, F) were selected on the whole training set, so their CV scores contain selection bias (Hastie et al. 2009 §7.10.2); their test scores are clean. The greedy p-values also come from an OLS fit with exact collinearity (e.g. HouseAge = YrSold − YearBuilt), which makes individual p-values unreliable.
"""

    # =========================================================== 8. Coefficients (§8.1)
    nonref = coef[~coef["is_reference"]]
    sig = coef[coef["p"] < 0.05]
    key = sig[sig["practical"] == "yes"].head(12)
    t_key = pd.DataFrame({"Feature": key["feature"], "Unit": key["unit"], "β": key["beta"].map(lambda v: f"{v:+.3f}"),
                          "95% CI": [f"[{a:+.3f}, {b:+.3f}]" for a, b in zip(key["ci_lo"], key["ci_hi"])], "p": key["p"].map(fp),
                          "Sel. freq.": key["sel_freq"].map(lambda v: f"{v:.2f}"),
                          "VIF": key["VIF"].map(lambda v: "inf" if np.isinf(v) else f"{v:.1f}"),
                          "Effect ($k)": key["effect_usd"].map(lambda v: f"{v / 1e3:+,.1f}")})
    vif_bold = [(i, 6) for i, v in enumerate(key["VIF"]) if v >= 10]
    hyp = coef[coef["expected"].notna()]
    rev = hyp[hyp["hypothesis_check"].str.startswith("reversal")]
    snp = sig[sig["practical"] == "negligible"]
    big_ns = coef.head(10)[coef.head(10)["p"] >= 0.05]
    g = gv.head(10)
    t_gv = pd.DataFrame({"Categorical": g["categorical"], "df": g["df"], "GVIF^(1/(2df))": g["GVIF^(1/(2df))"].map(lambda v: f"{v:.2f}"),
                         "Reference level": g["reference_level"]})

    def beta_txt(f):
        if f not in cf.index:
            return f"{f}: not selected", None
        r = cf.loc[f]
        return f"{f}: β = {r['beta']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}], {pv(r['p'])}", r

    def verdict(feats, sign):
        rows = [cf.loc[f] for f in feats if f in cf.index]
        ok = [r for r in rows if r["p"] < 0.05 and np.sign(r["beta"]) == sign]
        bad = [r for r in rows if r["p"] < 0.05 and np.sign(r["beta"]) == -sign]
        return "contradicted" if bad and not ok else "supported" if ok and len(ok) == len(rows) else "partly supported" if ok else "not supported"

    nb_sig = sig[sig["feature"].str.startswith("Neighborhood_")]
    zone = sig[sig["feature"].str.startswith("MSZoning_")]
    pAB = pw[pw["comparison"] == "A vs B"].iloc[0]
    h6_sig = min(pAB["p_test_bootstrap_holm"], pAB["p_test_wilcoxon_holm"]) < 0.05
    pen_best = bf[bf["model"] != "OLS"]["cv_rmse"].min()
    checks = [
        ("H1 size", "; ".join(beta_txt(f)[0] for f in ["TotalArea", "GrLivArea"]), verdict(["TotalArea", "GrLivArea"], 1)),
        ("H2 quality", "; ".join(beta_txt(f)[0] for f in ["OverallQual", "TotalQualCond"]), verdict(["OverallQual", "TotalQualCond"], 1)),
        ("H3 age", beta_txt("HouseAge")[0] + " (size and quality held fixed)", verdict(["HouseAge"], -1)),
        ("H4 location", f"{len(nb_sig)} neighbourhood dummies significant; zoning: " +
         (", ".join(f"{r.feature} {r.effect_pct:+.0f}%" for r in zone.itertuples()) or "none significant"),
         "supported" if len(nb_sig) or len(zone) else "not supported"),
        ("H5 luxury premium", beta_txt("Qual_x_Area")[0], verdict(["Qual_x_Area"], 1)),
        ("H6 engineered features", f"Set B vs A: test MAE {pAB['test_diff_usd']:+,.0f} $ [{pAB['test_ci_lo_usd']:+,.0f}, {pAB['test_ci_hi_usd']:+,.0f}] (§7.2)",
         "supported" if h6_sig and pAB["test_diff_usd"] > 0 else "direction supported, not significant" if pAB["test_diff_usd"] > 0 else "not supported"),
        ("H7 penalised > OLS", f"CV RMSE OLS {ols_rmse:.4f} vs. best penalised {pen_best:.4f} (S1)",
         "supported" if pen_best < ols_rmse else "not supported"),
    ]
    t_check = pd.DataFrame(checks, columns=["Hypothesis", "Evidence", "Verdict"])
    S["coef"] = f"""## 8. Coefficients and statistical significance (§8.1)

- **Scale:** coefficients are standardised (β = change in log price per +1 SD of the feature).
- **Test:** {cs['method']}. Percentile 95% CI; two-sided p = 2·min(P*(β ≤ 0), P*(β ≥ 0)). *Sel. freq.* = share of resamples with a non-zero coefficient (Meinshausen & Bühlmann 2010).
- **VIF:** computed on a reference-coded design (one level per categorical dropped). `inf` = exact identity, e.g. HouseAge = YrSold − YearBuilt. Categoricals are judged as a group by GVIF^(1/(2df)) against √10 ≈ 3.16 (Fox & Monette 1992).
- **Practical significance:** effect on the median house (${cs['median_price']:,.0f}); below {cs['practical']:.0%} of price counts as negligible (Wasserstein & Lazar 2016).

**Findings.** {len(sig)} of {len(coef)} features are significant (p < 0.05); {len(sig) - len(snp)} also change the median price by ≥ {cs['practical']:.0%}, and {len(snp)} are significant but negligible ({", ".join(snp['feature']) or "none"}). There are **{len(rev)} sign reversals** among {len(hyp)} features with a prior hypothesis. {int((nonref['VIF'] >= 10).sum())} features have VIF ≥ 10: the area, quality and age blocks must be read as groups, not one coefficient at a time.{(" Large but non-significant: " + ", ".join(f"{r['feature']} (p = {r['p']:.2f})" for _, r in big_ns.iterrows()) + ", rare categories with unstable estimates.") if len(big_ns) else ""}

{T(t_key, "Significant and practically relevant coefficients of the final model (bootstrap 95% CI; effect on the median house for +1 SD or 0 → 1). Bold VIF ≥ 10.", "coefficients", bold=vif_bold)}
![F6](figures/F6_coefficients_forest.png)

**Hypotheses revisited.** Each hypothesis from §4 ({T['hypotheses']}) against the evidence:

{T(t_check, "The hypotheses of §4 revisited with the final model's coefficients and the ablation.", "hypothesis_check")}
{T(t_gv, "Generalised VIF of categorical variables (values above 3.16 correspond to VIF > 10).", "gvif", bold=[(i, 2) for i, v in enumerate(g["GVIF^(1/(2df))"]) if v > 10 ** 0.5])}
<details><summary>Full coefficient table ({len(coef)} features)</summary>

{md(coef[['feature', 'unit', 'beta', 'ci_lo', 'ci_hi', 'p', 'sel_freq', 'VIF', 'marginal_r', 'effect_usd', 'practical', 'hypothesis_check']], '{:.4f}')}

</details>
"""

    # =========================================================== 9. Error analysis (§8.2)
    n_cases = er["n_cases"]
    e = ep.head(n_cases).copy()
    e_all = ep.copy()
    e_all["verdict"] = e_all["comps_verdict"].map(short_verdict)
    e["cause"] = e["Likely cause"].map(short_cause)
    e["verdict"] = e["comps_verdict"].map(short_verdict)
    t_err = pd.DataFrame({"#": e["#"], "Id": e["Id"], "Sale": e["SaleCondition"], "Actual ($k)": e["Actual"].map(fk),
                          "Pred. ($k)": e["Pred"].map(fk), "Comps ($k)": e["comps_median"].map(fk),
                          "Error ($k)": e["Error"].map(lambda v: f"{v / 1e3:+,.1f}"), "z": e["z"].map(lambda v: f"{v:+.2f}"),
                          "Comparable-sales verdict": e["verdict"], "Rule-based cause": e["cause"]})
    vc = e_all["verdict"].value_counts()
    t_cmp = pd.DataFrame({"Feature": ec["feature"].replace({"Actual": "SalePrice ($)"}), "Flagged mean": ec["flagged_mean"].map(lambda v: f"{v:,.1f}"),
                          "Other test houses": ec["rest_mean"].map(lambda v: f"{v:,.1f}"), "SMD": ec["SMD"].map(lambda v: f"{v:+.2f}")})
    cases = [f"- **{r['#']} (Id {r['Id']}, {r['SaleCondition']}, {str(r['date'])[:7]}):** sold ${r['Actual']:,.0f}, predicted ${r['Pred']:,.0f} "
             f"({r['Error'] / r['Actual'] * 100:+.0f}% error). {r['GrLivArea']:,.0f} sq ft, quality {r['OverallQual']}, condition {r['OverallCond']}, built {r['YearBuilt']}. "
             f"Ten similar houses, {r['comps_scope']}, sold for a median ${r['comps_median']:,.0f}: actual {r['actual_vs_comps_pct']:+.0f}%, prediction {r['pred_vs_comps_pct']:+.0f}% vs. them → **{r['verdict']}**. "
             f"Rule-based cause: {r['cause']}." for _, r in e.iterrows()]
    main_kind = er["main_kind"]
    S["error"] = f"""## 9. Error analysis (§8.2, test houses outside ±2 SD)

Residual = actual − predicted on the log scale. SD = {er['sd']:.4f}, so ±2 SD ≈ −{er['usd_band'][1]:.0f}% / +{er['usd_band'][0]:.0f}% of the price.
**{er['n_flag']} of {er['n_test']}** test houses fall outside (≈ {er['expected_flag']:.0f} expected under normality): {er['n_over']} over-predicted, {er['n_under']} under-predicted.
Each case is examined two ways:
1. **Comparable sales** (as in appraisal; Pagourtzi et al. 2003): the median price of the 10 most similar training houses in the same neighbourhood, by standardised size, quality, condition, age and garage. If the actual price is far from them while the prediction is close, the house differs in ways the features do not record (*missing information*). If the comparables agree with the actual price but not with the prediction, the model itself is wrong (*model error*).
2. **Rule-based cause** from the inputs: price or size beyond the training range, non-Normal sale type, poor condition.

The {n_cases} largest errors (|z| ≥ {e['z'].abs().min():.2f}) are analysed one by one below; all {er['n_flag']} flagged houses enter the pattern statistics, and the full list with comparables is in `error_points.csv`.

{T(t_err, f"The {n_cases} largest test errors (|residual| > 2 SD, ranked by |z|). Comps = median sale price of the 10 most similar training houses; error = actual − predicted.", "error_cases")}
**Case by case**

{chr(10).join(cases)}

![F11](figures/F11_error_points.png)
![F13](figures/F13_error_feature_scatter.png)
![F14](figures/F14_error_profiles.png)

**Shared pattern.**
- **Mostly missing information, not model failure.** Comparable-sales verdicts over all {er['n_flag']} flagged houses: {", ".join(f"{k} × {v}" for k, v in vc.items())}. In most cases similar houses sold near the model's prediction, so the price was set by something the data does not record.
- **Abnormal sales dominate.** Non-Normal sales are {er['nonnormal_flag']:.0%} of the flagged houses vs. {er['nonnormal_all']:.0%} of the test set (F12); the forced-sale discount is invisible in the features.
- **Older, lower-quality, cheaper houses.** Flagged houses are older and in worse condition than the rest ({T['error_compare']}, F13–F14). At the low end the training data are sparse, so prices beyond the training range are extrapolated.
- **Model errors.** The *model error* cases ({", ".join(e_all.loc[e_all['verdict'] == 'model error', '#'])}) are sales whose comparables agree with the actual price; they point to coefficient trade-offs inside the collinear area / quality / condition blocks.

**First fix to try:** {er['fixes'].get(main_kind, er['fixes']['not explained by available features'])}.

{T(t_cmp, "Flagged vs. other test houses (raw units; |SMD| > 0.1 = a real difference).", "error_compare", bold=[(i, 3) for i, v in enumerate(ec["SMD"]) if abs(v) > 0.1])}
![F12](figures/F12_error_patterns.png)
"""

    S["standards"] = """## 10. Evaluation standards used and why

| Standard | Section | Why |
|:---|:---|:---|
| Time-series CV only, fold-only preprocessing | §1, §4 | features include time; validation rows must not inform training |
| MAE in $ with a decision threshold; RMSE, MAPE, R², within ±10% | §5 | the user's cost in dollars plus relative and explained-variance views |
| Baselines of increasing strength under the same protocol | §5, §6 | shows what the features and the model add |
| Corrected t (folds) + paired bootstrap / Wilcoxon (test) vs. the strongest baseline | §6 | whether the gap exceeds chance |
| Mean-residual t-test, Breusch-Pagan, Durbin-Watson | §6 | level shift between periods; justify bootstrap intervals |
| Bootstrap CI / p, selection frequency, VIF / GVIF | §8 | valid inference for a penalised model; stability; collinearity |
| Holm correction | §7 | 15 pairwise ablation comparisons |
| Comparable sales + rule-based causes | §9 | separate missing information from model error |
"""
    S["limits"] = """## 11. Limitations

- Hyperparameters are chosen on the same CV that reports the tuning score (no nested CV); the test set is the unbiased estimate.
- The 10% decision threshold is an assumption about the user, not a measured cost.
- Greedy feature lists were selected on the whole training set, so CV scores of Sets E and F are optimistic.
- Bootstrap p-values after penalised selection are approximate.
- Comparable-sales verdicts use 10 neighbours on 7 features; they are a diagnostic, not a valuation.
"""
    S["refs"] = "## References\n\n" + REFERENCES + "\n"

    head = "# Linear Regression on Ames House Prices (time-split data)\n\n"
    gen = ("Generated by `model/report.py` (`python model/run_all.py` reruns everything). Figures: `results/figures/`; "
           "tables as CSV in `results/`, as LaTeX (booktabs, ACL style) in `results/tables/` named by table code.\n\n")
    order = ["data", "eda", "pipeline", "model", "metrics", "main", "ablation"]
    full = head + gen + "\n".join(S[k] for k in order) + S["ablation_valid"] + "\n".join(S[k] for k in ["coef", "error", "standards", "limits", "refs"])
    (OUT / "report.md").write_text(full, encoding="utf-8")
    summary = (head.replace("(time-split data)", "— summary") + "Key sections of `report.md` (same table numbers).\n\n" +
               "\n".join(S[k] for k in order + ["coef", "error", "refs"]))
    (OUT / "report_summary.md").write_text(re.sub(r"^(#{2,3}) \d+(\.\d+)*\.? ", lambda m: m.group(1) + " ", summary, flags=re.M), encoding="utf-8")
    write_json(T.codes, "table_codes.json")
    print(f"report -> {OUT / 'report.md'}, {OUT / 'report_summary.md'}; {len(T.codes)} tables -> {OUT / 'tables'}")


if __name__ == "__main__":
    main()
