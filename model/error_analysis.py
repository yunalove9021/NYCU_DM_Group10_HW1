"""Step 4: error analysis of the final model on the time-split test set.

Points whose residual (log scale, actual - predicted) lies outside +/- 2 SD of the test residuals are analysed:
input context, rule-based likely cause, and how the flagged houses differ from the rest.
Outputs: results/error_points.csv, results/error_compare.csv, results/error_summary.json, figures F11-F12.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from common import BLUE, GRAY, INK, MUTED, ORANGE, OUT, RED, SURFACE, load_data, save, write_json

PCT = LinearSegmentedColormap.from_list("pct", [ORANGE, "#f0efec", BLUE])

CONTEXT = ["Neighborhood", "MSZoning", "SaleType", "SaleCondition", "GrLivArea", "TotalBsmtSF", "LotArea",
           "OverallQual", "OverallCond", "Functional", "YearBuilt", "GarageCars"]
KEY = ["GrLivArea", "TotalBsmtSF", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "GarageCars"]
N_COMPS = 10
N_CASES = 5  # cases analysed one by one (largest |z|); all flagged houses still enter the pattern statistics
SALE_MSG = {"Abnorml": "missing information: abnormal sale (foreclosure / short sale / trade), discount not visible in house features",
            "Family": "missing information: sale between relatives, not at market price",
            "Partial": "measurement mismatch: new home sold before completion, recorded features may not match what was paid for",
            "AdjLand": "missing information: adjoining-land purchase",
            "Alloca": "missing information: allocation sale of linked properties"}
FIXES = {
    "extrapolation": "restrict the scope or cap extreme size features (De Cock 2011 recommends excluding > 4000 sq ft houses), or use a robust loss",
    "missing information": "model non-Normal sales separately or add SaleCondition x size / quality interactions",
    "measurement mismatch": "add SaleType_New / SaleCondition_Partial x size interactions for new builds",
    "regime change": "add a time trend / recent-market feature, or re-fit more often so the model sees the latest prices",
    "not explained by available features": "collect finer features (interior condition, exact location)",
}


def likely_cause(row, lo, hi):
    causes = []
    for f in ["GrLivArea", "TotalBsmtSF", "LotArea"]:
        if row[f] > hi[f]:
            causes.append(f"extrapolation: {f} = {row[f]:,.0f} above training 99th pct ({hi[f]:,.0f})")
    if row["GrLivArea"] < lo["GrLivArea"]:
        causes.append(f"extrapolation: GrLivArea = {row['GrLivArea']:,.0f} below training 1st pct ({lo['GrLivArea']:,.0f})")
    if row["Actual"] < lo["SalePrice"] or row["Actual"] > hi["SalePrice"]:
        causes.append(f"extrapolation: price ${row['Actual']:,.0f} outside the training 1st-99th pct range")
    if row["SaleCondition"] in SALE_MSG:
        causes.append(SALE_MSG[row["SaleCondition"]])
    if row["OverallCond"] <= 3 or row["Functional"] in ("Maj1", "Maj2", "Sev", "Sal"):
        causes.append(f"missing information: poor condition / functional damage (OverallCond {row['OverallCond']}, Functional {row['Functional']})")
    if row["MSZoning"] == "C (all)":
        causes.append("missing information: commercial zoning, few comparable sales")
    return "; ".join(causes) or "not explained by available features (unrecorded condition / location detail or price noise)"


def main():
    D = load_data()
    pr = pd.read_csv(OUT / "test_predictions.csv", parse_dates=["date"])
    raw = D["raw_te"]
    assert (pr["Id"].to_numpy() == raw["Id"].to_numpy()).all()
    df = pd.concat([pr, raw[CONTEXT]], axis=1)
    df["resid_log"] = df["actual_log"] - df["pred_log"]
    sd = df["resid_log"].std(ddof=1)
    df["z"] = df["resid_log"] / sd
    df["Actual"], df["Pred"] = df["actual_usd"], df["pred_usd"]
    df["Error"] = df["Actual"] - df["Pred"]
    df["flag"] = df["z"].abs() > 2

    rtr = D["raw_tr"]
    lo = {f: np.percentile(rtr[f], 1) for f in ["GrLivArea", "SalePrice"]}
    hi = {f: np.percentile(rtr[f], 99) for f in ["GrLivArea", "TotalBsmtSF", "LotArea", "SalePrice"]}
    fl = df[df["flag"]].reindex(df.loc[df["flag"], "z"].abs().sort_values(ascending=False).index).copy()
    fl["Likely cause"] = [likely_cause(r, lo, hi) for _, r in fl.iterrows()]
    fl.insert(0, "#", [f"#{i + 1}" for i in range(len(fl))])
    fl["direction"] = np.where(fl["Error"] > 0, "under-predicted", "over-predicted")

    # comparable sales: the N_COMPS most similar training houses (same neighbourhood if it has enough sales),
    # distance on the standardised KEY features, as an appraiser would do
    Ktr = rtr[KEY].fillna(0).to_numpy(float)
    mu, sdv = Ktr.mean(0), Ktr.std(0)
    comp_rows = []
    for _, r in fl.iterrows():
        pool = rtr[rtr["Neighborhood"] == r["Neighborhood"]]
        scope = f"same neighbourhood ({len(pool)} train sales)"
        if len(pool) < N_COMPS:
            pool, scope = rtr, "all neighbourhoods (too few local sales)"
        Z = (pool[KEY].fillna(0).to_numpy(float) - mu) / sdv
        z = (r[KEY].astype(float).to_numpy() - mu) / sdv
        idx = np.argsort(((Z - z) ** 2).sum(1))[:N_COMPS]
        med = float(pool["SalePrice"].iloc[idx].median())
        a, pv = r["Actual"] / med - 1, r["Pred"] / med - 1
        if abs(a) > 0.25 and abs(pv) <= 0.15:
            verdict = "house sold far from similar houses while the model agrees with them: information the features miss (or noise)"
        elif abs(pv) > 0.15 and abs(a) <= 0.15:
            verdict = "similar houses agree with the actual price but the model does not: model error (extrapolation / coefficients)"
        elif abs(a) > 0.15 and abs(pv) > 0.15:
            verdict = "both differ from similar houses: an atypical house with few true comparables"
        else:
            verdict = "close to similar houses on both sides: residual within normal price noise"
        comp_rows.append({"comps_median": med, "actual_vs_comps_pct": a * 100, "pred_vs_comps_pct": pv * 100,
                          "comps_scope": scope, "comps_verdict": verdict})
    fl = pd.concat([fl.reset_index(drop=True), pd.DataFrame(comp_rows)], axis=1)
    prof = pd.DataFrame({f: [stats.percentileofscore(rtr[f].fillna(0), v) for v in fl[f]] for f in KEY})
    prof["SalePrice"] = [stats.percentileofscore(rtr["SalePrice"], v) for v in fl["Actual"]]
    prof.insert(0, "#", fl["#"].to_numpy())
    prof.to_csv(OUT / "error_profiles.csv", index=False)
    cols = ["#", "Id", "date", "Neighborhood", "SaleCondition", "SaleType", "GrLivArea", "TotalBsmtSF", "LotArea", "OverallQual",
            "OverallCond", "YearBuilt", "GarageCars", "Actual", "Pred", "Error", "resid_log", "z", "direction", "Likely cause",
            "comps_median", "actual_vs_comps_pct", "pred_vs_comps_pct", "comps_scope", "comps_verdict"]
    fl[cols].to_csv(OUT / "error_points.csv", index=False)

    # how flagged houses differ from the rest
    rest = df[~df["flag"]]
    num = ["GrLivArea", "TotalBsmtSF", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "Actual"]
    cmp = pd.DataFrame([{"feature": f, "flagged_mean": fl[f].mean(), "rest_mean": rest[f].mean(),
                         "SMD": (fl[f].mean() - rest[f].mean()) / np.sqrt((fl[f].var() + rest[f].var()) / 2)} for f in num])
    cmp.to_csv(OUT / "error_compare.csv", index=False)
    kinds = pd.Series([k for c in fl["Likely cause"] for k in {x.split(":")[0] for x in c.split("; ")}]).value_counts()
    sale_share = pd.DataFrame({"flagged": fl["SaleCondition"].value_counts(normalize=True),
                               "all test": df["SaleCondition"].value_counts(normalize=True)}).fillna(0).sort_values("all test", ascending=False)
    sale_share.to_csv(OUT / "error_sale_share.csv")
    month = df.groupby(df["date"].dt.to_period("M").astype(str))["resid_log"].mean()
    nonnormal_flag = float((fl["SaleCondition"] != "Normal").mean()) if len(fl) else 0.0
    nonnormal_all = float((df["SaleCondition"] != "Normal").mean())
    write_json({"sd": sd, "n_test": len(df), "n_flag": len(fl), "n_cases": N_CASES, "expected_flag": 0.0455 * len(df),
                "n_under": int((fl["Error"] > 0).sum()), "n_over": int((fl["Error"] < 0).sum()),
                "kinds": kinds.to_dict(), "main_kind": kinds.index[0] if len(kinds) else "not explained by available features",
                "fixes": FIXES, "nonnormal_flag": nonnormal_flag, "nonnormal_all": nonnormal_all,
                "mean_resid": float(df["resid_log"].mean()), "month_resid": month.to_dict(),
                "usd_band": [float(np.expm1(2 * sd) * 100), float(-np.expm1(-2 * sd) * 100)]}, "error_summary.json")

    # ---------------- F11 residuals with flagged points
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, x, xl in [(axes[0], "date", "sale date"), (axes[1], "GrLivArea", "above-ground living area (sq ft)")]:
        ax.scatter(df.loc[~df["flag"], x], df.loc[~df["flag"], "resid_log"], s=10, color=GRAY, alpha=0.6, lw=0)
        ax.axhline(0, color=INK, lw=1)
        for s in (-2 * sd, 2 * sd):
            ax.axhline(s, color=MUTED, lw=1, ls="--")
        ax.scatter(fl[x], fl["resid_log"], s=80, facecolor="none", edgecolor=RED, lw=1.6, zorder=3)
        for _, r in fl.head(N_CASES).iterrows():
            ax.annotate(r["#"], (r[x], r["resid_log"]), xytext=(0, 8 if r["resid_log"] > 0 else -13), textcoords="offset points",
                        ha="center", fontsize=7, color=RED, fontweight="bold")
        ax.set(xlabel=xl, ylabel="residual y - y_hat (log price)")
    axes[0].text(0.99, 0.02, f"dashed: +/-2 SD = +/-{2 * sd:.3f} (about -{-np.expm1(-2 * sd) * 100:.0f}% / +{np.expm1(2 * sd) * 100:.0f}% in price)",
                 transform=axes[0].transAxes, ha="right", fontsize=8, color=MUTED)
    fig.suptitle(f"F11. Test residuals; {len(fl)} of {len(df)} houses outside +/-2 SD (circled, numbered by |z|)", fontsize=11)
    save(fig, "F11_error_points.png")

    # ---------------- F12 how flagged houses differ
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    xs = np.arange(len(sale_share))
    axes[0].bar(xs - 0.2, sale_share["all test"] * 100, 0.38, color=BLUE, label="all test houses")
    axes[0].bar(xs + 0.2, sale_share["flagged"] * 100, 0.38, color=ORANGE, label="flagged (|z| > 2)")
    axes[0].set_xticks(xs, sale_share.index)
    axes[0].legend(frameon=False)
    axes[0].set(title="Sale condition mix", ylabel="% of houses")
    c = cmp.sort_values("SMD")
    axes[1].axvspan(-0.1, 0.1, color="#e8e7e3", lw=0)
    axes[1].scatter(c["SMD"], range(len(c)), s=40, color=RED, zorder=3)
    axes[1].axvline(0, color=INK, lw=1)
    axes[1].set_yticks(range(len(c)), c["feature"])
    axes[1].set(title="Flagged vs. other houses: standardised mean difference", xlabel="SMD (flagged - rest); shaded = negligible")
    fig.suptitle("F12. What the flagged houses have in common", fontsize=11)
    save(fig, "F12_error_patterns.png")
    # ---------------- F13 feature scatter: where the flagged houses sit (train grey, test blue, flagged red with predicted -> actual)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, f in zip(axes.ravel(), ["GrLivArea", "OverallQual", "YearBuilt", "TotalBsmtSF", "LotArea", "OverallCond"]):
        ax.scatter(rtr[f], np.log1p(rtr["SalePrice"]), s=6, color=GRAY, alpha=0.35, lw=0, label="train")
        ax.scatter(df.loc[~df["flag"], f], df.loc[~df["flag"], "actual_log"], s=8, color=BLUE, alpha=0.5, lw=0, label="test")
        top, other = fl.head(N_CASES), fl.iloc[N_CASES:]
        ax.scatter(other[f], np.log1p(other["Actual"]), s=14, facecolor="none", edgecolor=RED, lw=0.8, label="other flagged houses")
        for _, r in top.iterrows():
            ax.plot([r[f], r[f]], [np.log1p(r["Pred"]), np.log1p(r["Actual"])], color=RED, lw=1.2)
        ax.scatter(top[f], np.log1p(top["Pred"]), s=34, facecolor=SURFACE, edgecolor=RED, lw=1.3, zorder=3, label="case: predicted")
        ax.scatter(top[f], np.log1p(top["Actual"]), s=34, color=RED, zorder=4, label="case: actual")
        for _, r in top.iterrows():
            ax.annotate(r["#"], (r[f], np.log1p(r["Actual"])), xytext=(4, 2), textcoords="offset points", fontsize=8, color=RED, fontweight="bold")
        if f == "LotArea":
            ax.set_xscale("log")
        ax.set(xlabel=f + (" (log scale)" if f == "LotArea" else ""), ylabel="log price")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle(f"F13. The {N_CASES} largest errors on key features: hollow = predicted, filled = actual, red line = error", fontsize=11)
    save(fig, "F13_error_feature_scatter.png")

    # ---------------- F14 percentile profile of each flagged house (relative to training houses)
    M = prof.head(N_CASES)[KEY + ["SalePrice"]].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 0.5 * N_CASES + 1.8))
    im = ax.imshow(M, cmap=PCT, vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7, color=INK)
    ax.set_xticks(range(M.shape[1]), KEY + ["SalePrice"], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(M)), [f"{h} Id {i} ({sc})" for h, i, sc in zip(fl["#"], fl["Id"], fl["SaleCondition"])][:N_CASES], fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="percentile among training houses")
    ax.set_title(f"F14. Percentile profile of the {N_CASES} largest errors (orange = low, blue = high vs. training houses)")
    save(fig, "F14_error_profiles.png")
    print(f"{len(fl)} of {len(df)} test houses outside +/-2 SD ({2 * sd:.3f} log units)")


if __name__ == "__main__":
    main()
