"""Step 0: exploratory feature analysis on the training set (test only for train -> test drift).

Outputs: results/eda_correlations.csv, results/eda_drift.csv, results/eda_summary.json, figures E1-E6.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from common import BLUE, GRAY, INK, MUTED, ORANGE, OUT, RED, load_data, save, write_json

DIVERGING = LinearSegmentedColormap.from_list("div", [ORANGE, "#f0efec", BLUE])


def main():
    D = load_data()
    X, y, feats = D["Xtr"], D["ytr"], D["feats"]
    eng = set(D["engineered"])
    sd = X.std(0)
    r = np.array([np.corrcoef(X[:, j], y)[0, 1] if sd[j] > 0 else 0.0 for j in range(len(feats))])
    kind = ["engineered" if f in eng else "dummy" if f in D["dummy_set"] else "original" for f in feats]
    corr = pd.DataFrame({"feature": feats, "kind": kind, "r": r, "abs_r": np.abs(r)}).sort_values("abs_r", ascending=False)
    corr["effect_size"] = pd.cut(corr["abs_r"], [-1, 0.1, 0.3, 0.5, 1.01], labels=["negligible", "small", "medium", "large"], right=False)
    corr.to_csv(OUT / "eda_correlations.csv", index=False)

    # train -> test drift (time split), non-dummy features
    nd = [j for j, f in enumerate(feats) if f not in D["dummy_set"] and sd[j] > 0]
    Xte = D["Xte"]
    drift = pd.DataFrame({"feature": [feats[j] for j in nd],
                          "SMD": [(Xte[:, j].mean() - X[:, j].mean()) / np.sqrt((Xte[:, j].var() + X[:, j].var()) / 2) for j in nd],
                          "KS_p": [stats.ks_2samp(X[:, j], Xte[:, j]).pvalue for j in nd]})
    drift = drift.reindex(drift["SMD"].abs().sort_values(ascending=False).index)
    drift.to_csv(OUT / "eda_drift.csv", index=False)

    rtr = D["raw_tr"]
    price = rtr["SalePrice"]
    write_json({"n_train": len(y), "n_test": len(D["yte"]), "skew_raw": float(stats.skew(price)), "skew_log": float(stats.skew(y)),
                "price_median_train": float(price.median()), "price_median_test": float(D["raw_te"]["SalePrice"].median()),
                "effect_counts": corr["effect_size"].value_counts().to_dict(),
                "n_drift": int((drift["SMD"].abs() > 0.1).sum()), "n_nondummy": len(nd)}, "eda_summary.json")

    # E1 target distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(price / 1e3, 40, color=BLUE)
    axes[0].set(title=f"Raw SalePrice (train), skew = {stats.skew(price):.2f}", xlabel="$k", ylabel="houses")
    bins = np.linspace(min(y.min(), D["yte"].min()), max(y.max(), D["yte"].max()), 35)
    axes[1].hist(y, bins, density=True, histtype="step", lw=2, color=BLUE, label=f"train ({len(y)})")
    axes[1].hist(D["yte"], bins, density=True, histtype="step", lw=2, color=ORANGE, label=f"test ({len(D['yte'])})")
    axes[1].legend(frameon=False)
    axes[1].set(title=f"log1p(SalePrice), skew = {stats.skew(y):.2f}", xlabel="log price", ylabel="density")
    fig.suptitle("E1. Target: the log transform removes the right skew; train and test have similar distributions", fontsize=11)
    save(fig, "E1_target.png")

    # E2 top correlations
    top = corr.head(25)[::-1]
    col = {"original": BLUE, "engineered": RED, "dummy": GRAY}
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top["feature"], top["r"], color=[col[k] for k in top["kind"]])
    for t, lab in [(0.1, "small"), (0.3, "medium"), (0.5, "large")]:
        for s in (-1, 1):
            ax.axvline(s * t, color=MUTED, lw=0.8, ls=":")
        ax.text(t, len(top) - 0.3, lab, fontsize=7, color=MUTED, ha="center")
    ax.axvline(0, color=INK, lw=1)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=k) for k, c in col.items()], frameon=False, loc="lower right")
    ax.set(title="E2. Top 25 features by |Pearson r| with log price (train)", xlabel="r (dotted: Cohen's 0.1 / 0.3 / 0.5)")
    save(fig, "E2_top_correlations.png")

    # E3 scatter of the 6 strongest non-dummy features
    top6 = [f for f in corr["feature"] if f not in D["dummy_set"]][:6]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, f in zip(axes.ravel(), top6):
        x = X[:, D["col_index"][f]]
        ax.scatter(x, y, s=8, color=BLUE, alpha=0.4, lw=0)
        b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, np.polyval(b, xs), color=RED, lw=1.5)
        ax.set(title=f"{f} (r = {corr.set_index('feature').at[f, 'r']:+.2f})", xlabel=f"{f} (processed scale)", ylabel="log price")
    fig.suptitle("E3. Strongest non-dummy features vs. log price (train), with linear fit", fontsize=11)
    save(fig, "E3_top_scatter.png")

    # E4 collinearity among top 15 non-dummy features
    top15 = [f for f in corr["feature"] if f not in D["dummy_set"]][:15]
    C = np.corrcoef(X[:, [D["col_index"][f] for f in top15]], rowvar=False)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(C, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(15), top15, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(15), top15, fontsize=8)
    for i in range(15):
        for j in range(15):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=6, color=INK)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("E4. Correlation among the 15 strongest features (collinear blocks)")
    save(fig, "E4_feature_correlation.png")

    # E5 drift
    dtop = drift.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.axvspan(-0.1, 0.1, color="#e8e7e3", lw=0)
    ax.scatter(dtop["SMD"], dtop["feature"], color=[RED if p < 0.05 else BLUE for p in dtop["KS_p"]], s=35, zorder=3)
    ax.axvline(0, color=INK, lw=1)
    ax.set(title="E5. Train -> test drift (time split): 15 largest standardised mean differences",
           xlabel="SMD (test - train); shaded = negligible; red = KS p < 0.05")
    save(fig, "E5_drift.png")

    # E6 neighbourhood (raw labels)
    nb = rtr.groupby("Neighborhood")["SalePrice"].median().sort_values()
    order = list(nb.index)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.boxplot([np.log1p(rtr.loc[rtr["Neighborhood"] == n, "SalePrice"]) for n in order], labels=order, showfliers=False,
               medianprops={"color": RED}, boxprops={"color": BLUE}, whiskerprops={"color": MUTED}, capprops={"color": MUTED})
    ax.tick_params(axis="x", labelrotation=60, labelsize=8)
    ax.set(title="E6. Log price by neighbourhood (train), sorted by median", ylabel="log price")
    save(fig, "E6_neighbourhood.png")
    print(f"EDA: {int((corr['abs_r'] >= 0.5).sum())} features with |r| >= 0.5; {int((drift['SMD'].abs() > 0.1).sum())} drifted features")


if __name__ == "__main__":
    main()
