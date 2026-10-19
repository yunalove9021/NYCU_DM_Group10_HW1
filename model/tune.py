"""Step 1: correlation threshold x one-hot handling x outlier rule x model hyperparameters, time-series CV on train.

Outputs: results/tuning_results.csv, results/best_config.json, results/tuning_summary.json, figures F1-F5.
"""
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from common import (BLUE, CAT_MODES, COLORS, FAMILIES, GRAY, INK, MUTED, ORANGE, OUT, THRESHOLDS, load_data,
                    run_cv, save, to_jsonable_key, write_json)


def line_by_family(ax, table, xlabel, end_labels=False):
    for fam in FAMILIES:
        if fam in table:
            s = table[fam].dropna()
            ax.plot(s.index, s.values, "o-", ms=6, color=COLORS[fam], label=fam)
    if end_labels:  # direct labels, dodged so close lines do not overlap
        ends = table[[f for f in FAMILIES if f in table]].iloc[-1].sort_values()
        lo, hi = ax.get_ylim()
        gap, last = (hi - lo) * 0.08, -np.inf
        span = table.index[-1] - table.index[0]
        for fam, v in ends.items():
            last = max(v, last + gap)
            ax.annotate(fam, (table.index[-1], v), xytext=(table.index[-1] + span * 0.03, last), textcoords="data",
                        fontsize=14, color=INK, va="center")
        ax.set_xlim(right=table.index[-1] + span * 0.18)
    ax.set_xlabel(xlabel)


def main():
    t0 = time.time()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    D = load_data()
    settings = [{"kind": "thr", "threshold": t, "cat_mode": c, "drop_outliers": d}
                for t in THRESHOLDS for c in CAT_MODES for d in (False, True) if not (t == 0.0 and c == "filtered")]
    cv = run_cv(D, D["feats"], settings, FAMILIES)
    res = cv["agg"]
    res.drop(columns=["sid", "pid"]).to_csv(OUT / "tuning_results.csv", index=False)
    print(f"tuning: {len(settings)} selection settings x {len(cv['keys'])} hyperparameter values in {time.time() - t0:.0f}s")

    # baselines on the same folds
    mean_rmse = float(np.mean([np.sqrt(np.mean((D["ytr"][va] - D["ytr"][tr].mean()) ** 2)) for tr, va in D["splits"]]))
    oq = run_cv(D, ["OverallQual"], [{"kind": "all"}], ["OLS"])["agg"].iloc[0]

    best = res.iloc[0]
    best_fam = res.loc[res.groupby("model")["cv_rmse"].idxmin()].set_index("model").loc[FAMILIES]
    write_json({"setting": settings[int(best["sid"])], "key": to_jsonable_key(cv["keys"][int(best["pid"])]),
                "family": best["model"], "params": best["params"], "cv_rmse": best["cv_rmse"],
                "cv_rmse_sd": best["cv_rmse_sd"], "cv_mae_usd": best["cv_mae_usd"]}, "best_config.json")
    best_fam.reset_index().drop(columns=["sid", "pid"]).to_csv(OUT / "tuning_best_per_family.csv", index=False)
    thr_tab = res.groupby(["threshold", "model"])["cv_rmse"].min().unstack()[FAMILIES]
    thr_tab.to_csv(OUT / "tuning_threshold_table.csv")
    g = D["grids"]
    write_json({"n_settings": len(settings), "n_keys": len(cv["keys"]), "baseline_mean_rmse": mean_rmse,
                "baseline_oq_rmse": oq["cv_rmse"], "baseline_oq_mae": oq["cv_mae_usd"],
                "ridge_range": [g["ridge"][0], g["ridge"][-1]], "rank": g["rank"],
                "enet_alpha_max": {str(k): v[0] for k, v in g["enet"].items()},
                "huber_range": [g["huber"][0], g["huber"][-1]],
                "fold_sizes": [[len(tr), len(va)] for tr, va in D["splits"]],
                "fold_periods": [f"{D['raw_tr']['date'].iloc[va[0]]:%Y-%m}..{D['raw_tr']['date'].iloc[va[-1]]:%Y-%m}" for _, va in D["splits"]],
                "seconds": time.time() - t0}, "tuning_summary.json")

    # F1 threshold curve
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(oq["cv_rmse"], color=GRAY, ls="--", lw=1)
    ax.text(THRESHOLDS[0], oq["cv_rmse"], " baseline: OLS on OverallQual only", color=MUTED, va="bottom", fontsize=14)
    line_by_family(ax, thr_tab.clip(upper=oq["cv_rmse"]), "threshold |r| >= t (Cohen: 0.1 small, 0.3 medium, 0.5 large)", end_labels=True)
    ax.set_xticks(THRESHOLDS)
    ax.set(title="F1. Best time-series CV RMSE (log price) at each threshold", ylabel="CV RMSE (lower is better)")
    save(fig, "F1_threshold_vs_rmse.png")

    # F2 features kept
    nf = res[res["model"] == "OLS"].groupby(["threshold", "cat_mode"])["n_feats"].mean().unstack()
    fig, ax = plt.subplots(figsize=(7, 4))
    for c, col in zip(CAT_MODES, [BLUE, ORANGE, "#1baf7a"]):
        s = nf[c].dropna()
        ax.plot(s.index, s.values, "o-", ms=6, color=col, label=f"dummies: {c}")
    ax.set_yscale("log")
    ax.set_xticks(THRESHOLDS)
    ax.legend(frameon=False)
    ax.set(title="F2. Features kept vs. threshold (mean over folds)", xlabel="correlation threshold", ylabel="features (log scale)")
    save(fig, "F2_threshold_vs_n_features.png")

    # F3 dummies / outliers
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (col, title) in zip(axes, [("cat_mode", "one-hot dummies"), ("drop_outliers", "drop GrLivArea > 4000 (training folds)")]):
        t = res.groupby([col, "model"])["cv_rmse"].min().unstack()[FAMILIES]
        t.index = t.index.astype(str)
        t = t.clip(upper=oq["cv_rmse"])
        line_by_family(ax, t, title)
    axes[0].set_ylabel("best CV RMSE")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("F3. Effect of the other selection settings (best over everything else)", fontsize=19)
    save(fig, "F3_selection_settings.png")

    # F4 regularisation paths at each family's best selection setting
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, f in zip(axes.flat, ["Ridge", "Lasso", "Huber", "ElasticNet"]):
        b = best_fam.loc[f]
        sub = res[(res["model"] == f) & (res["sid"] == b["sid"])]
        groups = [(e, sub[sub["extra"] == e]) for e in sorted(sub["extra"].dropna().unique())] if f in ("ElasticNet", "Huber") else [(None, sub)]
        palette = [BLUE, ORANGE, "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
        for i, (e, s) in enumerate(groups[:6]):
            s = s.sort_values("alpha")
            ax.plot(s["alpha"], s["cv_rmse"], "-", color=palette[i] if e is not None else COLORS[f],
                    label=(f"l1_ratio={e:g}" if f == "ElasticNet" else f"epsilon={e:g}") if e is not None else None)
        if len(groups) > 1:
            ax.legend(frameon=False, fontsize=14, **({"loc": "upper left", "bbox_to_anchor": (1.02, 1)} if f == "ElasticNet" else {"loc": "upper center"}))
        ax.axvline(b["alpha"], color=MUTED, lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_ylim(top=min(ax.get_ylim()[1], oq["cv_rmse"]))
        ax.set(title=f"{f} (t={b['threshold']:g}, dummies={b['cat_mode']})", xlabel="alpha (dotted = chosen)")
    for ax in axes[:, 0]:
        ax.set_ylabel("CV RMSE (log price)")
    fig.suptitle("F4. Regularisation strength over the data-driven ranges", fontsize=19)
    save(fig, "F4_alpha_curves.png")

    # F5 best per family
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(len(FAMILIES))
    ax.bar(xs, best_fam["cv_rmse"], 0.6, yerr=best_fam["cv_rmse_sd"], color=[COLORS[f] for f in FAMILIES], error_kw={"ecolor": INK, "lw": 1, "capsize": 3})
    for i, f in enumerate(FAMILIES):
        ax.text(i, best_fam.loc[f, "cv_rmse"] / 2, f"{best_fam.loc[f, 'cv_rmse']:.3f}", ha="center", color="white", fontsize=16)
    ax.axhline(oq["cv_rmse"], color=GRAY, ls="--", lw=1)
    ax.text(len(FAMILIES) - 0.6, oq["cv_rmse"], "OverallQual-only baseline", color=MUTED, ha="right", va="bottom", fontsize=14)
    ax.set_xticks(xs, FAMILIES)
    ax.set(title="F5. Best configuration per model family (error bar = SD over 5 time-series folds)", ylabel="CV RMSE (log price)")
    save(fig, "F5_best_per_family.png")
    print(f"best: {best['model']} {best['params']} threshold={best['threshold']:g} dummies={best['cat_mode']} "
          f"drop_outliers={best['drop_outliers']} cv_rmse={best['cv_rmse']:.4f} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
