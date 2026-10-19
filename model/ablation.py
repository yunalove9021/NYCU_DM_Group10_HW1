"""Step 3: feature ablation. Same model family, same time-series CV, same tuning procedure; only the feature set changes.

  A  all original features              B  all original + engineered
  C  top-k of original features (|r| ranked inside each training fold)
  D  top-k of original + engineered (ranked inside each training fold)
  E  greedy-selected original features  F  greedy-selected original + engineered (models/greedy_select_feature.xlsx)

Outputs: results/ablation.csv, results/ablation_pairwise.csv, results/ablation_slices.csv, figures F8-F10.
"""
import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from statsmodels.stats.multitest import multipletests

from common import (INK, MUTED, N_BOOT, N_SPLITS, NAVY, OUT, RED, SEED, SURFACE, TOPK, fit_config, from_json_key,
                    load_data, pool_parts, predict, read_json, run_cv, save, usd, write_json)

SET_COLORS = {"A": "#8d99a6", "B": "#33414f", "C": "#e34948", "D": "#eb6834", "E": "#1baf7a", "F": "#4a3aa7"}
SET_NAMES = {"A": "All original", "B": "All original + engineered", "C": "Top-k original",
             "D": "Top-k original + engineered", "E": "Greedy original", "F": "Greedy original + engineered"}


def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    D, cfg = load_data(), read_json("best_config.json")
    fam, drop = cfg["family"], bool(cfg["setting"].get("drop_outliers"))
    orig, allf = D["original"], D["feats"]
    greedy_orig = [f for f in D["greedy"] if f in set(orig)]
    plan = {
        "A": (orig, [{"kind": "all", "drop_outliers": drop}]),
        "B": (allf, [{"kind": "all", "drop_outliers": drop}]),
        "C": (orig, [{"kind": "topk", "k": k, "drop_outliers": drop} for k in TOPK if k < len(orig)]),
        "D": (allf, [{"kind": "topk", "k": k, "drop_outliers": drop} for k in TOPK if k < len(allf)]),
        "E": (greedy_orig, [{"kind": "all", "drop_outliers": drop}]),
        "F": (D["greedy"], [{"kind": "all", "drop_outliers": drop}]),
    }
    act = np.expm1(D["yte"])
    sets, kcurves = {}, {}
    for name, (pool, setts) in plan.items():
        r = run_cv(D, pool, setts, [fam])
        top = r["agg"].iloc[0]
        sid, pid = int(top["sid"]), int(top["pid"])
        pi, ba, du, _ = pool_parts(D, pool)
        m = fit_config(D["Xtr"], D["ytr"], D["outlier_tr"], setts[sid], r["keys"][pid], pi, ba, du, D["log"])
        p = predict(m, D["Xte"])
        sets[name] = {"pool": len(pool), "n_feats": len(m["cols"]), "params": top["params"], "k": setts[sid].get("k", np.nan),
                      "fold_mae": r["R"][sid, :, pid, 1], "fold_rmse": r["R"][sid, :, pid, 0], "pred": p,
                      "abs_err": np.abs(act - usd(p))}
        if setts[0]["kind"] == "topk":
            kcurves[name] = r["agg"].groupby("k")[["cv_mae_usd", "cv_mae_usd_sd"]].min()
        print(f"Set {name}: {len(m['cols'])} features, CV MAE {sets[name]['fold_mae'].mean():,.0f}")

    tab = pd.DataFrame([{"set": k, "definition": SET_NAMES[k], "pool": v["pool"], "n_feats": v["n_feats"], "k": v["k"],
                         "params": v["params"], "cv_mae_usd": v["fold_mae"].mean(), "cv_mae_usd_sd": v["fold_mae"].std(ddof=1),
                         "cv_rmse": v["fold_rmse"].mean(), "cv_rmse_sd": v["fold_rmse"].std(ddof=1),
                         "test_mae_usd": v["abs_err"].mean(), "test_rmse": np.sqrt(np.mean((v["pred"] - D["yte"]) ** 2))}
                        for k, v in sets.items()])
    tab.to_csv(OUT / "ablation.csv", index=False)

    # ---------------- pairwise significance (Holm over the 15 pairs, per test)
    rng = np.random.default_rng(SEED)
    bs = [rng.integers(0, len(act), len(act)) for _ in range(N_BOOT)]
    ratio = np.mean([len(va) / len(tr) for tr, va in D["splits"]])
    rows = []
    for a, b in itertools.combinations(sets, 2):
        d_cv = sets[a]["fold_mae"] - sets[b]["fold_mae"]
        t = d_cv.mean() / np.sqrt((1 / N_SPLITS + ratio) * d_cv.var(ddof=1)) if d_cv.var(ddof=1) > 0 else 0.0
        d = sets[a]["abs_err"] - sets[b]["abs_err"]
        boot = np.array([d[i].mean() for i in bs])
        rows.append({"comparison": f"{a} vs {b}", "cv_diff_usd": d_cv.mean(), "p_cv_corrected_t": 2 * stats.t.sf(abs(t), N_SPLITS - 1),
                     "test_diff_usd": d.mean(), "test_ci_lo_usd": np.percentile(boot, 2.5), "test_ci_hi_usd": np.percentile(boot, 97.5),
                     "p_test_bootstrap": min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())),
                     "p_test_wilcoxon": stats.wilcoxon(sets[a]["abs_err"], sets[b]["abs_err"]).pvalue if np.any(d != 0) else 1.0})
    pw = pd.DataFrame(rows)
    for c in ["p_cv_corrected_t", "p_test_bootstrap", "p_test_wilcoxon"]:
        pw[c + "_holm"] = multipletests(pw[c], method="holm")[1]
    pw.to_csv(OUT / "ablation_pairwise.csv", index=False)
    write_json({"cv_ratio": ratio, "family": fam, "drop_outliers": drop, "n_greedy": len(D["greedy"]),
                "n_greedy_orig": len(greedy_orig), "n_orig": len(orig), "n_all": len(allf)}, "ablation_summary.json")

    # ---------------- slices of the test set
    raw = D["raw_te"]
    age = raw["YrSold"] - raw["YearBuilt"]
    q = np.quantile(D["raw_tr"]["GrLivArea"], [1 / 3, 2 / 3])
    slices = {"Age: new (<= 10 y)": age <= 10, "Age: 11-50 y": (age > 10) & (age <= 50), "Age: old (> 50 y)": age > 50,
              "Size: small": raw["GrLivArea"] <= q[0], "Size: medium": (raw["GrLivArea"] > q[0]) & (raw["GrLivArea"] <= q[1]),
              "Size: large": raw["GrLivArea"] > q[1], "Sale: Normal": raw["SaleCondition"] == "Normal",
              "Sale: non-Normal": raw["SaleCondition"] != "Normal"}
    sl = pd.DataFrame([{"slice": s, "n": int(mk.sum()), **{f"MAE_usd {k}": v["abs_err"][mk.to_numpy()].mean() for k, v in sets.items()}}
                       for s, mk in slices.items()])
    sl.to_csv(OUT / "ablation_slices.csv", index=False)

    # ---------------- F8 overall + by age slice
    names = list(sets)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8), gridspec_kw={"width_ratios": [1, 1.5]})
    t = tab.set_index("set")
    for i, k in enumerate(names):
        axes[0].bar(i, t.loc[k, "cv_mae_usd"], 0.65, yerr=t.loc[k, "cv_mae_usd_sd"], color=SET_COLORS[k],
                    error_kw={"ecolor": INK, "lw": 1.1, "capsize": 4})
        axes[0].scatter(i, t.loc[k, "test_mae_usd"], marker="D", s=40, color=SURFACE, edgecolor=INK, zorder=3,
                        label="test MAE" if i == 0 else None)
    nl = "\n"
    ticks = [k + nl + SET_NAMES[k].replace(" + ", " +" + nl).replace(" ", nl) + nl + f"({int(t.loc[k, 'n_feats'])} feats)" for k in names]
    axes[0].set_xticks(range(len(names)), ticks, fontsize=12)
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].set(title="(a) Time-series 5-fold CV MAE (bars, error bar = fold SD)", ylabel="MAE ($)")
    age_rows = sl[sl["slice"].str.startswith("Age")]
    xs = np.arange(len(age_rows))
    w = 0.13
    for j, k in enumerate(names):
        axes[1].bar(xs + (j - 2.5) * w, age_rows[f"MAE_usd {k}"], w * 0.95, color=SET_COLORS[k], label=f"Set {k}")
    axes[1].set_xticks(xs, [f"{s.replace('Age: ', '')}\n(n={n})" for s, n in zip(age_rows["slice"], age_rows["n"])])
    axes[1].legend(frameon=False, ncol=6, fontsize=14, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    axes[1].set(title="(b) Time-split test set, by house age at sale", ylabel="test MAE ($)")
    fig.suptitle(f"F8. Feature ablation ({fam}; same split, folds and tuning; only the feature set changes)", fontsize=19)
    save(fig, "F8_ablation.png")

    # ---------------- F9 choosing k
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for k, c in kcurves.items():
        ax.errorbar(c.index, c["cv_mae_usd"], yerr=c["cv_mae_usd_sd"], fmt="o-", ms=5, color=SET_COLORS[k], ecolor=SET_COLORS[k],
                    alpha=0.9, label=f"Set {k} (chosen k = {int(sets[k]['k'])})", capsize=0)
        ax.axvline(sets[k]["k"], color=SET_COLORS[k], ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=14, loc="upper right")
    ax.set(title="F9. Top-k sets: CV MAE vs. k (ranking recomputed in every training fold)", xlabel="k (log scale)", ylabel="CV MAE ($)")
    save(fig, "F9_topk_curves.png")

    # ---------------- F10 pairwise significance
    sg = pw.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, r in sg.iterrows():
        b, wx = r["p_test_bootstrap_holm"] < 0.05, r["p_test_wilcoxon_holm"] < 0.05
        col = RED if (b or wx) else NAVY
        ax.hlines(i, r["test_ci_lo_usd"], r["test_ci_hi_usd"], color=col, lw=1.8)
        ax.scatter(r["test_diff_usd"], i, s=50, zorder=3, color=RED if b else SURFACE, edgecolor=col, lw=1.6)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(range(len(sg)), sg["comparison"])
    ax.set(title="F10. Pairwise test MAE difference (first set - second), 95% paired-bootstrap CI",
           xlabel="$ per house; > 0 = the second set is better")
    fig.text(0.01, 0.01, "red = Holm-adjusted p < 0.05 (filled: paired bootstrap; hollow red: Wilcoxon signed-rank only); dark = not significant",
             fontsize=14, color=MUTED)
    save(fig, "F10_ablation_pairwise.png")


if __name__ == "__main__":
    main()
