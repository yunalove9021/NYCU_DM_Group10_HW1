"""Step 3: metrics, baselines and main results (§6 / §7.1), plus the pipeline figure (§5.1).

Every method is scored under the same protocol: the same 5 time-series folds inside the training set,
then one fit on the full training set evaluated on the chronological test set.
Baselines follow model/baseline2.py (trivial / simple / strong tiers), plus a persistence baseline.

Outputs: results/main_results.csv, results/baselines_strong.csv, results/main_summary.json,
         results/test_predictions_baselines.csv, figures F15 (pipeline) and F16 (main results).
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from common import (BLUE, GRAY, INK, MUTED, N_BOOT, N_JOBS, N_SPLITS, NAVY, OUT, RED, SEED, SURFACE, fit_config,
                    from_json_key, load_data, pool_parts, predict, read_json, save, usd, write_json)

DECISION_SHARE = 0.10   # decision threshold: MAE within 10% of the median training price
REENCODED = {"HouseAge", "RemodAge", "GarageAge", "sin_MoSold", "cos_MoSold"}  # as in baseline2.py
STRONG = ([("RF", {"max_features": mf, "min_samples_leaf": leaf}) for mf in [0.3, 0.5] for leaf in [1, 3]] +
          [("HGB", {"lr": lr, "leaves": lv, "l2": l2}) for lr in [0.03, 0.06] for lv in [8, 15] for l2 in [0.0, 1.0]])


def metrics(p, y):
    a, f = np.expm1(y), usd(p)
    return {"mae": np.mean(np.abs(a - f)), "rmse": np.sqrt(np.mean((p - y) ** 2)),
            "r2": 1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2),
            "mape": np.mean(np.abs(a - f) / a) * 100, "ppe10": np.mean(np.abs(a - f) / a <= 0.10) * 100}


def tree_model(kind, p):
    if kind == "RF":
        return RandomForestRegressor(n_estimators=1000, max_features=p["max_features"], min_samples_leaf=p["min_samples_leaf"],
                                     n_jobs=1, random_state=SEED)
    return HistGradientBoostingRegressor(max_iter=1000, learning_rate=p["lr"], max_leaf_nodes=p["leaves"],
                                         l2_regularization=p["l2"], random_state=SEED)


def tree_fold(kind, p, X, y, tr, te):
    warnings.filterwarnings("ignore")
    return tree_model(kind, p).fit(X[tr], y[tr]).predict(X[te])


def main():
    warnings.filterwarnings("ignore")
    D, cfg = load_data(), read_json("best_config.json")
    setting, key = cfg["setting"], from_json_key(cfg["key"])
    Xtr, Xte, ytr, yte = D["Xtr"], D["Xte"], D["ytr"], D["yte"]
    dates = D["raw_tr"]["date"]
    splits = D["splits"]
    all_tr = np.arange(len(ytr))

    # ---------------- method definitions: fn(train_rows, X_eval) -> log-price predictions
    def mean_fn(tr, Xe):
        return np.full(len(Xe), ytr[tr].mean())

    def persistence_fn(tr, Xe):  # median price of the last 3 months of the training part
        last = dates.iloc[tr].max()
        recent = tr[(dates.iloc[tr] > last - pd.DateOffset(months=3)).to_numpy()]
        return np.full(len(Xe), np.median(ytr[recent]))

    raw_cols = {"OverallQual": D["col_index"]["OverallQual"]}
    gla_raw = {"tr": np.log(D["raw_tr"]["GrLivArea"].to_numpy(float)), "te": np.log(D["raw_te"]["GrLivArea"].to_numpy(float))}

    pidx, base, dummy, _ = pool_parts(D, D["feats"])

    def ours_fn(tr, Xe):
        m = fit_config(Xtr[tr], ytr[tr], D["outlier_tr"][tr], setting, key, pidx, base, dummy, D["log"])
        return predict(m, Xe)

    # ---------------- cross-validation + test for the non-tree methods
    def single_feature(col, tr, rows, test=False):
        x_tr = Xtr[tr, raw_cols[col]] if col == "OverallQual" else gla_raw["tr"][tr]
        b = np.polyfit(x_tr, ytr[tr], 1)
        x = (Xte[:, raw_cols[col]] if col == "OverallQual" else gla_raw["te"]) if test else \
            (Xtr[rows, raw_cols[col]] if col == "OverallQual" else gla_raw["tr"][rows])
        return np.polyval(b, x)

    methods = {
        "Training mean": (lambda tr, va: mean_fn(tr, Xtr[va]), lambda: mean_fn(all_tr, Xte)),
        "Persistence (last 3 months)": (lambda tr, va: persistence_fn(tr, Xtr[va]), lambda: persistence_fn(all_tr, Xte)),
        "Single-feature OLS (OverallQual)": (lambda tr, va: single_feature("OverallQual", tr, va),
                                             lambda: single_feature("OverallQual", all_tr, None, test=True)),
        "Single-feature OLS (log GrLivArea)": (lambda tr, va: single_feature("GrLivArea", tr, va),
                                               lambda: single_feature("GrLivArea", all_tr, None, test=True)),
        f"Ours ({cfg['family']} + features)": (lambda tr, va: ours_fn(tr, Xtr[va]), lambda: ours_fn(all_tr, Xte)),
    }
    fold_rows, test_preds = [], {}
    for name, (cv_fn, test_fn) in methods.items():
        for k, (tr, va) in enumerate(splits):
            fold_rows.append({"method": name, "fold": k, **metrics(cv_fn(tr, va), ytr[va])})
        test_preds[name] = test_fn()

    # ---------------- strong baselines (raw inputs, small grid from baseline2.py), tuned by the same CV
    raw_names = set(D["raw_tr"].columns) - {"Id", "SalePrice", "date"}
    raw_cols_tree = [i for i, c in enumerate(D["feats"]) if c in raw_names or c in REENCODED or
                     any(c.startswith(r + "_") for r in raw_names)]
    Xr_tr, Xr_te = Xtr[:, raw_cols_tree], Xte[:, raw_cols_tree]
    jobs = [(i, k) for i in range(len(STRONG)) for k in range(N_SPLITS)]
    preds = Parallel(n_jobs=N_JOBS)(delayed(tree_fold)(STRONG[i][0], STRONG[i][1], Xr_tr, ytr, splits[k][0], splits[k][1])
                                    for i, k in jobs)
    strong = []
    for (i, k), p in zip(jobs, preds):
        strong.append({"cfg": i, "kind": STRONG[i][0], "params": str(STRONG[i][1]), "fold": k, **metrics(p, ytr[splits[k][1]])})
    strong = pd.DataFrame(strong)
    sagg = strong.groupby(["cfg", "kind", "params"]).agg(cv_rmse=("rmse", "mean"), cv_mae=("mae", "mean")).reset_index().sort_values("cv_rmse")
    sagg.to_csv(OUT / "baselines_strong.csv", index=False)
    best_strong = int(sagg.iloc[0]["cfg"])
    kind, params = STRONG[best_strong]
    strong_name = f"{'Random forest' if kind == 'RF' else 'Gradient boosting'} (raw inputs)"
    for _, r in strong[strong["cfg"] == best_strong].iterrows():
        fold_rows.append({"method": strong_name, "fold": r["fold"], **{m: r[m] for m in ["mae", "rmse", "r2", "mape", "ppe10"]}})
    test_preds[strong_name] = tree_model(kind, params).fit(Xr_tr, ytr).predict(Xr_te)

    # ---------------- table
    folds = pd.DataFrame(fold_rows)
    order = ["Training mean", "Persistence (last 3 months)", "Single-feature OLS (OverallQual)",
             "Single-feature OLS (log GrLivArea)", strong_name, f"Ours ({cfg['family']} + features)"]
    tier = dict(zip(order, ["trivial", "trivial", "simple", "simple", "strong", "ours"]))
    rows = []
    for mth in order:
        f = folds[folds["method"] == mth]
        t = metrics(test_preds[mth], yte)
        rows.append({"method": mth, "tier": tier[mth], "cv_mae": f["mae"].mean(), "cv_mae_sd": f["mae"].std(ddof=1),
                     "cv_rmse": f["rmse"].mean(), "cv_rmse_sd": f["rmse"].std(ddof=1), "cv_r2": f["r2"].mean(),
                     "test_mae": t["mae"], "test_rmse": t["rmse"], "test_r2": t["r2"], "test_mape": t["mape"], "test_ppe10": t["ppe10"]})
    main = pd.DataFrame(rows)
    med = float(np.median(np.expm1(ytr)))
    thr = DECISION_SHARE * med
    main["meets_threshold_test"] = main["test_mae"] <= thr
    main.to_csv(OUT / "main_results.csv", index=False)
    folds.to_csv(OUT / "main_results_folds.csv", index=False)
    pd.DataFrame({"Id": D["id_te"], **{m: usd(p) for m, p in test_preds.items()}}).to_csv(OUT / "test_predictions_baselines.csv", index=False)

    # ---------------- ours vs. the strongest baseline (lowest CV MAE among baselines)
    ours = order[-1]
    rival = main[main["tier"] != "ours"].sort_values("cv_mae").iloc[0]["method"]
    d_cv = (folds[folds["method"] == rival].sort_values("fold")["mae"].to_numpy()
            - folds[folds["method"] == ours].sort_values("fold")["mae"].to_numpy())
    ratio = np.mean([len(va) / len(tr) for tr, va in splits])
    t_cv = d_cv.mean() / np.sqrt((1 / N_SPLITS + ratio) * d_cv.var(ddof=1))
    p_cv = 2 * stats.t.sf(abs(t_cv), N_SPLITS - 1)
    act = np.expm1(yte)
    e_r, e_o = np.abs(act - usd(test_preds[rival])), np.abs(act - usd(test_preds[ours]))
    rng = np.random.default_rng(SEED)
    boot = np.array([(e_r[i] - e_o[i]).mean() for i in (rng.integers(0, len(act), len(act)) for _ in range(N_BOOT))])
    write_json({"median_price": med, "threshold_usd": thr, "threshold_share": DECISION_SHARE, "rival": rival, "ours": ours,
                "strong_name": strong_name, "strong_params": str(params), "n_strong_cfgs": len(STRONG),
                "cv_gap": float(d_cv.mean()), "cv_t": float(t_cv), "p_cv": float(p_cv),
                "test_gap": float((e_r - e_o).mean()), "test_ci": np.percentile(boot, [2.5, 97.5]).tolist(),
                "p_boot": float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))),
                "p_wilcoxon": float(stats.wilcoxon(e_r, e_o).pvalue)}, "main_summary.json")

    # ---------------- F16 main results
    fig, ax = plt.subplots(figsize=(10, 4.8))
    xs = np.arange(len(main))
    cols = [GRAY, GRAY, "#8d99a6", "#8d99a6", NAVY, RED]
    ax.bar(xs, main["cv_mae"] / 1e3, 0.62, yerr=main["cv_mae_sd"] / 1e3, color=cols, error_kw={"ecolor": INK, "lw": 1, "capsize": 3})
    ax.scatter(xs, main["test_mae"] / 1e3, marker="D", s=45, color=SURFACE, edgecolor=INK, zorder=3, label="test MAE")
    ax.axhline(thr / 1e3, color=RED, ls="--", lw=1.2)
    ax.text(len(main) - 0.5, thr / 1e3, f" decision threshold ${thr / 1e3:.1f}k", color=RED, va="bottom", ha="right", fontsize=8)
    ax.set_xticks(xs, [m.replace(" (", "\n(") for m in main["method"]], fontsize=8)
    ax.legend(frameon=False, loc="upper right")
    ax.set(title="F16. Main results: CV MAE (bars, ± fold SD) and test MAE (diamonds)", ylabel="MAE ($k)")
    save(fig, "F16_main_results.png")

    # ---------------- F15 pipeline figure
    nz = int(np.sum(fit_config(Xtr, ytr, D["outlier_tr"], setting, key, pidx, base, dummy, D["log"])["w"] != 0))
    stages = [
        ("Raw data", f"Dataset/train.csv\n1,460 sales, 79 columns\nsorted by sale date"),
        ("Preprocessing", "NA → 0, label fixes,\nordinal / one-hot coding,\ntime split 80 / 20"),
        ("Feature\nengineering", f"ages, sin/cos month,\naggregates, interactions,\nflags; log1p if skew > 0.75*"),
        ("Feature\nselection", f"|r| ≥ {setting['threshold']:g} on fold-train*\none-hot: {setting['cat_mode']}\n→ {nz} non-zero"),
        ("Linear model", f"{cfg['family']}\n{cfg['params'].replace(', ', chr(10))}\nstandardised*"),
        ("Prediction &\nevaluation", "log price → $\nMAE, RMSE, MAPE, R²\nTS-CV + test"),
    ]
    fig, ax = plt.subplots(figsize=(15, 3.6))
    ax.set_xlim(0, len(stages) * 2.5)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    for i, (title, desc) in enumerate(stages):
        x = i * 2.5 + 0.15
        hi = i in (2, 4)
        ax.add_patch(FancyBboxPatch((x, 1.75), 2.1, 1.1, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=RED if hi else NAVY, ec="none"))
        ax.text(x + 1.05, 2.3, title, ha="center", va="center", color="white", fontsize=10.5, fontweight="bold")
        ax.text(x + 1.05, 1.5, desc, ha="center", va="top", color=INK, fontsize=8)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + 2.38, 2.3), xytext=(x + 2.12, 2.3), arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.5})
    ax.text(0.15, 0.05, "* decided on the training rows of each fold (or of the full training set for the final model) and applied unchanged "
            "to validation / test rows", fontsize=8, color=MUTED)
    ax.set_title("F15. Pipeline: from raw data to a prediction (red = feature engineering and the linear model)", fontsize=11)
    save(fig, "F15_pipeline.png")
    print(main[["method", "cv_mae", "cv_rmse", "test_mae", "test_r2"]].round(4).to_string())
    print(f"threshold ${thr:,.0f}; ours vs {rival}: test gap {np.mean(e_r - e_o):+,.0f}")


if __name__ == "__main__":
    main()
