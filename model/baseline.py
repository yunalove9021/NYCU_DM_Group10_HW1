"""Correlation-threshold + model hyperparameter tuning for linear regression on Ames house prices.

Run from anywhere:  python model/threshold_tuning.py
Outputs (in model/): report.md, results_all.csv, results_folds.csv, submission.csv, figures/*.png

Design
- Target: log1p(SalePrice), as stored in process_dataset/one_hot_train.csv.
- The correlation threshold is a hyperparameter: inside every CV fold, |Pearson r| with the target is
  recomputed on the TRAINING part only, so feature selection never sees the validation fold.
- Scaling and outlier removal also happen inside the fold (training part only).
- Rows are sorted by sale date (YrSold, MoSold). Every config is scored with BOTH
  TimeSeriesSplit (expanding window, train on past -> validate on future) and shuffled KFold.
  A config whose selected features include time-related columns gets TimeSeriesSplit as its
  primary CV; ranking uses TimeSeriesSplit for all configs so they are comparable.
- Test labels are NOT available (Dataset/sample_submission.csv is a Kaggle placeholder),
  so all evaluation is cross-validation on train.csv.
"""
import itertools
import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
N_SPLITS = 5
SEED = 0

# ---------------------------------------------------------------- tunable parameters
THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]   # min |r| with target to keep a feature
CAT_MODES = ["none", "filtered", "all"]            # one-hot dummies: drop / same threshold / keep all
PRUNE = [1.0, 0.8]                                 # drop a feature if |r| with a stronger kept feature > this (1.0 = off)
DROP_OUTLIERS = [False, True]                      # drop GrLivArea>4000 & price<300k from training folds
MODEL_GRID = (
    [("OLS", {})]
    + [("Ridge", {"alpha": a}) for a in [0.1, 1, 10, 30, 100, 300, 1000]]
    + [("Lasso", {"alpha": a}) for a in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]]
    + [("ElasticNet", {"alpha": a, "l1_ratio": l}) for a in [1e-3, 3e-3, 1e-2, 3e-2] for l in [0.2, 0.5, 0.8]]
    + [("Huber", {"epsilon": e, "alpha": a}) for e in [1.35, 2.0] for a in [1e-4, 1.0, 10.0]]
)
TIME_FEATURES = {"YrSold", "sin_MoSold", "cos_MoSold", "HouseAge", "RemodAge", "GarageAge"}

FAMILIES = ["OLS", "Ridge", "Lasso", "ElasticNet", "Huber"]
COLORS = dict(zip(FAMILIES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
GRAY, INK, MUTED, SURFACE = "#8a8984", "#0b0b0b", "#52514e", "#fcfcfb"

# ---------------------------------------------------------------- data
train = pd.read_csv(ROOT / "process_dataset/one_hot_train.csv", index_col=0).reset_index(drop=True)
test = pd.read_csv(ROOT / "process_dataset/one_hot_test.csv", index_col=0).reset_index(drop=True)
raw = pd.read_csv(ROOT / "Dataset/train.csv")
types = json.loads((ROOT / "mapping/after_process_column_types.json").read_text(encoding="utf-8"))
assert (train["Id"].values == raw["Id"].values).all(), "processed and raw train rows are not aligned"

order = np.lexsort((raw["Id"], raw["MoSold"], raw["YrSold"]))  # chronological
train, raw = train.iloc[order].reset_index(drop=True), raw.iloc[order].reset_index(drop=True)

y = train["SalePrice"].to_numpy()
X = train.drop(columns=["Id", "SalePrice"]).astype(float)
X_test = test[X.columns].astype(float)
cat_roots = [k for k, v in types.items() if v == "binary"]
DUMMY = [c for c in X if c not in types and any(c.startswith(k + "_") for k in cat_roots)]
BASE = [c for c in X if c not in DUMMY]
OUTLIER = ((raw["GrLivArea"] > 4000) & (raw["SalePrice"] < 300000)).to_numpy()
raw_names = set(raw.columns) - {"Id", "SalePrice"}
# 原始欄位的重新編碼:資訊量等同原欄位,Strong baseline 應該拿得到
REENCODED = {"HouseAge", "RemodAge", "GarageAge", "sin_MoSold", "cos_MoSold"}
is_raw = lambda c: c in raw_names or c in REENCODED or any(c.startswith(k + "_") for k in raw_names)
RAW_COLS = [c for c in X if is_raw(c)]
ENGINEERED = [c for c in X if not is_raw(c)]
print("engineered (truly new):", ENGINEERED)
SALE_DATE = raw["YrSold"].astype(str) + "-" + raw["MoSold"].map("{:02d}".format)

SPLITS = {
    "timeseries": list(TimeSeriesSplit(N_SPLITS).split(X)),
    "kfold": list(KFold(N_SPLITS, shuffle=True, random_state=SEED).split(X)),
}


def fold_stats(tr):
    """|r| of every feature with y, and |r| among base features, on the training rows only."""
    Xt = X.iloc[tr]
    with np.errstate(divide="ignore", invalid="ignore"):  # constant dummy in this fold -> r = 0
        r = Xt.corrwith(pd.Series(y[tr], index=Xt.index)).abs().fillna(0.0)
        C = Xt[BASE].corr().abs().fillna(0.0)
    return tr, r, C


PRE = {}
for scheme, splits in SPLITS.items():
    for k, (tr, _) in enumerate(splits):
        for drop in DROP_OUTLIERS:
            PRE[scheme, k, drop] = fold_stats(tr[~OUTLIER[tr]] if drop else tr)
FULL = {drop: fold_stats(np.where(~OUTLIER)[0] if drop else np.arange(len(y))) for drop in DROP_OUTLIERS}


def select(r, C, cfg):
    if cfg.get("fixed"):
        return list(cfg["fixed"])
    thr = cfg["threshold"]
    base = r[BASE][r[BASE] >= thr].sort_values(ascending=False).index.tolist()
    if cfg["prune"] < 1.0:
        kept = []
        for f in base:
            if all(C.at[f, g] <= cfg["prune"] for g in kept):
                kept.append(f)
        base = kept
    dummies = {"none": [], "all": DUMMY, "filtered": r[DUMMY][r[DUMMY] >= thr].index.tolist()}[cfg["cat_mode"]]
    return base + dummies


def make_model(cfg):
    p = cfg["params"]
    est = {
        "Mean": lambda: DummyRegressor(),
        "OLS": lambda: LinearRegression(),
        "Ridge": lambda: Ridge(alpha=p["alpha"]),
        "Lasso": lambda: Lasso(alpha=p["alpha"], max_iter=5000),
        "ElasticNet": lambda: ElasticNet(alpha=p["alpha"], l1_ratio=p["l1_ratio"], max_iter=5000),
        "Huber": lambda: HuberRegressor(epsilon=p["epsilon"], alpha=p["alpha"], max_iter=1000),
        "RF": lambda: RandomForestRegressor(
            n_estimators=500, max_depth=p.get("max_depth"), max_features=p.get("max_features", 1.0),
            min_samples_leaf=p.get("min_samples_leaf", 1), n_jobs=1, random_state=SEED),
        "HGB": lambda: HistGradientBoostingRegressor(
            max_iter=p.get("max_iter", 500), learning_rate=p["lr"], max_leaf_nodes=p["leaves"],
            l2_regularization=p.get("l2", 0.0), random_state=SEED),
    }[cfg["model"]]()
    return make_pipeline(StandardScaler(), est)


def usd(log_pred):
    # clip only for the dollar metric so a wild log prediction cannot overflow expm1
    return np.expm1(np.clip(log_pred, y.min() - 1, y.max() + 1))


def run(cfg, scheme, keep=False):
    rows, oof, coefs = [], np.full(len(y), np.nan), []
    for k, (_, va) in enumerate(SPLITS[scheme]):
        tr, r, C = PRE[scheme, k, cfg["drop_outliers"]]
        feats = select(r, C, cfg)
        m = make_model(cfg).fit(X.iloc[tr][feats], y[tr])
        p, pt = m.predict(X.iloc[va][feats]), m.predict(X.iloc[tr][feats])
        rows.append({
            "scheme": scheme, "fold": k, "n_feats": len(feats), "uses_time": bool(TIME_FEATURES & set(feats)),
            "rmse": np.sqrt(np.mean((p - y[va]) ** 2)),
            "train_rmse": np.sqrt(np.mean((pt - y[tr]) ** 2)),
            "mae_usd": np.mean(np.abs(usd(p) - np.expm1(y[va]))),
            "r2": 1 - np.sum((p - y[va]) ** 2) / np.sum((y[va] - y[va].mean()) ** 2),
            "val_period": f"{SALE_DATE[va[0]]}..{SALE_DATE[va[-1]]}",
        })
        if keep:
            oof[va] = p
            if hasattr(m[-1], "coef_"):
                coefs.append(pd.Series(m[-1].coef_, feats))
    return rows, oof, coefs


def run_chunk(cfgs):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    out = []
    for i, cfg in cfgs:
        for scheme in SPLITS:
            out += [{"cfg_id": i, **row} for row in run(cfg, scheme)[0]]
    return out


def cfg_label(cfg):
    return ", ".join(f"{k}={v}" for k, v in cfg["params"].items()) or "-"


# ---------------------------------------------------------------- figures
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e8e7e3",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 2, "font.size": 10, "axes.titlesize": 11,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    return f"figures/{name}"


def line_by_family(ax, table, x, ymax=None, end_labels=False):
    """One line per model family. Points above ymax are drawn at the top edge and annotated."""
    for fam in FAMILIES:
        s = table[fam].dropna()
        shown = s.clip(upper=ymax) if ymax else s
        ax.plot(shown.index, shown.values, "o-", ms=6, color=COLORS[fam], label=fam)
        for xi, v in s[s > ymax].items() if ymax else []:
            ax.annotate(f"{fam}: {v:.3g}\n(off scale)", (xi, ymax), xytext=(10, -14), textcoords="offset points",
                        ha="left", fontsize=8, color=MUTED)
    if ymax:
        ax.set_ylim(top=ymax * 1.01)
    if end_labels:  # dodge labels so close lines do not overlap
        ends = table[FAMILIES].iloc[-1].sort_values()
        lo, hi = ax.get_ylim()
        gap, last = (hi - lo) * 0.045, -np.inf
        for fam, v in ends.items():
            last = max(v, last + gap)
            ax.annotate(fam, (table.index[-1], v), xytext=(table.index[-1] * 1.03 + 0.01, last), textcoords="data",
                        fontsize=8, color=INK, va="center", ha="left")
        ax.set_xlim(right=table.index[-1] + (table.index[-1] - table.index[0]) * 0.12)
    ax.set_xlabel(x)


# ---------------------------------------------------------------- main
def main():
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    grid = [
        {"model": m, "params": p, "threshold": t, "cat_mode": c, "prune": pr, "drop_outliers": d}
        for t, c, pr, d, (m, p) in itertools.product(THRESHOLDS, CAT_MODES, PRUNE, DROP_OUTLIERS, MODEL_GRID)
        if not (t == 0.0 and c == "filtered")  # identical to cat_mode="all" at threshold 0
    ]
    def _bl(name, model, params, fixed):
        return {"model": model, "params": params, "fixed": fixed, "threshold": np.nan,
                "cat_mode": "-", "prune": 1.0, "drop_outliers": False, "baseline": name}

    baselines = [
        # Trivial(順序不能動,後面 baselines[0] 被當成 mean baseline 用)
        _bl("Predict mean", "Mean", {}, ["OverallQual"]),
        # Simple:兩個最直覺的單變數,取較強者
        _bl("OLS on OverallQual only", "OLS", {}, ["OverallQual"]),
        _bl("OLS on GrLivArea only", "OLS", {}, ["GrLivArea"]),
        # Strong:原始欄位 + 非線性模型,小範圍調參
        *[_bl(f"RF raw (mf={mf}, leaf={l})", "RF", {"max_features": mf, "min_samples_leaf": l}, RAW_COLS)
          for mf in [0.3, 0.5] for l in [1, 3]],
        *[_bl(f"HGB raw (lr={lr}, leaves={lv}, l2={l2})", "HGB", {"lr": lr, "leaves": lv, "l2": l2}, RAW_COLS)
          for lr in [0.03, 0.06] for lv in [8, 15] for l2 in [0.0, 1.0]],
    ]
    cfgs = grid + baselines
    jobs = list(enumerate(cfgs))
    chunks = [jobs[i::32] for i in range(32)]
    folds = pd.DataFrame([r for part in Parallel(n_jobs=-1)(delayed(run_chunk)(c) for c in chunks) for r in part])
    print(f"{len(cfgs)} configs x {len(SPLITS)} CV schemes x {N_SPLITS} folds in {time.time() - t0:.0f}s")

    meta = pd.DataFrame([{
        "cfg_id": i, "model": c["model"], "params": cfg_label(c), "threshold": c["threshold"],
        "cat_mode": c["cat_mode"], "prune": c["prune"], "drop_outliers": c["drop_outliers"],
        "baseline": c.get("baseline", ""), **{f"p_{k}": v for k, v in c["params"].items()},
    } for i, c in enumerate(cfgs)])
    agg = folds.groupby(["cfg_id", "scheme"]).agg(
        rmse=("rmse", "mean"), rmse_sd=("rmse", "std"), train_rmse=("train_rmse", "mean"),
        mae_usd=("mae_usd", "mean"), r2=("r2", "mean"), n_feats=("n_feats", "mean"), uses_time=("uses_time", "any"),
    ).unstack("scheme")
    agg.columns = [f"{'ts' if s == 'timeseries' else 'kf'}_{m}" for m, s in agg.columns]
    res = meta.merge(agg.reset_index(), on="cfg_id")
    res["uses_time"] = res["ts_uses_time"] | res["kf_uses_time"]
    res["primary_cv"] = np.where(res["uses_time"], "timeseries", "kfold")
    res["primary_rmse"] = np.where(res["uses_time"], res["ts_rmse"], res["kf_rmse"])
    res = res.drop(columns=["ts_uses_time", "kf_uses_time"]).sort_values("ts_rmse").reset_index(drop=True)
    res.to_csv(OUT / "results_all.csv", index=False)
    folds.merge(meta, on="cfg_id").to_csv(OUT / "results_folds.csv", index=False)

    gridres = res[res["baseline"] == ""]
    base = res[res["baseline"] != ""].set_index("baseline")
    best_fam = gridres.loc[gridres.groupby("model")["ts_rmse"].idxmin()].set_index("model").loc[FAMILIES]
    best = gridres.iloc[0]
    best_cfg = cfgs[int(best["cfg_id"])]
    figs = {}

    # F1 threshold curve (best over all other params, per family) + F2 feature count
    thr_tab = gridres.groupby(["threshold", "model"])["ts_rmse"].min().unstack()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(base.loc["OLS on OverallQual only", "ts_rmse"], color=GRAY, ls="--", lw=1)  # before labels: sets y-range for dodging
    line_by_family(ax, thr_tab, "correlation threshold |r| >= t", end_labels=True)
    ax.text(THRESHOLDS[0], base.loc["OLS on OverallQual only", "ts_rmse"], " baseline: OverallQual only", color=MUTED, va="bottom", fontsize=8)
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax.set(title="F1. Best time-series CV RMSE (log price) at each threshold", ylabel="RMSE, lower is better")
    figs["F1"] = save(fig, "F1_threshold_vs_rmse.png")

    nf = gridres[gridres["model"] == "Ridge"].groupby(["threshold", "cat_mode"])["ts_n_feats"].mean().unstack()
    fig, ax = plt.subplots(figsize=(7, 4))
    for c, col in zip(CAT_MODES, ["#2a78d6", "#eb6834", "#1baf7a"]):
        s = nf[c].dropna()
        ax.plot(s.index, s.values, "o-", ms=6, color=col, label=f"dummies: {c}")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    ax.set(title="F2. Number of selected features vs. threshold (mean over folds)", xlabel="correlation threshold", ylabel="features (log scale)")
    figs["F2"] = save(fig, "F2_threshold_vs_n_features.png")

    # F3 other selection params: cat_mode / prune / outliers
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, (col, title) in zip(axes, [("cat_mode", "one-hot dummies"), ("prune", "collinearity pruning (1.0 = off)"), ("drop_outliers", "drop GrLivArea outliers")]):
        t = gridres.groupby([col, "model"])["ts_rmse"].min().unstack()[FAMILIES]
        t.index = t.index.astype(str)
        line_by_family(ax, t, title, ymax=base.loc["OLS on OverallQual only", "ts_rmse"])
        ax.set_title(title)
    axes[0].set_ylabel("best time-series CV RMSE")
    axes[1].legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.suptitle("F3. Effect of the other selection parameters (best over everything else)", fontsize=11)
    figs["F3"] = save(fig, "F3_selection_params.png")

    # F4 regularisation strength at each family's best selection setting
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, fam in zip(axes, ["Ridge", "Lasso", "ElasticNet"]):
        b = best_fam.loc[fam]
        sub = gridres[(gridres["model"] == fam) & (gridres["threshold"] == b["threshold"]) & (gridres["cat_mode"] == b["cat_mode"])
                      & (gridres["prune"] == b["prune"]) & (gridres["drop_outliers"] == b["drop_outliers"])]
        if fam == "ElasticNet":
            for l, col in zip([0.2, 0.5, 0.8], ["#2a78d6", "#eb6834", "#1baf7a"]):
                s = sub[sub["p_l1_ratio"] == l].sort_values("p_alpha")
                ax.plot(s["p_alpha"], s["ts_rmse"], "o-", ms=6, color=col, label=f"l1_ratio={l}")
            ax.legend(frameon=False)
        else:
            s = sub.sort_values("p_alpha")
            ax.plot(s["p_alpha"], s["ts_rmse"], "o-", ms=6, color=COLORS[fam])
            ax.plot(s["p_alpha"], s["kf_rmse"], "o--", ms=6, color=GRAY, label="KFold")
            ax.legend(frameon=False)
        ax.set_xscale("log")
        ax.set(title=f"{fam} (t={b['threshold']}, dummies={b['cat_mode']})", xlabel="alpha")
    axes[0].set_ylabel("CV RMSE (solid = time-series)")
    fig.suptitle("F4. Regularisation strength", fontsize=11)
    figs["F4"] = save(fig, "F4_alpha_curves.png")

    # F5 best per family: time-series vs kfold
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(len(FAMILIES))
    ax.bar(xs - 0.2, best_fam["ts_rmse"], 0.38, yerr=best_fam["ts_rmse_sd"], color="#2a78d6", label="TimeSeriesSplit", error_kw={"ecolor": MUTED, "lw": 1})
    ax.bar(xs + 0.2, best_fam["kf_rmse"], 0.38, yerr=best_fam["kf_rmse_sd"], color="#eb6834", label="KFold (shuffled)", error_kw={"ecolor": MUTED, "lw": 1})
    for i, fam in enumerate(FAMILIES):
        ax.text(i - 0.2, best_fam.loc[fam, "ts_rmse"] / 2, f"{best_fam.loc[fam, 'ts_rmse']:.3f}", ha="center", color="white", fontsize=8)
        ax.text(i + 0.2, best_fam.loc[fam, "kf_rmse"] / 2, f"{best_fam.loc[fam, 'kf_rmse']:.3f}", ha="center", color="white", fontsize=8)
    ax.axhline(base.loc["Predict mean", "ts_rmse"], color=GRAY, ls="--", lw=1)
    ax.text(len(FAMILIES) - 0.5, base.loc["Predict mean", "ts_rmse"], "predict-mean baseline", color=MUTED, ha="right", va="bottom", fontsize=8)
    ax.set_xticks(xs, FAMILIES)
    ax.legend(frameon=False)
    ax.set(title="F5. Best config per model family (error bar = SD over folds)", ylabel="CV RMSE (log price)")
    figs["F5"] = save(fig, "F5_best_per_family.png")

    # Best model: time-series out-of-fold predictions, fold errors, coefficients
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    ts_rows, oof, fold_coefs = run(best_cfg, "timeseries", keep=True)
    mean_rows = run(baselines[0], "timeseries")[0]
    ts_rows, mean_rows = pd.DataFrame(ts_rows), pd.DataFrame(mean_rows)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ts_rows["val_period"], ts_rows["rmse"], "o-", ms=7, color="#2a78d6", label=f"best: {best['model']}")
    ax.plot(mean_rows["val_period"], mean_rows["rmse"], "o--", ms=7, color=GRAY, label="predict-mean baseline")
    ax.legend(frameon=False)
    ax.tick_params(axis="x", labelrotation=20)
    ax.set(title="F6. Time-series CV: error on each future validation period", ylabel="RMSE (log price)", xlabel="validation period (sale dates)")
    figs["F6"] = save(fig, "F6_error_over_time.png")

    m = ~np.isnan(oof)
    d = raw.loc[m, ["Id", "Neighborhood", "SaleCondition", "GrLivArea", "OverallQual", "SalePrice"]].copy()
    d["pred"] = usd(oof[m])
    d["resid_log"] = oof[m] - y[m]
    d["err_pct"] = (d["pred"] - d["SalePrice"]) / d["SalePrice"] * 100

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    axes[0].scatter(d["SalePrice"] / 1e3, d["pred"] / 1e3, s=10, color="#2a78d6", alpha=0.5, lw=0)
    lim = [0, d[["SalePrice", "pred"]].max().max() / 1e3 * 1.05]
    axes[0].plot(lim, lim, color=MUTED, lw=1)
    axes[0].set(title="Actual vs. predicted", xlabel="actual ($k)", ylabel="predicted ($k)", xlim=lim, ylim=lim)
    axes[1].scatter(oof[m], d["resid_log"], s=10, color="#2a78d6", alpha=0.5, lw=0)
    axes[1].axhline(0, color=MUTED, lw=1)
    axes[1].set(title="Residuals vs. fitted", xlabel="predicted log price", ylabel="residual (pred - actual, log)")
    (osm, osr), (slope, icpt, _) = stats.probplot(d["resid_log"], dist="norm")
    axes[2].scatter(osm, osr, s=10, color="#2a78d6", alpha=0.5, lw=0)
    axes[2].plot(osm, slope * osm + icpt, color=MUTED, lw=1)
    axes[2].set(title="Normal Q-Q of residuals", xlabel="theoretical quantile", ylabel="residual quantile")
    fig.suptitle(f"F7. Residual diagnostics, best model ({best['model']}), time-series out-of-fold predictions", fontsize=11)
    figs["F7"] = save(fig, "F7_residuals.png")

    d["price_band"] = pd.qcut(d["SalePrice"], 5, labels=["Q1 cheapest", "Q2", "Q3", "Q4", "Q5 priciest"])
    band = d.groupby("price_band", observed=True).agg(mape=("err_pct", lambda e: e.abs().mean()), bias=("err_pct", "mean"), n=("Id", "size"))
    nb = d.groupby("Neighborhood").agg(mape=("err_pct", lambda e: e.abs().mean()), n=("Id", "size"))
    nb = nb[nb["n"] >= 15].sort_values("mape")
    sc = d.groupby("SaleCondition").agg(mape=("err_pct", lambda e: e.abs().mean()), bias=("err_pct", "mean"), n=("Id", "size")).sort_values("mape")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={"width_ratios": [1, 1, 1.2]})
    axes[0].bar(band.index.astype(str), band["mape"], color="#2a78d6")
    axes[0].set(title="Mean absolute % error by price quintile", ylabel="MAPE (%)")
    axes[0].tick_params(axis="x", labelrotation=20)
    axes[1].bar(band.index.astype(str), band["bias"], color=np.where(band["bias"] >= 0, "#2a78d6", "#eb6834"))
    axes[1].axhline(0, color=MUTED, lw=1)
    axes[1].set(title="Mean signed % error (bias) by price quintile", ylabel="% (+ over-, - under-predict)")
    axes[1].tick_params(axis="x", labelrotation=20)
    axes[2].barh(nb.index, nb["mape"], color="#2a78d6")
    axes[2].set(title="MAPE by neighbourhood (n >= 15)", xlabel="MAPE (%)")
    fig.suptitle("F8. Error analysis, best model", fontsize=11)
    figs["F8"] = save(fig, "F8_error_groups.png")

    tr_full, r_full, C_full = FULL[best_cfg["drop_outliers"]]
    feats_full = select(r_full, C_full, best_cfg)
    final = make_model(best_cfg).fit(X.iloc[tr_full][feats_full], y[tr_full])
    coef = pd.Series(final[-1].coef_, feats_full)
    top = coef.reindex(coef.abs().sort_values(ascending=False).index[:20])
    fc = pd.concat(fold_coefs, axis=1).reindex(top.index) if fold_coefs else None
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ypos = np.arange(len(top))[::-1]
    ax.barh(ypos, top.values, color=np.where(top.values >= 0, "#2a78d6", "#eb6834"), height=0.6)
    if fc is not None:
        ax.hlines(ypos, fc.min(axis=1), fc.max(axis=1), color=INK, lw=1.2, label="range across time-series folds")
        ax.legend(frameon=False, loc="lower right")
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(ypos, top.index)
    ax.set(title="F9. Top 20 standardised coefficients (final model on all train)", xlabel="change in log price per +1 SD of feature")
    figs["F9"] = save(fig, "F9_coefficients.png")

    sub = pd.DataFrame({"Id": test["Id"], "SalePrice": np.expm1(final.predict(X_test[feats_full]))})
    sub.to_csv(OUT / "submission.csv", index=False)
    # one Kaggle submission per baseline tier (best config of each tier by time-series CV)
    bt = base.reset_index()
    bt["tier"] = np.where(bt["baseline"].str.startswith("Predict"), "1_trivial",
                  np.where(bt["baseline"].str.startswith("OLS"), "2_simple", "3_strong"))
    picks = bt.sort_values("ts_rmse").groupby("tier").head(1)
    sub_dir = OUT / "submissions"
    sub_dir.mkdir(exist_ok=True)
    for _, row in picks.iterrows():
        cfg = cfgs[int(row["cfg_id"])]
        tr_b, r_b, C_b = FULL[cfg["drop_outliers"]]
        feats_b = select(r_b, C_b, cfg)
        mdl = make_model(cfg).fit(X.iloc[tr_b][feats_b], y[tr_b])
        out = pd.DataFrame({"Id": test["Id"], "SalePrice": np.expm1(mdl.predict(X_test[feats_b]))})
        out.to_csv(sub_dir / f"submission_{row['tier']}.csv", index=False)
        print(row["tier"], row["baseline"], "->", f"submission_{row['tier']}.csv")
    write_report(res, gridres, base, best_fam, best, best_cfg, thr_tab, ts_rows, band, nb, sc, d, coef, feats_full, figs, len(cfgs))
    print(f"done in {time.time() - t0:.0f}s -> {OUT / 'report.md'}")


# ---------------------------------------------------------------- report
def md(df, floatfmt="{:.4f}"):
    df = df.copy()
    for c in df:
        if pd.api.types.is_float_dtype(df[c]):
            fmt = ("{:,.0f}" if any(k in str(c) for k in ("usd", "SalePrice", "pred", "GrLivArea"))
                   else "{:g}" if c in ("threshold", "prune") else "{:.1f}" if "n_feats" in str(c) else floatfmt)
            df[c] = df[c].map(lambda v, f=fmt: "" if pd.isna(v) else f.format(v))
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def write_report(res, gridres, base, best_fam, best, best_cfg, thr_tab, ts_rows, band, nb, sc, d, coef, feats_full, figs, n_cfgs):
    show = ["model", "params", "threshold", "cat_mode", "prune", "drop_outliers", "ts_rmse", "ts_rmse_sd", "kf_rmse",
            "ts_mae_usd", "ts_r2", "ts_train_rmse", "ts_n_feats", "uses_time", "primary_cv"]
    best_tab = best_fam.reset_index()[show]
    bt = base.reset_index()
    bt["tier"] = np.where(bt["baseline"].str.startswith("Predict"), "1 Trivial",
                  np.where(bt["baseline"].str.startswith("OLS"), "2 Simple", "3 Strong"))
    base_tab = (bt.sort_values("ts_rmse").groupby("tier").head(1).sort_values("tier")
                [["tier", "baseline", "ts_rmse", "ts_rmse_sd", "kf_rmse", "ts_mae_usd", "ts_r2"]])
    worst = d.reindex(d["err_pct"].abs().sort_values(ascending=False).index[:10])[
        ["Id", "Neighborhood", "SaleCondition", "GrLivArea", "OverallQual", "SalePrice", "pred", "err_pct"]]
    best_thr = thr_tab.idxmin()
    gap = (best_fam["ts_rmse"] - best_fam["kf_rmse"])
    outlier_fold = [k for k, (_, va) in enumerate(SPLITS["timeseries"]) if OUTLIER[va].any()]
    by_param = {col: gridres.groupby(col)["ts_rmse"].min() for col in ["cat_mode", "prune", "drop_outliers"]}

    grid_rows = pd.DataFrame([
        ["Feature selection", "threshold", str(THRESHOLDS), "keep base feature if fold-train abs(Pearson r) with log price >= t"],
        ["Feature selection", "cat_mode", str(CAT_MODES), "one-hot dummies: none / same threshold / all"],
        ["Feature selection", "prune", str(PRUNE), "greedy: drop feature whose abs(r) with a stronger kept feature > value (1.0 = off)"],
        ["Training data", "drop_outliers", str(DROP_OUTLIERS), "remove GrLivArea>4000 & SalePrice<300k from training folds"],
        ["OLS", "-", "-", "LinearRegression"],
        ["Ridge", "alpha", str([p["alpha"] for m, p in MODEL_GRID if m == "Ridge"]), "L2 penalty"],
        ["Lasso", "alpha", str([p["alpha"] for m, p in MODEL_GRID if m == "Lasso"]), "L1 penalty"],
        ["ElasticNet", "alpha x l1_ratio", f"{sorted({p['alpha'] for m, p in MODEL_GRID if m == 'ElasticNet'})} x {sorted({p['l1_ratio'] for m, p in MODEL_GRID if m == 'ElasticNet'})}", "mixed L1/L2"],
        ["Huber", "epsilon x alpha", f"{sorted({p['epsilon'] for m, p in MODEL_GRID if m == 'Huber'})} x {sorted({p['alpha'] for m, p in MODEL_GRID if m == 'Huber'})}", "robust loss; alpha = L2 penalty"],
        ["CV", "scheme", "TimeSeriesSplit(5), KFold(5, shuffle, seed 0)", "rows sorted by YrSold, MoSold"],
    ], columns=["group", "parameter", "values tried", "meaning"])

    txt = f"""# Linear Regression with Correlation-Threshold Tuning (Ames House Prices)

Generated by `model/threshold_tuning.py`. Re-run it to regenerate every number and figure.

## 1. Setup

- **Target:** `log1p(SalePrice)` (from `process_dataset/one_hot_train.csv`). Metrics are in log units unless marked $.
  RMSE on log price is roughly the typical **relative** error (0.12 ≈ 12%) and is the official Kaggle metric.
- **Data:** {len(y)} training houses sold {SALE_DATE.iloc[0]} – {SALE_DATE.iloc[-1]}; {len(BASE)} base features
  (numeric / ordinal / binary / cyclic / engineered) + {len(DUMMY)} one-hot dummies.
- **Test labels:** `Dataset/sample_submission.csv` holds Kaggle placeholder prices, **not** true prices, so
  it is not used for evaluation. All scores below are cross-validated on train. `submission.csv` contains
  the best model's predictions for Kaggle upload.
- **Leakage control:** correlation-based selection, collinearity pruning, scaling and outlier removal are all
  computed on the training part of each fold only.
- **Time-series CV:** rows are sorted by sale date. `TimeSeriesSplit(5)` trains on the past and validates on the
  next period. Time-related features are `{sorted(TIME_FEATURES)}`. A config that selects any of them has
  `primary_cv = timeseries`; otherwise shuffled `KFold(5)` would be acceptable. **For a fair comparison every config
  is ranked by its time-series CV RMSE** (KFold is reported alongside).
  {int(gridres['uses_time'].sum())} of {len(gridres)} grid configs use time-related features.
- **Search size:** {n_cfgs} configurations × 2 CV schemes × {N_SPLITS} folds = {n_cfgs * 2 * N_SPLITS:,} model fits.

## 2. Tunable parameters

{md(grid_rows)}

All metric columns: `ts_` = TimeSeriesSplit, `kf_` = KFold; `*_rmse` = mean validation RMSE (log), `*_rmse_sd` = SD over folds,
`ts_mae_usd` = mean absolute error in dollars, `ts_train_rmse` = training-fold RMSE (overfitting check),
`ts_n_feats` = mean number of selected features.

## 3. Baselines

{md(base_tab)}

## 4. Best configuration per model family

{md(best_tab)}

**Overall best:** {best['model']} ({best['params']}), threshold = {best['threshold']}, dummies = {best['cat_mode']},
prune = {best['prune']}, drop_outliers = {best['drop_outliers']} → time-series CV RMSE **{best['ts_rmse']:.4f} ± {best['ts_rmse_sd']:.4f}**,
MAE ≈ **${best['ts_mae_usd']:,.0f}**, R² = {best['ts_r2']:.3f} (vs. {base.loc['Predict mean', 'ts_rmse']:.4f} for predicting the mean).
The final model uses {len(feats_full)} features when refit on all training data.

![F5]({figs['F5']})

## 5. Effect of the correlation threshold

Best time-series CV RMSE at each threshold (minimum over all other parameters):

{md(thr_tab[FAMILIES].reset_index())}

Best threshold per family: {", ".join(f"{k} → {v}" for k, v in best_thr[FAMILIES].items())}.

![F1]({figs['F1']})
![F2]({figs['F2']})

## 6. Other selection parameters

Best time-series CV RMSE for each value (minimum over everything else):

- **cat_mode:** {", ".join(f"{k} = {v:.4f}" for k, v in by_param['cat_mode'].items())}
- **prune:** {", ".join(f"{k} = {v:.4f}" for k, v in by_param['prune'].items())}
- **drop_outliers:** {", ".join(f"{k} = {v:.4f}" for k, v in by_param['drop_outliers'].items())}

![F3]({figs['F3']})

## 7. Regularisation strength

![F4]({figs['F4']})

## 8. Time-series vs. shuffled KFold

Difference (time-series − KFold RMSE) for the best config of each family:
{", ".join(f"{k}: {v:+.4f}" for k, v in gap.items())}.
A positive gap means shuffled KFold looks optimistic. Outliers are removed from training folds only, so the two
huge partial sales (Id 524, 1299) stay in validation fold {outlier_fold} and dominate its error. Part of the gap is because the early time-series folds have
far less training data (the first fold trains on ~{len(SPLITS['timeseries'][0][0])} rows), not only because of
future information leaking into the training folds.

Per-fold errors of the best model:

{md(ts_rows[['fold', 'val_period', 'n_feats', 'rmse', 'train_rmse', 'mae_usd', 'r2']])}

![F6]({figs['F6']})

## 9. Error analysis (best model, time-series out-of-fold predictions, {len(d)} houses)

![F7]({figs['F7']})

By price quintile (`bias` > 0 = over-predicts):

{md(band.reset_index(), "{:.2f}")}

By sale condition:

{md(sc.reset_index(), "{:.2f}")}

Neighbourhoods (n ≥ 15), largest MAPE: {", ".join(f"{k} {v:.1f}%" for k, v in nb['mape'].iloc[::-1].head(5).items())};
smallest: {", ".join(f"{k} {v:.1f}%" for k, v in nb['mape'].head(5).items())}.

![F8]({figs['F8']})

Ten largest errors:

{md(worst, "{:,.1f}")}

## 10. Coefficients of the final model

Features are standardised, so coefficient size is comparable: change in log price per +1 SD (≈ % change ×100).
Whiskers in F9 show the range across time-series folds; a whisker crossing 0 means the sign is unstable.

{md(coef.reindex(coef.abs().sort_values(ascending=False).index[:20]).rename('coef').reset_index().rename(columns={'index': 'feature'}))}

![F9]({figs['F9']})

## 11. Top 15 configurations overall

{md(gridres.head(15)[show])}

Full results: `results_all.csv` (one row per configuration) and `results_folds.csv` (one row per fold).

## 12. Notes and limitations

- Hyperparameters are chosen on the same CV that reports the score (no nested CV), so the best score is slightly optimistic.
- Pearson correlation only captures linear, one-feature-at-a-time relationships; dummies are judged by point-biserial r.
- The processed data was z-scored on all of train before this script; re-scaling inside each fold removes the
  effect on the models, but fill-NA and category cleaning were done globally.
- Price bands use quintiles of the actual price, so the bias by band partly reflects regression to the mean.
"""
    (OUT / "report.md").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
