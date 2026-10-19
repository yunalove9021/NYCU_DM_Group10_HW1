"""Shared data loading, feature sets, models and time-series CV engine for the linear-regression pipeline.

Pipeline (run in this order, or `python model/run_all.py`):
  eda.py            exploratory feature analysis (train; drift to test)
  tune.py           correlation threshold x model hyperparameters, time-series CV on train -> best_config.json
  final_model.py    refit best config on train, evaluate on the time-split test, coefficients / significance / VIF
  ablation.py       feature sets A-F, same model family + tuning procedure, paired significance tests
  error_analysis.py test points with |residual| > 2 SD
  report.py         assemble results/report.md
All outputs go to model/results/.
"""
import json
import warnings
from pathlib import Path

import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import skew
from sklearn.linear_model import HuberRegressor, enet_path
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "results"
FIG = OUT / "figures"
SEED = 0
N_SPLITS = 5
N_BOOT = 1000
N_JOBS = -1

# ------------------------------------------------------------------ tunable parameters (sources in report section 2)
THRESHOLDS = [0.0, 0.1, 0.3, 0.5]            # Cohen (1988): none / small / medium / large |r|
CAT_MODES = ["none", "filtered", "all"]       # one-hot dummies: drop / same threshold / keep all
DROP_OUTLIERS = [False, True]                 # GrLivArea > 4000 from training folds (De Cock 2011)
TOPK = [5, 10, 15, 20, 30, 40, 60, 80, 120, 160, 200, 240, 260, 280]
N_ALPHAS, ALPHA_MIN_RATIO = 100, 1e-4         # glmnet path for Lasso / ElasticNet (Friedman et al. 2010)
L1_RATIOS = [0.01, 0.05, 0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]  # 1.0 = Lasso; 0.01 / 0.05 added: optimum was on the 0.1 edge
N_RIDGE = 30                                  # Ridge alphas spanning effective df 1 .. 0.95 p (Hastie et al. 2009 §3.4.1)
HUBER_EPS = [1.345, 2.0]                      # 1.345 = 95% efficiency under normal errors (Huber 1981)
N_HUBER = 8
FAMILIES = ["OLS", "Ridge", "Lasso", "ElasticNet", "Huber"]

ENGINEERED = ["HouseAge", "RemodAge", "GarageAge", "HasGarage", "IsRemod", "sin_MoSold", "cos_MoSold",
              "TotalArea", "TotalBaths", "TotalPorchArea", "AvgRoomSize", "LotUtilization",
              "Qual_x_Area", "TotalQualCond", "HasBasement", "HasFireplace", "HasPool"]
# Columns the team's pre-processing log1p-transforms when skewness > 0.75 (data_preprocess.add_custom_features).
# The decision is re-made inside every fit, on that fit's training rows only, from the raw values.
SKEW_THRESHOLD = 0.75
SKEW_COLS = ["LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF",
             "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr",
             "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF",
             "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal", "HouseAge", "RemodAge", "GarageAge", "YrSold",
             "TotalArea", "TotalBaths", "TotalPorchArea", "AvgRoomSize", "LotUtilization", "Qual_x_Area", "TotalQualCond"]
TIME_FEATURES = {"YrSold", "MoSold", "YearBuilt", "YearRemodAdd", "GarageYrBlt", "HouseAge", "RemodAge",
                 "GarageAge", "sin_MoSold", "cos_MoSold"}

COLORS = dict(zip(FAMILIES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
BLUE, ORANGE, RED, NAVY = "#2a78d6", "#eb6834", "#e34948", "#33414f"
GRAY, INK, MUTED, SURFACE = "#8a8984", "#0b0b0b", "#52514e", "#ffffff"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e8e7e3",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 2, "font.size": 17, "axes.titlesize": 18, "axes.labelsize": 17, "legend.fontsize": 14, "figure.titlesize": 19,
})


# ================================================================== data
def standardize(Xtr, *others):
    """z-score with mean / SD of the FIRST argument only (training part); applied unchanged to the others."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return [(Z - mu) / sd for Z in (Xtr, *others)]


def raw_numeric(r):
    """Pre-log values of SKEW_COLS, rebuilt from Dataset/train.csv with the team's formulas
    (NA -> 0, time features from data_preprocess.time_to_features, engineered features from add_custom_features)."""
    d = r.copy()
    base = [c for c in SKEW_COLS if c in r.columns]
    d[base] = d[base].fillna(0)
    garage = d["GarageYrBlt"].where(~(d["GarageYrBlt"] > 2010), d["YearBuilt"])
    out = {c: d[c] for c in base}
    out["HouseAge"] = (d["YrSold"] - d["YearBuilt"]).clip(lower=0)
    out["RemodAge"] = (d["YrSold"] - d["YearRemodAdd"]).clip(lower=0)
    out["GarageAge"] = (d["YrSold"] - garage).clip(lower=0).fillna(0)
    out["TotalArea"] = d["GrLivArea"] + d["TotalBsmtSF"]
    out["TotalBaths"] = d["FullBath"] + 0.5 * d["HalfBath"] + d["BsmtFullBath"] + 0.5 * d["BsmtHalfBath"]
    out["TotalPorchArea"] = d["WoodDeckSF"] + d["OpenPorchSF"] + d["EnclosedPorch"] + d["3SsnPorch"] + d["ScreenPorch"]
    out["AvgRoomSize"] = d["GrLivArea"] / d["TotRmsAbvGrd"].replace(0, 1)
    out["LotUtilization"] = d["GrLivArea"] / d["LotArea"].replace(0, 1)
    out["Qual_x_Area"] = d["OverallQual"] * d["GrLivArea"]
    out["TotalQualCond"] = d["OverallQual"] + d["OverallCond"]
    return pd.DataFrame(out)[SKEW_COLS].astype(float).to_numpy()


def log_decision(X, LOG):
    """log1p a column iff its skewness on these (training) rows exceeds SKEW_THRESHOLD (the team's rule)."""
    with np.errstate(all="ignore"):
        return np.nan_to_num(skew(X[:, LOG["raw_idx"]], axis=0), nan=0.0) > SKEW_THRESHOLD


def apply_log(X, logged, LOG):
    """Overwrite the processed SKEW_COLS with raw or log1p(raw) values, as decided on the training rows."""
    X = X.copy()
    R = X[:, LOG["raw_idx"]]
    X[:, LOG["num_idx"]] = np.where(logged, np.log1p(np.clip(R, 0, None)), R)
    return X


def repair_test(tr, te):
    """Work around a pre-processing bug: data_preprocess.add_custom_features() picks skewed (log1p) features
    separately on train and test, so some test columns got a different transform than train before the train
    scaler was applied. For a column far off the train distribution (> 1.5 SD), invert the train scaler and apply
    log1p (train logged, test not) or expm1 (test logged, train not) if that brings it back within 1 SD.
    Real shifts (e.g. YrSold under a time split) are left alone because neither transform fixes them."""
    # train scaler parameters from the team's pre-standardisation summary (the scaler pickle needs numpy 2.x);
    # StandardScaler divides by the population SD (ddof = 0)
    summ = pd.read_excel(ROOT / "summary/train_data_before_stand_summary.xlsx", index_col=0)
    n = len(tr)
    fixed = []
    for c in summ.index:
        if c not in te or tr[c].std() == 0 or abs(te[c].mean() - tr[c].mean()) <= 1.5 * tr[c].std():
            continue
        mean, scale = summ.at[c, "mean"], summ.at[c, "std"] * np.sqrt((n - 1) / n)
        pre = te[c].to_numpy(float) * scale + mean
        for name, f in (("log1p", np.log1p), ("expm1", np.expm1)):
            with np.errstate(all="ignore"):
                cand = (f(pre) - mean) / scale
            if np.isfinite(cand).all() and abs(cand.mean() - tr[c].mean()) < tr[c].std():
                fixed.append({"column": c, "applied": name, "test_mean_before": float(te[c].mean()), "test_mean_after": float(cand.mean())})
                te[c] = cand
                break
    return te, fixed


def load_data():
    FIG.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(ROOT / "process_dataset/one_hot_train.csv", index_col=0).reset_index(drop=True)
    te = pd.read_csv(ROOT / "process_dataset/one_hot_test.csv", index_col=0).reset_index(drop=True)
    te, repaired = repair_test(tr, te)
    raw = pd.read_csv(ROOT / "Dataset/train.csv").set_index("Id")
    types = json.loads((ROOT / "mapping/column_types.json").read_text(encoding="utf-8"))
    feats = [c for c in tr if c not in ("Id", "SalePrice")]
    assert [c for c in te if c not in ("Id", "SalePrice")] == feats, "train / test columns differ"
    assert set(ENGINEERED) <= set(feats), set(ENGINEERED) - set(feats)

    y_te = te["SalePrice"].to_numpy(float)
    if y_te.max() > 100:  # test target is stored in dollars, train target as log1p
        y_te = np.log1p(y_te)
    roots = [k for k, v in types.items() if v == "category"]
    dummy = [c for c in feats if c not in types and any(c.startswith(k + "_") for k in roots)]
    root_of = {c: max((k for k in roots if c.startswith(k + "_")), key=len) for c in dummy}
    original = [c for c in feats if c not in ENGINEERED]
    greedy = [f for f in pd.read_excel(ROOT / "models/greedy_select_feature.xlsx")["Feature"] if f in feats]
    raw_tr, raw_te = raw.loc[tr["Id"]].reset_index(), raw.loc[te["Id"]].reset_index()
    for r in (raw_tr, raw_te):
        r["date"] = pd.to_datetime(dict(year=r["YrSold"], month=r["MoSold"], day=15))
    assert raw_tr["date"].is_monotonic_increasing and raw_tr["date"].max() <= raw_te["date"].min(), "split is not chronological"

    # the raw (pre-log) values of SKEW_COLS ride along as extra columns after the features, so every fit can
    # decide its own log transform from its own training rows (columns beyond len(feats) are never model inputs)
    nf = len(feats)
    LOG = {"num_idx": np.array([feats.index(c) for c in SKEW_COLS]), "raw_idx": np.arange(nf, nf + len(SKEW_COLS))}
    D = {"Xtr": np.hstack([tr[feats].astype(float).to_numpy(), raw_numeric(raw_tr)]),
         "Xte": np.hstack([te[feats].astype(float).to_numpy(), raw_numeric(raw_te)]), "log": LOG,
         "ytr": tr["SalePrice"].to_numpy(float), "yte": y_te, "id_tr": tr["Id"].to_numpy(), "id_te": te["Id"].to_numpy(),
         "raw_tr": raw_tr, "raw_te": raw_te, "feats": feats, "col_index": {c: i for i, c in enumerate(feats)},
         "dummy": dummy, "dummy_set": set(dummy), "root_of": root_of, "original": original, "engineered": list(ENGINEERED),
         "greedy": greedy, "outlier_tr": (raw_tr["GrLivArea"] > 4000).to_numpy(),
         "splits": list(TimeSeriesSplit(N_SPLITS).split(tr)), "repaired": repaired}
    D["grids"] = make_grids(D)
    return D


def make_grids(D):
    """Data-driven hyperparameter ranges, computed once on the full training set (as cv.glmnet does)."""
    A = standardize(D["Xtr"][:, : len(D["feats"])])[0]
    yc = D["ytr"] - D["ytr"].mean()
    enet = {}
    for l1 in L1_RATIOS:  # alpha_max = smallest alpha with every coefficient 0
        amax = np.abs(A.T @ yc).max() / (len(yc) * l1)
        enet[l1] = np.logspace(np.log10(amax), np.log10(amax * ALPHA_MIN_RATIO), N_ALPHAS)
    d2 = np.linalg.svd(A, compute_uv=False) ** 2
    d2 = d2[d2 > 1e-10 * d2.max()]

    def alpha_for_df(target):  # effective df(alpha) = sum d^2 / (d^2 + alpha) is monotone in alpha
        lo, hi = -8.0, 14.0
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if np.sum(d2 / (d2 + 10 ** mid)) > target else (lo, mid)
        return 10 ** mid

    ridge = np.logspace(np.log10(alpha_for_df(0.95 * len(d2))), np.log10(alpha_for_df(1.0)), N_RIDGE)
    return {"enet": enet, "ridge": ridge, "huber": ridge[:: max(1, N_RIDGE // N_HUBER)][:N_HUBER],
            "rank": len(d2)}


def pool_parts(D, pool):
    pidx = [D["col_index"][c] for c in pool]
    base = [i for i, c in enumerate(pool) if c not in D["dummy_set"]]
    dummy = [i for i, c in enumerate(pool) if c in D["dummy_set"]]
    is_time = np.array([c in TIME_FEATURES for c in pool])
    return pidx, base, dummy, is_time


# ================================================================== models
def param_keys(families, grids):
    keys = []
    if "OLS" in families:
        keys.append(("OLS", None, None))
    if "Ridge" in families:
        keys += [("Ridge", a, None) for a in grids["ridge"]]
    for l1 in L1_RATIOS:
        fam = "Lasso" if l1 == 1.0 else "ElasticNet"
        if fam in families:
            keys += [(fam, a, l1) for a in grids["enet"][l1]]
    if "Huber" in families:
        keys += [("Huber", a, e) for e in HUBER_EPS for a in grids["huber"]]
    return keys


def key_label(key):
    fam, a, x = key
    if fam == "OLS":
        return "-"
    if fam == "ElasticNet":
        return f"alpha={a:.3g}, l1_ratio={x}"
    if fam == "Huber":
        return f"epsilon={x}, alpha={a:.3g}"
    return f"alpha={a:.3g}"


def fit_single(A, ytr, key):
    """Fit one model on standardised A. Returns (coef, intercept)."""
    fam, a, x = key
    ym = ytr.mean()
    yc = ytr - ym
    if A.shape[1] == 0:
        return np.zeros(0), ym
    if fam == "OLS":
        return np.linalg.lstsq(A, yc, rcond=None)[0], ym
    if fam == "Ridge":
        return np.linalg.solve(A.T @ A + a * np.eye(A.shape[1]), A.T @ yc), ym
    if fam in ("Lasso", "ElasticNet"):
        return enet_path(A, yc, l1_ratio=x, alphas=[a], max_iter=10000)[1][:, 0], ym
    m = HuberRegressor(epsilon=x, alpha=a, max_iter=1000).fit(A, ytr)
    return m.coef_, m.intercept_


def fit_all(Xtr, ytr, Xva, keys, grids):
    """Validation and training predictions for every hyperparameter value in keys (paths where possible)."""
    A, B = standardize(Xtr, Xva)
    ym = ytr.mean()
    yc = ytr - ym
    if A.shape[1] == 0:
        return {k: (np.full(len(B), ym), np.full(len(A), ym)) for k in keys}
    fams = {k[0] for k in keys}
    W = {}
    if "OLS" in fams:
        W["OLS", None, None] = (np.linalg.lstsq(A, yc, rcond=None)[0], ym)
    if "Ridge" in fams:
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        Uy = U.T @ yc
        for a in grids["ridge"]:
            W["Ridge", a, None] = (Vt.T @ (s / (s ** 2 + a) * Uy), ym)
    for l1 in L1_RATIOS:
        fam = "Lasso" if l1 == 1.0 else "ElasticNet"
        if fam in fams:
            coefs = enet_path(A, yc, l1_ratio=l1, alphas=grids["enet"][l1], max_iter=5000)[1]
            for j, a in enumerate(grids["enet"][l1]):
                W[fam, a, l1] = (coefs[:, j], ym)
    for k in keys:
        if k[0] == "Huber":
            W[k] = fit_single(A, ytr, k)
    return {k: (B @ W[k][0] + W[k][1], A @ W[k][0] + W[k][1]) for k in keys}


def fold_stats(Xtr, ytr):
    """|Pearson r| of every column with y, on training rows only."""
    Z = standardize(Xtr)[0]
    Z[:, Xtr.std(0) == 0] = 0.0
    yz = (ytr - ytr.mean()) / ytr.std()
    return np.abs(Z.T @ yz) / len(ytr)


def select(s, r, base, dummy):
    if s["kind"] == "all":
        return base + dummy
    if s["kind"] == "topk":
        return sorted(np.argsort(-r, kind="stable")[: s["k"]].tolist())
    t = s["threshold"]
    keep = [c for c in base if r[c] >= t]
    return keep + {"none": [], "all": dummy, "filtered": [c for c in dummy if r[c] >= t]}[s["cat_mode"]]


def fit_config(X, y, out, setting, key, pidx, base, dummy, LOG):
    """Fit on all rows of X (full matrix incl. raw columns): outlier rule, log choice, |r| selection and scaling
    all use these training rows only."""
    keep = ~out if setting.get("drop_outliers") else np.ones(len(y), bool)
    logged = log_decision(X[keep], LOG)
    Xt, yt = apply_log(X[keep], logged, LOG)[:, pidx], y[keep]
    cols = select(setting, fold_stats(Xt, yt), base, dummy)
    mu, sd = Xt[:, cols].mean(0), Xt[:, cols].std(0)
    sd = np.where(sd > 0, sd, 1.0)
    w, b = fit_single((Xt[:, cols] - mu) / sd, yt, key)
    return {"cols": cols, "mu": mu, "sd": sd, "w": w, "b": b, "logged": logged, "pidx": pidx, "LOG": LOG}


def design(m, X):
    """Selected, log-transformed (training-row decision) model inputs for rows of X, before scaling."""
    return apply_log(X, m["logged"], m["LOG"])[:, m["pidx"]][:, m["cols"]]


def predict(m, X):
    return ((design(m, X) - m["mu"]) / m["sd"]) @ m["w"] + m["b"]


def usd(log_pred):
    return np.expm1(np.clip(log_pred, 9, 15))  # clip so a wild prediction cannot overflow


def scores(p, yt):
    e = p - yt
    return [np.sqrt(np.mean(e ** 2)), np.mean(np.abs(usd(p) - np.expm1(yt))), 1 - np.sum(e ** 2) / np.sum((yt - yt.mean()) ** 2)]


# ================================================================== time-series CV engine
def cv_task(X, y, tr, va, pidx, base, dummy, is_time, outlier, work, keys, grids, fold, LOG):
    warnings.filterwarnings("ignore")
    cache, out = {}, []
    for sid, s in work:
        drop = bool(s.get("drop_outliers"))
        trk = tr[~outlier[tr]] if drop else tr
        if drop not in cache:  # log choice and |r| ranking from this fold's training rows only
            Xf = apply_log(X, log_decision(X[trk], LOG), LOG)[:, pidx]
            cache[drop] = (Xf, fold_stats(Xf[trk], y[trk]))
        Xp, r = cache[drop]
        cols = select(s, r, base, dummy)
        P = fit_all(Xp[trk][:, cols], y[trk], Xp[va][:, cols], keys, grids)
        M = np.array([[*scores(pv, y[va]), np.sqrt(np.mean((pt - y[trk]) ** 2))] for pv, pt in (P[k] for k in keys)])
        out.append((sid, fold, len(cols), bool(is_time[cols].any()) if cols else False, M))
    return out


def run_cv(D, pool, settings, families, chunk=1):
    """Every selection setting x hyperparameter value on every time-series fold."""
    pidx, base, dummy, is_time = pool_parts(D, pool)
    keys = param_keys(families, D["grids"])
    work = list(enumerate(settings))
    tasks = [delayed(cv_task)(D["Xtr"], D["ytr"], tr, va, pidx, base, dummy, is_time, D["outlier_tr"],
                              work[i:i + chunk], keys, D["grids"], k, D["log"])
             for k, (tr, va) in enumerate(D["splits"]) for i in range(0, len(work), chunk)]
    S, P = len(settings), len(keys)
    R = np.full((S, N_SPLITS, P, 4), np.nan)  # rmse, mae_usd, r2, train_rmse
    NF = np.zeros((S, N_SPLITS))
    UT = np.zeros(S, bool)
    for part in Parallel(n_jobs=N_JOBS)(tasks):
        for sid, k, nf, ut, M in part:
            R[sid, k], NF[sid, k] = M, nf
            UT[sid] |= ut
    sdf = pd.DataFrame([{"sid": i, "kind": st["kind"], "threshold": st.get("threshold", np.nan),
                         "cat_mode": st.get("cat_mode", "-"), "drop_outliers": bool(st.get("drop_outliers")),
                         "k": st.get("k", np.nan)} for i, st in enumerate(settings)])
    kdf = pd.DataFrame([{"pid": j, "model": k[0], "alpha": np.nan if k[1] is None else k[1],
                         "extra": np.nan if k[2] is None else k[2], "params": key_label(k)} for j, k in enumerate(keys)])
    df = sdf.merge(kdf, how="cross")
    si, pi = df["sid"].to_numpy(), df["pid"].to_numpy()
    mean, sd = np.nanmean(R, axis=1), np.nanstd(R, axis=1, ddof=1)
    for j, m in enumerate(["rmse", "mae_usd", "r2", "train_rmse"]):
        df[f"cv_{m}"] = mean[si, pi, j]
    df["cv_rmse_sd"], df["cv_mae_usd_sd"] = sd[si, pi, 0], sd[si, pi, 1]
    df["n_feats"] = NF.mean(1)[si]
    df["uses_time"] = UT[si]
    return {"agg": df.sort_values("cv_rmse").reset_index(drop=True), "R": R, "keys": keys, "settings": settings}


def to_jsonable_key(key):
    return [key[0], None if key[1] is None else float(key[1]), None if key[2] is None else float(key[2])]


def from_json_key(k):
    return (k[0], k[1], k[2])


# ================================================================== output helpers
def _wrap(t, inches):
    """Wrap a title / label to the width it has (big fonts would otherwise run off the figure)."""
    s = t.get_text()
    n = max(20, int(inches * 72 / (t.get_fontsize() * 0.5)))   # ponytail: average glyph width estimate
    if s and "\n" not in s and len(s) > n:
        t.set_text(textwrap.fill(s, n))


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    W = fig.get_figwidth()
    if fig._suptitle is not None:
        _wrap(fig._suptitle, W)
    for ax in fig.axes:
        w = ax.get_position().width * W
        for t in (ax.title, ax.xaxis.label):
            _wrap(t, w)
        _wrap(ax.yaxis.label, ax.get_position().height * fig.get_figheight())
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight", pad_inches=0.15)
    (FIG.parent / "figures_transparent").mkdir(exist_ok=True)
    fig.savefig(FIG.parent / "figures_transparent" / name, dpi=150, bbox_inches="tight", pad_inches=0.15, transparent=True)
    plt.close(fig)
    return f"figures/{name}"


def write_json(obj, name):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def read_json(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def md(df, floatfmt="{:.4f}"):
    df = df.copy()
    for c in df:
        if pd.api.types.is_float_dtype(df[c]):
            name = str(c)
            if name in ("p", "KS_p") or name.startswith("p_"):
                df[c] = df[c].map(lambda v: "" if pd.isna(v) else ("<0.001" if v < 0.001 else f"{v:.3f}"))
                continue
            fmt = ("{:,.0f}" if any(k in name for k in ("usd", "$", "SalePrice", "Actual", "Pred", "Error", "GrLivArea", "TotalBsmtSF", "LotArea"))
                   else "{:g}" if name in ("threshold", "k") else "{:.1f}" if "n_feats" in name else floatfmt)
            df[c] = df[c].map(lambda v, f=fmt: "" if pd.isna(v) else ("inf" if np.isinf(v) else f.format(v)))
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v).replace("|", "/") for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)
