"""Baselines only: Trivial / Simple / Strong baselines for Ames house prices.
 
Run:  python model/baselines.py
Outputs (in model/): baselines_results.csv, baselines_folds.csv, baselines_report.md,
                     submissions/submission_{1_trivial,2_simple,3_strong}.csv
 
- Target: log1p(SalePrice), as stored in process_dataset/one_hot_train.csv.
- Rows are sorted by sale date (YrSold, MoSold); every baseline is scored with both
  TimeSeriesSplit (expanding window) and shuffled KFold. Ranking uses TimeSeriesSplit.
- Baseline features are fixed (no correlation-threshold selection), so nothing leaks across folds.
"""
import time
from pathlib import Path
 
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
 
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
N_SPLITS = 5
SEED = 0
 
# ---------------------------------------------------------------- data
train = pd.read_csv(ROOT / "process_dataset/one_hot_train.csv", index_col=0).reset_index(drop=True)
test = pd.read_csv(ROOT / "process_dataset/one_hot_test.csv", index_col=0).reset_index(drop=True)
raw = pd.read_csv(ROOT / "Dataset/train.csv")
assert (train["Id"].values == raw["Id"].values).all(), "processed and raw train rows are not aligned"
 
order = np.lexsort((raw["Id"], raw["MoSold"], raw["YrSold"]))  # chronological
train, raw = train.iloc[order].reset_index(drop=True), raw.iloc[order].reset_index(drop=True)
 
y = train["SalePrice"].to_numpy()
X = train.drop(columns=["Id", "SalePrice"]).astype(float)
X_test = test[X.columns].astype(float)
 
raw_names = set(raw.columns) - {"Id", "SalePrice"}
# 原始欄位的重新編碼:資訊量等同原欄位,Strong baseline 應該拿得到
REENCODED = {"HouseAge", "RemodAge", "GarageAge", "sin_MoSold", "cos_MoSold"}
is_raw = lambda c: c in raw_names or c in REENCODED or any(c.startswith(k + "_") for k in raw_names)
RAW_COLS = [c for c in X if is_raw(c)]
SALE_DATE = raw["YrSold"].astype(str) + "-" + raw["MoSold"].map("{:02d}".format)
 
SPLITS = {
    "timeseries": list(TimeSeriesSplit(N_SPLITS).split(X)),
    "kfold": list(KFold(N_SPLITS, shuffle=True, random_state=SEED).split(X)),
}
 
 
# ---------------------------------------------------------------- baselines
def _bl(tier, name, model, params, fixed):
    return {"tier": tier, "baseline": name, "model": model, "params": params, "fixed": fixed}
 
 
BASELINES = [
    # 1 Trivial
    _bl("1_trivial", "Predict mean", "Mean", {}, ["OverallQual"]),
    # 2 Simple:兩個最直覺的單變數
    _bl("2_simple", "OLS on OverallQual only", "OLS", {}, ["OverallQual"]),
    _bl("2_simple", "OLS on GrLivArea only", "OLS", {}, ["GrLivArea"]),
    # 3 Strong:原始欄位 + 非線性模型,小範圍調參
    *[_bl("3_strong", f"RF raw (mf={mf}, leaf={l})", "RF", {"max_features": mf, "min_samples_leaf": l}, RAW_COLS)
      for mf in [0.3, 0.5] for l in [1, 3]],
    *[_bl("3_strong", f"HGB raw (lr={lr}, leaves={lv}, l2={l2})", "HGB", {"lr": lr, "leaves": lv, "l2": l2}, RAW_COLS)
      for lr in [0.03, 0.06] for lv in [8, 15] for l2 in [0.0, 1.0]],
]
 
 
def make_model(cfg):
    p = cfg["params"]
    est = {
        "Mean": lambda: DummyRegressor(),
        "OLS": lambda: LinearRegression(),
        "RF": lambda: RandomForestRegressor(
            n_estimators=1000, max_features=p.get("max_features", 1.0),
            min_samples_leaf=p.get("min_samples_leaf", 1), n_jobs=1, random_state=SEED),
        "HGB": lambda: HistGradientBoostingRegressor(
            max_iter=1000, learning_rate=p["lr"], max_leaf_nodes=p["leaves"],
            l2_regularization=p.get("l2", 0.0), random_state=SEED),
    }[cfg["model"]]()
    return make_pipeline(StandardScaler(), est)
 
 
def usd(log_pred):
    # clip only for the dollar metric so a wild log prediction cannot overflow expm1
    return np.expm1(np.clip(log_pred, y.min() - 1, y.max() + 1))
 
 
def run(cfg, scheme):
    feats, rows = cfg["fixed"], []
    for k, (tr, va) in enumerate(SPLITS[scheme]):
        m = make_model(cfg).fit(X.iloc[tr][feats], y[tr])
        p, pt = m.predict(X.iloc[va][feats]), m.predict(X.iloc[tr][feats])
        rows.append({
            "scheme": scheme, "fold": k, "n_feats": len(feats),
            "rmse": np.sqrt(np.mean((p - y[va]) ** 2)),
            "train_rmse": np.sqrt(np.mean((pt - y[tr]) ** 2)),
            "mae_usd": np.mean(np.abs(usd(p) - np.expm1(y[va]))),
            "r2": 1 - np.sum((p - y[va]) ** 2) / np.sum((y[va] - y[va].mean()) ** 2),
            "val_period": f"{SALE_DATE[va[0]]}..{SALE_DATE[va[-1]]}",
        })
    return rows
 
 
def run_cfg(i, cfg):
    return [{"cfg_id": i, **row} for scheme in SPLITS for row in run(cfg, scheme)]
 
 
def md(df, floatfmt="{:.4f}"):
    df = df.copy()
    for c in df:
        if pd.api.types.is_float_dtype(df[c]):
            fmt = "{:,.0f}" if "usd" in str(c) else floatfmt
            df[c] = df[c].map(lambda v, f=fmt: "" if pd.isna(v) else f.format(v))
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)
 
 
# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    folds = pd.DataFrame([r for part in Parallel(n_jobs=-1)(delayed(run_cfg)(i, c) for i, c in enumerate(BASELINES))
                          for r in part])
    print(f"{len(BASELINES)} baselines x {len(SPLITS)} CV schemes x {N_SPLITS} folds in {time.time() - t0:.0f}s")
 
    meta = pd.DataFrame([{"cfg_id": i, "tier": c["tier"], "baseline": c["baseline"], "model": c["model"]}
                         for i, c in enumerate(BASELINES)])
    agg = folds.groupby(["cfg_id", "scheme"]).agg(
        rmse=("rmse", "mean"), rmse_sd=("rmse", "std"), train_rmse=("train_rmse", "mean"),
        mae_usd=("mae_usd", "mean"), r2=("r2", "mean"), n_feats=("n_feats", "mean"),
    ).unstack("scheme")
    agg.columns = [f"{'ts' if s == 'timeseries' else 'kf'}_{m}" for m, s in agg.columns]
    res = meta.merge(agg.reset_index(), on="cfg_id").sort_values(["tier", "ts_rmse"]).reset_index(drop=True)
    res.to_csv(OUT / "baselines_results.csv", index=False)
    folds.merge(meta, on="cfg_id").to_csv(OUT / "baselines_folds.csv", index=False)
 
    # best config of each tier (by time-series CV) -> Kaggle submission
    picks = res.sort_values("ts_rmse").groupby("tier").head(1).sort_values("tier")
    sub_dir = OUT / "submissions"
    sub_dir.mkdir(exist_ok=True)
    for _, row in picks.iterrows():
        cfg = BASELINES[int(row["cfg_id"])]
        feats = cfg["fixed"]
        mdl = make_model(cfg).fit(X[feats], y)
        out = pd.DataFrame({"Id": test["Id"], "SalePrice": np.expm1(mdl.predict(X_test[feats]))})
        out.to_csv(sub_dir / f"submission_{row['tier']}.csv", index=False)
        print(row["tier"], row["baseline"], "->", f"submission_{row['tier']}.csv")
 
    cols = ["tier", "baseline", "ts_rmse", "ts_rmse_sd", "kf_rmse", "ts_mae_usd", "ts_r2", "ts_train_rmse"]
    report = f"""# Baselines (Ames House Prices)
 
Generated by `model/baselines.py`. Target is `log1p(SalePrice)`; RMSE is in log units (≈ relative error).
{len(y)} houses sold {SALE_DATE.iloc[0]} – {SALE_DATE.iloc[-1]}, scored with TimeSeriesSplit({N_SPLITS}) (`ts_`) and shuffled KFold({N_SPLITS}) (`kf_`).
 
## Best baseline per tier (ranked by time-series CV)
 
{md(picks[cols])}
 
## All baselines
 
{md(res[cols])}
"""
    (OUT / "baselines_report.md").write_text(report, encoding="utf-8")
    print(md(picks[cols]))
    print(f"done in {time.time() - t0:.0f}s -> {OUT / 'baselines_report.md'}")
 
 
if __name__ == "__main__":
    main()