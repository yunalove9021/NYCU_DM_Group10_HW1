"""Step 2: refit the best configuration on train, evaluate once on the time-split test set,
coefficients with 95% CI + p-values (bootstrap for penalised models, HC3 t-test for OLS), VIF / GVIF.

Outputs: results/test_metrics.json, results/test_predictions.csv, results/coefficients.csv, results/gvif.csv,
         figures F6 (forest plot), F7 (residual diagnostics).
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

from common import (BLUE, INK, MUTED, N_BOOT, N_JOBS, NAVY, OUT, RED, SEED, SURFACE, design, fit_config, from_json_key,
                    load_data, pool_parts, predict, read_json, save, scores, stars, usd, write_json)

PRACTICAL = 0.01  # effect < 1% of the median price = practically negligible
EXPECTED = {**{f: +1 for f in [
    "OverallQual", "OverallCond", "GrLivArea", "TotalArea", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LotArea",
    "LotFrontage", "GarageCars", "GarageArea", "FullBath", "HalfBath", "BsmtFullBath", "TotalBaths", "Fireplaces",
    "HasFireplace", "FireplaceQu", "KitchenQual", "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure",
    "HeatingQC", "GarageFinish", "GarageQual", "GarageCond", "Qual_x_Area", "TotRmsAbvGrd", "BedroomAbvGr",
    "MasVnrArea", "WoodDeckSF", "OpenPorchSF", "TotalPorchArea", "ScreenPorch", "PoolArea", "HasPool", "PoolQC",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "HasBasement", "HasGarage", "IsRemod", "YearBuilt", "YearRemodAdd",
    "GarageYrBlt", "AvgRoomSize", "TotalQualCond"]},
    **{f: -1 for f in ["HouseAge", "RemodAge", "GarageAge", "KitchenAbvGr", "LowQualFinSF", "EnclosedPorch"]}}


def boot_task(X, y, out, setting, key, pidx, base, dummy, LOG, seed, p):
    warnings.filterwarnings("ignore")
    idx = np.random.default_rng(seed).integers(0, len(y), len(y))
    m = fit_config(X[idx], y[idx], out[idx], setting, key, pidx, base, dummy, LOG)
    v = np.zeros(p)
    v[m["cols"]] = m["w"]
    return v


def vif_all(Z):
    """VIF_j = 1 / (1 - R^2_j) of column j regressed on all others (Z centred). inf = exact linear identity."""
    out = np.empty(Z.shape[1])
    for j in range(Z.shape[1]):
        others = np.delete(Z, j, axis=1)
        w = np.linalg.lstsq(others, Z[:, j], rcond=None)[0]
        rss, tss = np.sum((Z[:, j] - others @ w) ** 2), np.sum(Z[:, j] ** 2)
        out[j] = np.inf if rss <= 1e-9 * tss else tss / rss
    return out


def collinearity(D, names, Zdev):
    """Reference-coded VIF per feature + GVIF per categorical (Fox & Monette 1992)."""
    sd = Zdev.std(0)
    by_root = {}
    for c in names:
        if c in D["dummy_set"]:
            by_root.setdefault(D["root_of"][c], []).append(c)
    levels = pd.Series(D["root_of"]).value_counts()
    means = dict(zip(names, Zdev.mean(0)))
    ref = {max(cs, key=lambda c: means[c]) for root, cs in by_root.items() if len(cs) == levels[root]}
    use = [i for i, c in enumerate(names) if c not in ref and sd[i] > 0]
    Z = (Zdev[:, use] - Zdev[:, use].mean(0)) / Zdev[:, use].std(0)
    v = vif_all(Z)
    vif = pd.Series(np.nan, index=names, dtype=float)
    vif.iloc[use] = v
    finite = [j for j in range(len(use)) if np.isfinite(v[j])]
    Rm = np.corrcoef(Z[:, finite], rowvar=False)
    used = [names[use[j]] for j in finite]
    _, logdet = np.linalg.slogdet(Rm)
    rows = []
    for root, cs in by_root.items():
        g = [i for i, c in enumerate(used) if c in cs]
        if not g:
            continue
        o = [i for i in range(len(used)) if i not in g]
        s1, l1 = np.linalg.slogdet(Rm[np.ix_(g, g)])
        s2, l2 = np.linalg.slogdet(Rm[np.ix_(o, o)])
        gv = np.exp(l1 + l2 - logdet) if s1 > 0 and s2 > 0 else np.inf
        rows.append({"categorical": root, "df": len(g), "GVIF": gv, "GVIF^(1/(2df))": gv ** (1 / (2 * len(g))),
                     "reference_level": next((c for c in cs if c in ref), "(not all levels in model)")})
    return vif, pd.DataFrame(rows).sort_values("GVIF^(1/(2df))", ascending=False), ref


def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    D, cfg = load_data(), read_json("best_config.json")
    setting, key = cfg["setting"], from_json_key(cfg["key"])
    pidx, base, dummy, _ = pool_parts(D, D["feats"])
    Xtr, Xte, ytr, yte = D["Xtr"], D["Xte"], D["ytr"], D["yte"]
    final = fit_config(Xtr, ytr, D["outlier_tr"], setting, key, pidx, base, dummy, D["log"])
    ph = predict(final, Xte)

    # ---------------- test metrics
    resid = yte - ph
    act, pred = np.expm1(yte), usd(ph)
    err = act - pred
    rng = np.random.default_rng(SEED)
    bs = [rng.integers(0, len(yte), len(yte)) for _ in range(N_BOOT)]
    small = D["raw_te"]["GrLivArea"].to_numpy() <= 4000
    bp = het_breuschpagan(resid, sm.add_constant(ph))
    month_bias = pd.Series(resid).groupby(D["raw_te"]["date"].dt.to_period("M").astype(str).values).mean()
    metrics = {
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "rmse_ci": np.percentile([np.sqrt(np.mean(resid[i] ** 2)) for i in bs], [2.5, 97.5]).tolist(),
        "mae": float(np.mean(np.abs(err))), "mae_ci": np.percentile([np.mean(np.abs(err[i])) for i in bs], [2.5, 97.5]).tolist(),
        "medae": float(np.median(np.abs(err))), "mape": float(np.mean(np.abs(err) / act) * 100), "r2": scores(ph, yte)[2],
        "mean_resid": float(resid.mean()), "mean_resid_p": float(stats.ttest_1samp(resid, 0).pvalue),
        "bp_p": float(bp[1]), "dw": float(durbin_watson(resid)),
        "rmse_scope": float(np.sqrt(np.mean(resid[small] ** 2))), "n_big": int((~small).sum()),
        "n_train_used": int((~D["outlier_tr"]).sum() if setting.get("drop_outliers") else len(ytr)),
        "n_feats": len(final["cols"]), "n_nonzero": int(np.sum(final["w"] != 0)),
        "test_period": f"{D['raw_te']['date'].min():%Y-%m}..{D['raw_te']['date'].max():%Y-%m}",
        "train_period": f"{D['raw_tr']['date'].min():%Y-%m}..{D['raw_tr']['date'].max():%Y-%m}",
        "month_bias": month_bias.to_dict(),
    }
    write_json(metrics, "test_metrics.json")
    pd.DataFrame({"Id": D["id_te"], "date": D["raw_te"]["date"], "actual_log": yte, "pred_log": ph,
                  "actual_usd": act, "pred_usd": pred}).to_csv(OUT / "test_predictions.csv", index=False)

    # ---------------- coefficients & significance
    names = [D["feats"][i] for i in final["cols"]]
    Z = design(final, Xtr)
    if key[0] == "OLS":
        keep = ~D["outlier_tr"] if setting.get("drop_outliers") else np.ones(len(ytr), bool)
        fit = sm.OLS(ytr[keep], sm.add_constant((Z[keep] - final["mu"]) / final["sd"])).fit(cov_type="HC3")
        ci = np.asarray(fit.conf_int())[1:]
        tab = pd.DataFrame({"feature": names, "beta": fit.params[1:], "ci_lo": ci[:, 0], "ci_hi": ci[:, 1], "p": fit.pvalues[1:], "sel_freq": 1.0})
        method = "HC3 heteroskedasticity-robust t-test (OLS)"
    else:
        B = np.array(Parallel(n_jobs=N_JOBS)(delayed(boot_task)(Xtr, ytr, D["outlier_tr"], setting, key, pidx, base, dummy, D["log"], SEED + 1 + b, len(D["feats"]))
                                             for b in range(N_BOOT)))[:, final["cols"]]
        tab = pd.DataFrame({"feature": names, "beta": final["w"], "ci_lo": np.percentile(B, 2.5, axis=0),
                            "ci_hi": np.percentile(B, 97.5, axis=0),
                            "p": np.minimum(1.0, 2 * np.minimum((B <= 0).mean(0), (B >= 0).mean(0))), "sel_freq": (B != 0).mean(0)})
        method = f"nonparametric bootstrap (B = {N_BOOT}; feature selection and fit repeated in every resample)"
    vif, gvif, ref = collinearity(D, names, Z)
    tab["VIF"] = tab["feature"].map(vif)
    tab["marginal_r"] = [np.corrcoef(Z[:, j], ytr)[0, 1] if Z[:, j].std() > 0 else 0.0 for j in range(len(names))]
    C = np.nan_to_num(np.abs(np.corrcoef(Z, rowvar=False)))
    np.fill_diagonal(C, 0)
    tab["top_partner"] = [names[int(np.argmax(C[j]))] for j in range(len(names))]
    tab["partner_r"] = C.max(1)
    is_bin = np.array([set(np.unique(Z[:, j])) <= {0.0, 1.0} for j in range(len(names))])
    tab["unit"] = np.where(is_bin, "0 -> 1", "+1 SD")
    dlog = np.where(is_bin, tab["beta"] / final["sd"], tab["beta"])
    med = float(np.median(np.expm1(ytr)))
    tab["effect_pct"] = np.expm1(dlog) * 100
    tab["effect_usd"] = tab["effect_pct"] / 100 * med
    tab["practical"] = np.where(tab["effect_usd"].abs() >= PRACTICAL * med, "yes", "negligible")
    tab["expected"] = tab["feature"].map(EXPECTED)

    def verdict(r):
        if pd.isna(r["expected"]):
            return "no prior hypothesis"
        if r["ci_lo"] <= 0 <= r["ci_hi"]:
            return "CI includes 0: sign not established"
        if np.sign(r["beta"]) == r["expected"]:
            return "agrees"
        why = (f"reversal: marginal r = {r['marginal_r']:+.2f} but beta = {r['beta']:+.3f} with the other features held fixed (confounding / shared variance)"
               if np.sign(r["marginal_r"]) == r["expected"] else
               f"reversal: marginal r = {r['marginal_r']:+.2f} also opposes the hypothesis (hypothesis or sample may be off)")
        if r["VIF"] >= 10:
            why += f"; collinear (VIF {r['VIF']:.0f}), mostly with {r['top_partner']} (|r| = {r['partner_r']:.2f})"
        return why

    tab["hypothesis_check"] = tab.apply(verdict, axis=1)
    tab["is_reference"] = tab["feature"].isin(ref)
    tab = tab.reindex(tab["beta"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    tab.to_csv(OUT / "coefficients.csv", index=False)
    gvif.to_csv(OUT / "gvif.csv", index=False)
    write_json({"method": method, "median_price": med, "n_ref": len(ref), "practical": PRACTICAL}, "coef_summary.json")

    # ---------------- F6 forest plot (slide format)
    show = tab.head(30)[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 9))
    hi_vif, sig = show["VIF"] >= 10, show["p"] < 0.05
    for i, r in show.iterrows():
        col = RED if hi_vif[i] else NAVY
        ax.hlines(i, r["ci_lo"], r["ci_hi"], color=col, lw=1.6)
        ax.scatter(r["beta"], i, s=40, zorder=3, color=col if sig[i] else SURFACE, edgecolor=col, lw=1.5)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(range(len(show)), show["feature"], fontsize=8)
    x_hi, x_lo = max(show["ci_hi"].max(), show["beta"].max()), min(show["ci_lo"].min(), show["beta"].min())
    span = x_hi - x_lo
    ax.set_xlim(x_lo - 0.05 * span, x_hi + 0.32 * span)
    cp, cv = x_hi + 0.12 * span, x_hi + 0.25 * span
    ax.text(cp, len(show) - 0.3, "p", ha="center", fontsize=8, fontweight="bold")
    ax.text(cv, len(show) - 0.3, "VIF", ha="center", fontsize=8, fontweight="bold")
    for i, r in show.iterrows():
        ax.text(cp, i, stars(r["p"]), ha="center", va="center", fontsize=8, color=MUTED if r["p"] >= 0.05 else INK)
        vt = "ref" if r["is_reference"] else ("inf" if np.isinf(r["VIF"]) else f"{r['VIF']:.1f}")
        ax.text(cv, i, vt, ha="center", va="center", fontsize=8, color=RED if hi_vif[i] else INK)
    rev = show[show["hypothesis_check"].str.startswith("reversal")]
    if len(rev):
        i = rev["beta"].abs().idxmax()
        r = show.loc[i]
        ax.annotate(f"marginal r = {r['marginal_r']:+.2f}, but beta = {r['beta']:+.3f}\nwhen the other features are held fixed",
                    (r["beta"], i), xytext=(r["beta"] + 0.25 * span, i - 2.5 if i > 3 else i + 2.5), fontsize=8, color=RED,
                    arrowprops={"arrowstyle": "-", "color": RED, "lw": 1})
    ax.grid(axis="y", visible=False)
    ax.set(title=f"F6. Top 30 coefficients by |beta| ({key[0]}), 95% CI ({'HC3' if key[0] == 'OLS' else 'bootstrap'})",
           xlabel="standardised coefficient (change in log price per 1 SD), 95% CI")
    fig.text(0.01, 0.005, "filled = p < 0.05, hollow = n.s.; red = VIF >= 10 (collinear, interpret with care); ref = reference level dropped for VIF",
             fontsize=8, color=MUTED)
    save(fig, "F6_coefficients_forest.png")

    # ---------------- F7 residual diagnostics
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    lim = [0, max(act.max(), pred.max()) / 1e3 * 1.05]
    axes[0].scatter(act / 1e3, pred / 1e3, s=10, color=BLUE, alpha=0.6, lw=0)
    axes[0].plot(lim, lim, color=MUTED, lw=1)
    axes[0].set(title="Actual vs. predicted", xlabel="actual ($k)", ylabel="predicted ($k)", xlim=lim, ylim=lim)
    axes[1].scatter(ph, resid, s=10, color=BLUE, alpha=0.6, lw=0)
    axes[1].axhline(0, color=MUTED, lw=1)
    axes[1].set(title=f"Residuals vs. fitted (Breusch-Pagan p = {metrics['bp_p']:.3g})", xlabel="predicted log price", ylabel="residual y - y_hat (log)")
    (osm, osr), (slope, icpt, _) = stats.probplot(resid, dist="norm")
    axes[2].scatter(osm, osr, s=10, color=BLUE, alpha=0.6, lw=0)
    axes[2].plot(osm, slope * osm + icpt, color=MUTED, lw=1)
    axes[2].set(title="Normal Q-Q of residuals", xlabel="theoretical quantile", ylabel="residual quantile")
    fig.suptitle("F7. Residual diagnostics of the final model on the time-split test set", fontsize=11)
    save(fig, "F7_residual_diagnostics.png")
    print(f"test RMSE {metrics['rmse']:.4f}, MAE ${metrics['mae']:,.0f}; {int((tab['p'] < 0.05).sum())} significant features")


if __name__ == "__main__":
    main()
