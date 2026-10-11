
import os
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

def filter_features(df_corr, df_cof, threshold=0.7, p_threshold=0.05):
    df_removed_p = df_cof[
        df_cof["P_value"] > p_threshold
    ].copy()
    df = df_cof[
        df_cof["P_value"] <= p_threshold
    ].copy()
    df = df.reindex(
        df["Estimate"].abs().sort_values(ascending=False).index
    )
    selected = []
    removed_corr = []
    for _, row in df.iterrows():
        feature = row["Feature"]
        if feature not in df_corr.index or feature not in df_corr.columns:
            selected.append(feature)
            continue
        conflicts = []
        for kept in selected:
            if kept not in df_corr.index or kept not in df_corr.columns:
                continue
            corr = df_corr.loc[feature, kept]
            if pd.notna(corr) and abs(corr) > threshold:
                conflicts.append((kept, corr))
        if conflicts:
            kept, corr = max(conflicts, key=lambda x: abs(x[1]))
            removed_corr.append({
                "Feature": feature,
                "Estimate": row["Estimate"],
                "Removed_by": kept,
                "Corr": corr,
                "Abs_Corr": abs(corr)
            })
        else:
            selected.append(feature)
    df_selected = df[df["Feature"].isin(selected)].copy()
    df_removed_corr = pd.DataFrame(
        removed_corr,
        columns=["Feature", "Estimate", "Removed_by", "Corr", "Abs_Corr"]
    )
    df_selected.reset_index(inplace = True, drop = True)
    df_removed_p.reset_index(inplace = True, drop = True)
    df_removed_corr.reset_index(inplace = True, drop = True)
    return df_selected, df_removed_p, df_removed_corr

def linear_regression(
    X_train,
    X_test,
    y_train,
    y_test,
    model_path="./model/linear_regression.pkl"
):
    model = LinearRegression()
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    def evaluate(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R2": r2_score(y_true, y_pred)
        }
    metrics = pd.DataFrame({
        "Train": evaluate(y_train, train_pred),
        "Test": evaluate(y_test, test_pred)
    })
    metrics.to_excel(model_path.replace(".pkl","_metrics.xlsx"))
    X_train_sm = sm.add_constant(
        X_train, has_constant="add"
    )
    ols_model = sm.OLS(
        np.asarray(y_train).ravel(),
        np.asarray(X_train_sm)
    ).fit()
    feature_names = (
        list(X_train.columns)
        if isinstance(X_train, pd.DataFrame)
        else [f"X{i}" for i in range(X_train.shape[1])]
    )
    coefficient_table = pd.DataFrame({
        "Feature": ["Intercept"] + feature_names,
        "Estimate": ols_model.params,
        "Std_Error": ols_model.bse,
        "t_value": ols_model.tvalues,
        "P_value": ols_model.pvalues
    })
    coefficient_table.to_excel(model_path.replace(".pkl","_coefficient_table.xlsx"))
    model_data = {
        "model": model,
        "features": feature_names,
        "coefficient_table": coefficient_table
    }

    model_dir = os.path.dirname(
        os.path.abspath(model_path)
    )
    os.makedirs(model_dir, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
        
    print("=== Model Evaluation ===")
    print(metrics.round(4))

    print("\n=== Coefficient Table ===")
    print(coefficient_table.round(4))

    print(f"\nModel saved: {model_path}")

    return model, metrics, coefficient_table

def load_linear_regression(model_path="linear_regression.pkl"):
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    return model_data

def predict_linear_regression(
    X_new,
    model_path="linear_regression.pkl"
):
    model_data = load_linear_regression(model_path)

    model = model_data["model"]
    features = model_data["features"]

    if isinstance(X_new, pd.DataFrame):
        X_new = X_new[features]

    predictions = model.predict(X_new)

    return predictions

def check_null(df):
    null_cols = df.columns[df.isnull().any()]
    null_df = df.loc[df[null_cols].isnull().any(axis=1), ["Id", *null_cols]]
    print(null_df)
    return None

def main():
    Train = pd.read_csv("./process_dataset/one_hot_train.csv", index_col= 0)
    Test = pd.read_csv("./process_dataset/one_hot_test.csv", index_col = 0)
    X_train = Train.drop(columns=["Id","SalePrice"])
    X_test = Test.drop(columns=["Id","SalePrice"])
    y_train = Train["SalePrice"]
    y_test = Test["SalePrice"]
    print("Check Null Train...")
    check_null(Train)
    print("Check Null Test...")
    check_null(Test)
    model, metrics, coef_table = linear_regression(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model_path="models/linear_regression.pkl"
    )
    print("Greedy Feature select....")
    df_cof = pd.read_excel("./models/linear_regression_coefficient_table.xlsx", index_col= 0)
    df_corr = pd.read_excel("./corr/correlation.xlsx", index_col= 0 )
    df_selected, df_removed_p, df_removed_corr = filter_features(df_corr, df_cof, threshold=0.7)
    df_selected.to_excel("./models/greedy_select_feature.xlsx")
    df_removed_p.to_excel("./models/remove_feature_by_p.xlsx")
    df_removed_corr.to_excel("./models/remove_feature_by_corr.xlsx")
    print("Greedy Feature select Finish...")

if __name__ == '__main__':
    main()