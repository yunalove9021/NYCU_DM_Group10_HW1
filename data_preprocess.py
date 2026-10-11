import pandas as pd
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from openpyxl.formatting.rule import ColorScaleRule
import argparse
from scipy.stats import skew

##Load Mapping File
with open("./mapping/categories_mapping.json", "r", encoding="utf-8") as f:
    categories_mapping = json.load(f)
with open("./mapping/ordinal_categories_mapping.json", "r", encoding="utf-8") as f:
    ordinal_categories_mapping = json.load(f)
with open("./mapping/column_types.json", "r", encoding="utf-8") as f:
    column_types = json.load(f)
with open("./mapping/clean_value.json", "r", encoding="utf-8") as f:
    clean_value = json.load(f)
TIME_FEATURE_TYPES = {
    "HouseAge": "numeric",
    "RemodAge": "numeric",
    "GarageAge": "numeric",
    "YrSold": "numeric",     # 市場行情趨勢，不需要可在 time_to_features 設 keep_yrsold=False
    "HasGarage": "binary",   # 0/1 不標準化
    "IsRemod": "binary",     # 0/1 不標準化
    "sin_MoSold": "cyclic",  # 已在 [-1, 1] 不標準化
    "cos_MoSold": "cyclic"
}

def time_split(df, ratio=0.8):
    df = df.copy()
    df["Date"] = pd.to_datetime(
        dict(year=df["YrSold"], month=df["MoSold"], day=1)
    )
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.drop(columns=["Date"])
    split_idx = int(len(df) * ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df

##ord cal轉數字 1開始 0為原始空值
def ord_cal_mapping(df, json_list):
    df = df.copy()
    for col, mapping in json_list.items():
        if col not in df.columns:
            print(col)
            continue
        df[col] = df[col].replace(mapping).fillna(0)
    return df


##更新欄位名稱到原始col_type
def update_column_types(column_types, keep_yrsold=True):
    ct = {k: v for k, v in column_types.items() if v not in ("time", "target")}
    new = dict(TIME_FEATURE_TYPES)
    if not keep_yrsold:
        new.pop("YrSold")
    ct.update(new)
    ct.update({k: v for k, v in column_types.items() if v == "target"})
    return ct
#時間特徵：年份轉屋齡、月份轉 sin/cos (取代 time_to_unix)
column_types = update_column_types(column_types)

# 轉化 time 型欄位 (YearBuilt/YearRemodAdd/GarageYrBlt/SoldTime)，加入新特徵
def time_to_features(df, keep_yrsold=True, max_valid_year=2010):
    required = ["YearBuilt", "YearRemodAdd", "GarageYrBlt", "YrSold", "MoSold"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"time_to_features 缺少欄位: {missing}")
    df = df.copy()
    yr_sold = df["YrSold"]
    garage_yr = df["GarageYrBlt"]
    has_garage = garage_yr.notna()
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)#時間空值處理
    # GarageYrBlt 超過合理年份(如 test 的 2207)視為錯誤，改用 YearBuilt
    garage_yr = garage_yr.where(~(garage_yr > max_valid_year), df["YearBuilt"])
    df["HouseAge"] = (yr_sold - df["YearBuilt"]).clip(lower=0)
    df["RemodAge"] = (yr_sold - df["YearRemodAdd"]).clip(lower=0)
    df["GarageAge"] = (yr_sold - garage_yr).clip(lower=0).fillna(0)
    df["HasGarage"] = has_garage.astype("int8")
    df["IsRemod"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype("int8")
    angle = 2 * np.pi * df["MoSold"] / 12
    df["sin_MoSold"] = np.sin(angle)
    df["cos_MoSold"] = np.cos(angle)
    # drop_cols = ["YearBuilt", "YearRemodAdd", "GarageYrBlt", "MoSold"]
    # if not keep_yrsold:
    #     drop_cols.append("YrSold")
    return df

##問題值處理，對應回原始標籤 參考clean_value.json
def clean_category_values(df, clean_value, id_col="Id"):
    df = df.copy()
    replace_rules = clean_value
    logs = []
    for col, mapping in replace_rules.items():
        if col not in df.columns:
            continue
        for old_value, new_value in mapping.items():
            mask = df[col] == old_value
            count = int(mask.sum())
            if id_col in df.columns:
                ids = df.loc[mask, id_col].tolist()
            else:
                ids = df.index[mask].tolist()
            logs.append({
                "column": col,
                "old_value": old_value,
                "new_value": (
                    None
                    if pd.isna(new_value)
                    else new_value
                ),
                "replace_count": count,
                "ids": ids
            })
            df.loc[mask, col] = new_value
    log_df = pd.DataFrame(logs)
    return df, log_df

##自訂特徵工程 (加總、交乘、二元化與數學轉換)
def add_custom_features(df, is_train=False):
    df = df.copy()
    
    # 1. 領域知識加總特徵 (Domain-derived Aggregations)
    df['TotalArea'] = df.get('GrLivArea', 0) + df.get('TotalBsmtSF', 0)
    df['TotalBaths'] = df.get('FullBath', 0) + 0.5 * df.get('HalfBath', 0) + \
                       df.get('BsmtFullBath', 0) + 0.5 * df.get('BsmtHalfBath', 0)
    df['TotalPorchArea'] = df.get('WoodDeckSF', 0) + df.get('OpenPorchSF', 0) + \
                           df.get('EnclosedPorch', 0) + df.get('3SsnPorch', 0) + df.get('ScreenPorch', 0)
    # 加上極小值防呆，避免 TotRmsAbvGrd 或 LotArea 為 0 導致分母為零 (ZeroDivisionError)
    df['AvgRoomSize'] = df['GrLivArea'] / df.get('TotRmsAbvGrd', 1).replace(0, 1)
    df['LotUtilization'] = df['GrLivArea'] / df.get('LotArea', 1).replace(0, 1)
    
    # 2. 交互作用項 (Interactions)
    df['Qual_x_Area'] = df.get('OverallQual', 0) * df.get('GrLivArea', 0)
    df['TotalQualCond'] = df.get('OverallQual', 0) + df.get('OverallCond', 0)
    
    # 3. 二元指標特徵 (Binary Indicators)
    df['HasBasement'] = (df.get('TotalBsmtSF', 0) > 0).astype('int8')
    df['HasFireplace'] = (df.get('Fireplaces', 0) > 0).astype('int8')
    df['HasPool'] = (df.get('PoolArea', 0) > 0).astype('int8')
    
    # 在進行自動偏態校正前，必須先更新 column_types，讓新加總的特徵也能被納入偏態檢查
    column_types.update({
        "TotalArea": "numeric",
        "TotalBaths": "numeric",
        "TotalPorchArea": "numeric",
        "AvgRoomSize": "numeric",
        "LotUtilization": "numeric",
        "Qual_x_Area": "numeric",
        "TotalQualCond": "numeric",
        "HasBasement": "binary",
        "HasFireplace": "binary",
        "HasPool": "binary"
    })
    
    # 4. 自動化高偏態特徵校正 (Automated Skewness Fix)
    # 篩選出所有數值型特徵 (排除目標變數 SalePrice，它會在後面獨立處理)
    numeric_feats = [col for col in df.columns if column_types.get(col) == 'numeric' and col != 'SalePrice']
    
    # 計算偏態係數 (Skewness)
    skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    
    # 找出高度右偏 (偏態係數 > 0.75) 的特徵
    high_skew = skewed_feats[skewed_feats > 0.75].index
    
    # 對高偏態特徵統一進行 log1p 轉換
    for feat in high_skew:
        df[feat] = np.log1p(df[feat])
        
    # 目標變數 SalePrice 獨立轉換 (僅限訓練集)
    if is_train and 'SalePrice' in df.columns:
        df['SalePrice'] = np.log1p(df['SalePrice'])
        
    return df

##categories 資料做one hot轉換，另輸出數字標籤版
def one_hot_cols(df, json_list):
    df = df.copy()
    if "MSSubClass" in df.columns:
        df["MSSubClass"] = df["MSSubClass"].apply(lambda x: str(int(x)) if pd.notna(x) else x)

    cat_cols = [c for c in json_list if c in df.columns]
    missing = set(json_list) - set(cat_cols)
    if missing:
        raise KeyError(f"缺少類別欄位: {missing}")

    df_class = df.copy()
    parts = []
    for col in cat_cols:
        mapping = json_list[col]
        codes = df[col].map(mapping)                     # 用 map,未對應的會變 NaN
        unseen = df.loc[df[col].notna() & codes.isna(), col].unique()
        if len(unseen):
            raise ValueError(f"{col} 出現未知類別: {unseen}")
        df_class[col] = codes
        idx_to_name = {}
        for name, idx in mapping.items():
            idx_to_name.setdefault(int(idx), name)
        parts.append(pd.DataFrame(
            {f"{col}_{idx_to_name[i]}": (codes == i).astype("int8") for i in sorted(idx_to_name)},
            index=df.index))
    df_oh = pd.concat([df.drop(columns=cat_cols)] + parts, axis=1)
    df_class.reset_index(inplace = True, drop = True)
    df_oh.reset_index(inplace = True, drop = True)
    return df_class, df_oh

##數值分佈確認
def numeric_summary(df, column_types):
    numeric_cols = [
        col for col, col_type in column_types.items()
        if col_type == "numeric" and col in df.columns
    ]
    summary = pd.DataFrame({
        "min": df[numeric_cols].min(),
        "max": df[numeric_cols].max(),
        "mean": df[numeric_cols].mean(),
        "median": df[numeric_cols].median(),
        "std": df[numeric_cols].std(),
        "var": df[numeric_cols].var()
    })
    return summary

##數字空值填補 0
def fill_numeric_nan(df, column_types):
    df = df.copy()
    numeric_cols = [
        col for col, col_type in column_types.items()
        if col_type == "numeric" and col in df.columns
    ]
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df

##離群排除5%以上保留 4分位數/正負3*IQR
def filter_outliers(
    df,
    column_types,
    iqr_factor=3,
    max_remove_ratio=0.05
):
    df = df.copy()
    original_df = df.copy()
    numeric_cols = [
        col for col, col_type in column_types.items()
        if col_type == "numeric" and col in df.columns
    ]
    total_rows = len(original_df)
    outlier_info = {}
    for col in numeric_cols:
        Q1 = original_df[col].quantile(0.25)
        Q3 = original_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - iqr_factor * IQR
        upper = Q3 + iqr_factor * IQR
        outlier_mask = (
            original_df[col].notna() &
            ~original_df[col].between(lower, upper)
        )
        outlier_count = int(outlier_mask.sum())
        outlier_ratio = outlier_count / total_rows
        should_filter = outlier_ratio <= max_remove_ratio
        outlier_info[col] = {
            "outlier_count": outlier_count,
            "outlier_ratio": outlier_ratio,
            "Q1": float(Q1),
            "Q3": float(Q3),
            "IQR": float(IQR),
            "lower": float(lower),
            "upper": float(upper),
            "filtered": should_filter
        }
        if should_filter:
            df = df[
                df[col].isna() |
                df[col].between(lower, upper)
            ]
    outlier_info["_summary"] = {
        "original_rows": total_rows,
        "remaining_rows": len(df),
        "removed_rows": total_rows - len(df),
        "max_remove_ratio": max_remove_ratio
    }
    outlier_info = pd.DataFrame(outlier_info)
    return df, outlier_info

##標準化 z-score 輸出scaler
def standardize_numeric(df, column_types):
    from sklearn.preprocessing import StandardScaler
    df = df.copy()
    numeric_cols = [
        col for col, col_type in column_types.items()
        if col_type == "numeric" and col in df.columns
    ]
    scaler = StandardScaler()
    scaler_data = {
        "scaler": scaler,
        "numeric_cols": numeric_cols
    }
    df[numeric_cols] = scaler.fit_transform(
        df[numeric_cols]
    )
    return df, scaler_data

##標準化 使用train scaler
def standardize_numeric_test(df, scaler_data):
    df = df.copy()
    scaler = scaler_data["scaler"]
    numeric_cols = scaler_data["numeric_cols"]
    available_cols = [
        col for col in numeric_cols
        if col in df.columns
    ]
    if len(available_cols) != len(numeric_cols):
        missing_cols = set(numeric_cols) - set(available_cols)

        raise ValueError(
            f"Test data missing numeric columns: {missing_cols}"
        )
    df[numeric_cols] = scaler.transform(
        df[numeric_cols]
    )
    return df

##更新整理後資料欄位 消除cat改binary
def save_new_col_type(df,column_types):
    cat = {k: v for k, v in column_types.items() if v in ("category")}
    df_cols = list(df.columns)
    for cat_key in  cat.keys():
        column_types.pop(cat_key)
        results = [x for x in df_cols if cat_key in x]
        update_dict = {}
        for result in results:
            update_dict[result] = "binary"
        column_types.update(update_dict)
    df_cols.remove("Id")
    if len(set(column_types.keys()) ^ set(df_cols)) == 0:
        with open("./mapping/after_process_column_types.json", "w", encoding="utf-8") as f:
            json.dump(column_types, f, ensure_ascii=False, indent=4)
        print("col is same as data....")

##相關係數處理
def correlation_analysis(
    df,
    column_types,
    excel_path="./corr/correlation.xlsx",
    png_path="./corr/correlation.png"
):
    df = df.copy()
    use_types = {"numeric", "ordinal", "time", "target"}
    cols = [
        col
        for col, col_type in column_types.items()
        if col_type in use_types and col in df.columns
    ]
    corr_data = df[cols].apply(
        pd.to_numeric,
        errors="coerce"
    )
    corr_matrix = corr_data.corr(
        method="pearson"
    )
    type_df = pd.DataFrame({
        "column": cols,
        "type": [
            column_types[col]
            for col in cols
        ]
    })
    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl"
    ) as writer:
        corr_matrix.to_excel(
            writer,
            sheet_name="Correlation"
        )
        type_df.to_excel(
            writer,
            sheet_name="ColumnTypes",
            index=False
        )

    wb = load_workbook(excel_path)
    ws = wb["Correlation"]
    start_row = 2
    end_row = len(corr_matrix) + 1
    start_col = 2
    end_col = len(corr_matrix.columns) + 1
    from openpyxl.utils import get_column_letter
    cell_range = (
        f"{get_column_letter(start_col)}{start_row}:"
        f"{get_column_letter(end_col)}{end_row}"
    )
    color_rule = ColorScaleRule(
        start_type="num",
        start_value=-1,
        start_color="0000FF",
        mid_type="num",
        mid_value=0,
        mid_color="FFFFFF",
        end_type="num",
        end_value=1,
        end_color="FF0000"
    )
    ws.conditional_formatting.add(
        cell_range,
        color_rule
    )
    ws.column_dimensions["A"].width = 22

    for i in range(2, end_col + 1):
        ws.column_dimensions[
            get_column_letter(i)
        ].width = 12
    wb.save(excel_path)
    n = len(corr_matrix)
    size = max(10, n * 0.45)
    fig, ax = plt.subplots(
        figsize=(size, size)
    )
    im = ax.imshow(
        corr_matrix.values,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1
    )
    ax.set_xticks(
        np.arange(n)
    )
    ax.set_yticks(
        np.arange(n)
    )
    ax.set_xticklabels(
        corr_matrix.columns,
        rotation=90,
        fontsize=8
    )
    ax.set_yticklabels(
        corr_matrix.index,
        fontsize=8
    )
    for i in range(n):
        for j in range(n):
            value = corr_matrix.iloc[i, j]
            if pd.notna(value):
                if abs(value) >= 0.3 or i == j:
                    ax.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6
                    )
    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )
    cbar.set_label(
        "Pearson Correlation"
    )
    ax.set_title(
        "Correlation Matrix\n"
        "Numeric + Ordinal + Time"
    )
    plt.tight_layout()
    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    return corr_matrix

def train_process(train_data):
    train_data = fill_numeric_nan(train_data, column_types)
    train_data = time_to_features(train_data)
    train_data, log_df = clean_category_values(train_data, clean_value, id_col="Id")
    log_df.to_excel('./log/train_data_cat_clean.xlsx')
    train_data = ord_cal_mapping(train_data, ordinal_categories_mapping)
    train_data = add_custom_features(train_data, is_train=True)
    #train_data,filter_summary = filter_outliers(train_data, column_types)
    #filter_summary.to_excel('./summary/train_data_outliers_filter_summary.xlsx')
    train_summary = numeric_summary(train_data, column_types)
    train_summary.to_excel('./summary/train_data_before_stand_summary.xlsx')
    train_data, scaler = standardize_numeric(train_data, column_types)
    train_summary = numeric_summary(train_data, column_types)
    train_summary.to_excel('./summary/train_data_after_stand_summary.xlsx')
    with open('./mapping/train_scaler.pickle', "wb") as f:
        pickle.dump(scaler, f)
    save_new_col_type(train_data,column_types)
    corr_matrix = correlation_analysis(train_data, column_types)
    data_class, data_one_hot = one_hot_cols(train_data, categories_mapping)
    data_one_hot.to_csv("./process_dataset/one_hot_train.csv")
    data_class.to_csv("./process_dataset/label_train.csv")
    return None

def test_process(test_data):
    test_data = fill_numeric_nan(test_data, column_types)
    test_data = time_to_features(test_data)
    test_data, log_df = clean_category_values(test_data, clean_value, id_col="Id")
    log_df.to_excel('./log/test_data_cat_clean.xlsx')
    test_data = ord_cal_mapping(test_data, ordinal_categories_mapping)
    test_data = add_custom_features(test_data, is_train=False)
    test_summary = numeric_summary(test_data, column_types)
    test_summary.to_excel('./summary/test_data_before_stand_summary.xlsx')
    with open('./mapping/train_scaler.pickle', 'rb') as f:
        scaler_data = pickle.load(f)
    test_data = standardize_numeric_test(test_data, scaler_data)
    test_summary = numeric_summary(test_data, column_types)
    test_summary.to_excel('./summary/test_data_after_stand_summary.xlsx')
    data_class, data_one_hot = one_hot_cols(test_data, categories_mapping)
    data_one_hot.to_csv("./process_dataset/one_hot_test.csv")
    data_class.to_csv("./process_dataset/label_test.csv")
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        help="檔案位置（CSV only）"
    )
    args = parser.parse_args()
    data_path = args.data_path
    df = pd.read_csv(data_path)
    train, test = time_split(df)
    train_process(train_data = train)
    test_process(test_data = test)
