## 📋 資料處理與特徵工程說明 (俊宏、煜為)
> **最後更新日期：** `2026/10/13`

本專案主要針對房價預測（Ames Housing Dataset 結構）進行資料清洗與特徵優化，以下為完整的資料處理流水線（Pipeline）。

---

### 0. 資料切割
*   **時間處理**：依售屋時間 `MoSold` `YrSold` 合併後進行排序。
*   **資料切割**： `80%` / `20%` 切割 train ( `1168` ), test ( `292` )。
---

### 1. ⚙️ 基本資料清洗
*   **缺失值處理**：所有數值型欄位的空值（NaN / Null）皆以 **填 `0`** 處理。
*   **標籤異常修正**：人工確認的修正對應請參考 `mapping/clean_value.json`。
    *   *註：因 `BldgType` 欄位存在 `TwnhsE`、`TwnhsI` 模糊代號，單依 `Twnhs` 無法明確判斷，此類異常值統一歸為 `null`。*
*   **欄位名稱對照**：房間數與廚房數在原始資料的實際欄位名稱為 `BedroomAbvGr` 與 `KitchenAbvGr`。

---

### 2. 🕒 時間資料處理 (Time Features)
時間欄位依據其特性的不同，區分為三種處理機制（**時間資料皆不參與 Z-score 標準化**）：

| 特徵名稱 | 資料型態 | 處理邏輯與商業意義 |
| :--- | :--- | :--- |
| **HouseAge** | `numeric` | `YrSold - YearBuilt`：計算房屋的實際年齡。 |
| **RemodAge** | `numeric` | `YrSold - YearRemodAdd`：計算距離上次翻修經過了多久。 |
| **GarageAge** | `numeric` | `YrSold - garage_yr`：計算車庫屋齡（若無車庫則填 0）。 |
| **HasGarage** | `binary` | 檢查是否有車庫，有為 `1`，無為 `0`。 |
| **IsRemod** | `binary` | 檢查 `YearRemodAdd != YearBuilt`。若不相等（記為 1）代表曾翻修過，用以捕捉「翻修溢價效果」。 |
| **sin_MoSold**<br>**cos_MoSold**| `cyclic` | 原始月份 `MoSold` (1~12) 直接投入線性模型會失去週期連續性（如 12 月與 1 月距離最遠）。故轉換為**正弦/餘弦**循環特徵，數值壓縮在 `[-1, 1]`。 |

---

### 3. 🧪 客製化特徵工程 (Feature Engineering)
為解決共線性並強化模型對房地產商業邏輯的捕捉能力，新增以下特徵轉換：

#### 💡 領域知識加總特徵 (`numeric`, 參與標準化)
*   **TotalArea**：`GrLivArea + TotalBsmtSF`（合併地上與地下總居住面積）。
*   **TotalBaths**：`FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath`（整合全套與半套衛浴總數）。
*   **TotalPorchArea**：木質露台與五種門廊休閒面積之總和。
*   **AvgRoomSize**：`GrLivArea / TotRmsAbvGrd`（計算平均房間寬敞度，衡量空間開闊感）。
*   **LotUtilization**：`GrLivArea / LotArea`（計算土地建蔽率估計，評估庭院與建物比例）。

#### ✖️ 交互作用項 (`numeric`, 參與標準化)
*   **Qual_x_Area**：`OverallQual * GrLivArea`（品質與地上居住面積交乘，捕捉豪宅溢價效應）。
*   **TotalQualCond**：`OverallQual + OverallCond`（合併建材品質與保養狀況，提供綜合房屋體質分數）。

#### 🎯 二元指標特徵 (`binary`, 不參與標準化)
*   **HasBasement**、**HasFireplace**、**HasPool**：將原始面積或數量轉換為 `0/1` 二元特徵，專注於捕捉「擁有該設施」的價格斷層。

#### 📈 偏態特徵數學轉換 (Log Transform)
*   **特徵自動化高偏態校正**：利用 `scipy.stats.skew` 掃描所有數值特徵，針對偏態係數 **\(> 0.75\)** 的高度右偏特徵，自動進行 `np.log1p` 轉換以校正分佈。
    *   ⚠️ **Data Leakage 防範**：Train set 篩選出的高偏態特徵名單會統一儲存至 `mapping/skewed_features.json`，Test set 在進行轉換時會直接讀取該名單，確保測試集依循相同的特徵轉換規則。
*   ⚠️ **重要注意（目標變數）**：訓練集 (Train set) 會針對目標變數 `SalePrice` 進行對數轉換（Log Transform）。**模型預測完畢後，務必使用 `np.expm1` 轉回真實房價。**

---

### 4. 🗂️ 編碼與標準化 (Encoding & Scaling)
*   **類別資料 (Category)**：轉換為 One-hot 矩陣（以 `binary` 表示），原始值對應請參考 `mapping/categories_mapping.json`。
*   **順序資料 (Ordinal)**：因存在明確前後關係（如屋況好壞），依原始標籤轉為數值表達（以 `1` 開始，`NA` 標記為 `0`），但**不進行標準化處理**。
*   **數值資料 (Numeric)**：皆以 **Z-score** 進行標準化。
    *   ⚠️ **Data Leakage 防範**：測試集 (Test set) 必須完全使用訓練集 (Train set) 的 Scaler 參數進行轉換。

---

### 5. 初步特徵篩選 (Feature Selection)
針對所有特徵值透過簡單線性回歸後依相關係數進行篩選。
*   **類別資料 (Category)**： 依回歸模型 **p-value** < `0.05` 決定是否保留。
*   **順序資料 (Ordinal)**： 依回歸模型 **p-value** < `0.05` 決定是否保留，再依相關係數 **corr** 大於`0.7`以上和其他特徵比較，保留絕對值**estimate**較大者。
*   **數值資料 (Numeric)**： 依回歸模型 **p-value** < `0.05` 決定是否保留，再依相關係數 **corr** 大於`0.7`以上和其他特徵比較，保留絕對值**estimate**較大者。
*   **時間資料 (Time)**： 依回歸模型 **p-value** < `0.05` 決定是否保留，再依相關係數 **corr** 大於`0.7`以上和其他特徵比較，保留絕對值**estimate**較大者。
---

### 📂 資料夾架構說明

```text
├── corr/             # 相關係數資料夾（基於 Train data 標準化後的 numeric, ordinal, time, target 計算之皮爾森相關係數，僅呈現正負 0.3 以上之關係）
├── log/              # 特徵欄位異常修正相關紀錄
├── mapping/          # 資料欄位與原始標籤對應參考（含 column_types.json、categories_mapping.json、skewed_features.json 等）
├── summary/          # 數值資料分佈結果參考（標準化前後分別為獨立檔案）
├── models/           # 簡單線性回歸結果，包含貪婪特徵選擇列表。
└── process_dataset/  # 處理後的資料產出位置（區分類別資料 one_hot 表達與 index 表達，其餘欄位內容相同）
```
