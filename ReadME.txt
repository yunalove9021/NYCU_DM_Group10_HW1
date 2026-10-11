## 俊宏、煜為  last_update:2026/10/11

資料處理流程:

1. 所有數值資料空值以填0處理

2. 時間資料處理
	HouseAge、RemodAge、GarageAge、YrSold 被定義為 "numeric"（數值型，後續會參與 Z-score 標準化）。
	HasGarage、IsRemod 被定義為 "binary"（二元特徵，值為 0 或 1，不進行標準化）。
	sin_MoSold、cos_MoSold 被定義為 "cyclic"（循環週期特徵，數值已在 [-1, 1] 之間，不進行標準化）。

	HouseAge = YrSold - YearBuilt：計算房屋的實際年齡。
	RemodAge = YrSold - YearRemodAdd：計算距離上次翻修經過了多久。
	GarageAge = YrSold - garage_yr：計算車庫的屋齡（若沒有車庫則填 0）。
	HasGarage：檢查是否有車庫，有為 1，無為 0。
	IsRemod：檢查 YearRemodAdd 是否不等於 YearBuilt。如果不相等，代表這棟房子曾經被翻修過，記為 1；否則為 0。這讓模型能捕捉「翻修過的房子是否有溢價效果」。
	----------------------------------------
	原本的月份 MoSold 是數字 1 到 12。如果直接丟給線性模型，模型會認為 12 月（12）和 1 月（1）距離最遠。
	用 sin, cos 解決。

3. 客製化特徵工程 (Feature Engineering)
	為解決共線性並強化模型對房地產商業邏輯的捕捉能力，新增以下特徵轉換：
	*   領域知識加總特徵 ("numeric", 參與標準化): 
		*   TotalArea = GrLivArea + TotalBsmtSF (合併地上與地下總居住面積)
		*   TotalBaths = FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath (整合全套與半套衛浴總數)
		*   TotalPorchArea = 木質露台與五種門廊休閒面積加總。
		*	AvgRoomSize = GrLivArea / TotRmsAbvGrd (計算平均房間寬敞度，衡量空間開闊感)。
		*	LotUtilization = GrLivArea / LotArea (計算土地建蔽率估計，評估庭院與建物比例)。
	*   交互作用項 ("numeric", 參與標準化):
		*   Qual_x_Area = OverallQual * GrLivArea (品質與地上居住面積交乘，捕捉豪宅溢價效應)。
		*	TotalQualCond = OverallQual + OverallCond (合併建材品質與保養狀況，提供綜合房屋體質分數)。
	*   二元指標特徵 ("binary", 不進行標準化):
		*   HasBasement、HasFireplace、HasPool：將原始面積或數量轉換為 0/1 特徵，專注於捕捉「擁有該設施」的價格斷層。
	*   偏態特徵數學轉換 (Log Transform):
		*   自動化高偏態校正：利用 scipy.stats.skew 掃描所有數值特徵，針對偏態係數大於 0.75 的高度右偏特徵，自動進行 np.log1p 轉換以校正分佈。
		*	注意：訓練集 (Train set) 會針對目標變數 SalePrice 進行對數轉換，模型預測完畢後需使用 np.expm1 轉回真實房價。

4. 類別資料轉one-hot表示
	原始值對應請參考 "mapping/categories_mapping.json"
	轉以 binary 表示。

5. 數值資料皆以Z-score進行標準化
	Test set 使用 Train set scaler。


資料夾說明:
corr: 相關係數資料夾，可參考各特徵相關係數計算結果
log: 特徵欄位異常修正相關紀錄
mapping: 資料欄位與原始標籤對應參考
summary: 數值資料分佈結果參考
process_dataset: 處理後資料位置，區分類別資料 one_hot表達 跟 index表達，其餘欄位內容相同。


額外註記:

1. 資料型態部分請參考: mapping/column_types.json 
   (numeric/category/ordinal/time)
2. 部分標籤異常修正，人工確認對應 請參考: mapping/clean_value.json 
   因 "BldgType" 欄位存在 "TwnhsE"、"TwnhsI" 故單依 "Twnhs" 無法判斷，歸為 "null"
3. ordinal data因存在明確關係，故依原始標籤轉為數值表達但不進標準化處理。
   以 1 開始，NA 標記為 0。
4. 時間資料不進行標準化處理。
5. 數值資料分佈狀況可在 summary 中參考，標準化前後分別為獨立檔案。
6. Bedroom、Kitchen 在欄位實際名稱以 "BedroomAbvGr"、"KitchenAbvGr" 表示。
7. 相關係數以 train data 標準化後 "numeric", "ordinal", "time", "target" 計算皮爾森相關係數，圖數值僅呈現正負 0.3 以上之關係。