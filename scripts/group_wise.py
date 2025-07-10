import polars as pl


def split_data_by_group_size_test(
    df: pl.DataFrame,
    bins: list = [3, 21, 162, None],
    labels: list = ["small", "medium", "large"]
):
    """
    根據ranker_id分群，保留原始全域唯一順序 global_row_nr。
    """
    df = df.with_row_count("global_row_nr")

    # 不再 filter
    group_counts = (
        df.group_by("ranker_id")
          .agg(pl.count().alias("n_rows"))
          .filter(pl.col("n_rows") >= bins[0])
    )

    bins_fixed = bins.copy()
    if bins_fixed[-1] is None:
        max_value = group_counts["n_rows"].max()
        bins_fixed[-1] = int(max_value) + 1

    if len(labels) != len(bins_fixed) - 1:
        raise ValueError(f"bins={bins_fixed} 有 {len(bins_fixed)-1}個區間，但labels數={len(labels)}")

    cond = (
        pl.when((pl.col("n_rows") >= bins_fixed[0]) & (pl.col("n_rows") < bins_fixed[1]))
          .then(pl.lit(labels[0]))
    )
    for i in range(1, len(labels)):
        cond = cond.when(
            (pl.col("n_rows") >= bins_fixed[i]) & (pl.col("n_rows") < bins_fixed[i+1])
        ).then(pl.lit(labels[i]))
    cond = cond.otherwise(pl.lit("unknown"))

    group_counts = group_counts.with_columns([
        cond.alias("group_category")
    ])

    df = df.join(group_counts, on="ranker_id", how="left")

    split_data = {}
    for lbl in labels:
        subset = df.filter(pl.col("group_category") == lbl)
        mem_mb = subset.estimated_size() / (1024*1024)
        print(f"✅ {lbl}: {subset.height} rows, approx {mem_mb:.2f} MB")
        split_data[lbl] = subset

    summary = (
        group_counts.group_by("group_category")
        .agg([
            pl.count().alias("n_groups"),
            pl.col("n_rows").sum().alias("total_rows"),
            pl.col("n_rows").mean().alias("avg_rows_per_group")
        ])
        .sort("group_category")
    )

    print("✅ 分群統計：")
    print(summary)

    return {
        "data_all": df,
        "split_data": split_data,
        "summary": summary
    }



import numpy as np
import xgboost as xgb

def prepare_train_val_split(
    result: dict,
    split_label: str,
    feature_cols: list,
    train_fraction: float = 0.8,
    random_seed: int = 42
):
    """
    根據分群結果切 train/val 並轉成 numpy，保證不洩漏。
    
    參數:
    - result: split_data_by_group_size() 的輸出
    - split_label: 要使用的分群名稱 ("small"/"medium"/"large")
    - feature_cols: feature 欄位
    - train_fraction: train比例 (預設0.8)
    - random_seed: 隨機種子
    
    回傳:
    dict {
        X_train_np, y_train_np, groups_train_np,
        X_val_np, y_val_np, groups_val_np,
        group_sizes_train, group_sizes_val,
        val_ids: val用的ranker_id set
    }
    """
    # 取得分群 DataFrame
    df = result["split_data"][split_label]

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in ("selected", "ranker_id", "global_row_nr", "frequentFlyer")]

    # 抓所有 ranker_id
    unique_rankers = df.select("ranker_id").unique().to_series().to_list()
    np.random.seed(random_seed)
    np.random.shuffle(unique_rankers)

    # 切 train/val
    n_train = int(len(unique_rankers) * train_fraction)
    train_ids = set(unique_rankers[:n_train])
    val_ids = set(unique_rankers[n_train:])

    # 標記 train
    is_train = df.select(pl.col("ranker_id").is_in(list(train_ids)).alias("is_train"))
    exclude_cols = {"selected", "ranker_id", "global_row_nr", "frequentFlyer", "Id"}
    feature_cols = [f for f in feature_cols if f not in exclude_cols]

    # 建立 feature/label/groups DataFrame
    X_all = df.select(feature_cols)
    y_all = df.select("selected")
    groups_all = df.select("ranker_id")

    # 加入mask
    X_with_mask = X_all.with_columns(is_train)
    y_with_mask = y_all.with_columns(is_train)
    groups_with_mask = groups_all.with_columns(is_train)

    # 分train/val
    X_train_df = X_with_mask.filter(pl.col("is_train"))
    X_val_df = X_with_mask.filter(~pl.col("is_train"))
    y_train_df = y_with_mask.filter(pl.col("is_train"))
    y_val_df = y_with_mask.filter(~pl.col("is_train"))
    groups_train_df = groups_with_mask.filter(pl.col("is_train"))
    groups_val_df = groups_with_mask.filter(~pl.col("is_train"))

    # 轉 numpy
    X_train_np = X_train_df.drop("is_train").to_numpy()
    y_train_np = y_train_df.drop("is_train").to_numpy().flatten()
    groups_train_np = groups_train_df.drop("is_train").to_numpy().flatten()

    X_val_np = X_val_df.drop("is_train").to_numpy()
    y_val_np = y_val_df.drop("is_train").to_numpy().flatten()
    groups_val_np = groups_val_df.drop("is_train").to_numpy().flatten()

    # 計算 group size
    group_sizes_train = (
        pl.DataFrame({"ranker_id": groups_train_np})
        .group_by("ranker_id", maintain_order=True)
        .agg(pl.len())['len']
        .to_numpy()
    )
    group_sizes_val = (
        pl.DataFrame({"ranker_id": groups_val_np})
        .group_by("ranker_id", maintain_order=True)
        .agg(pl.len())['len']
        .to_numpy()
    )

    # 輸出
    print(f"✅ Train: {X_train_np.shape[0]} rows, {len(np.unique(groups_train_np))} groups")
    print(f"✅ Val: {X_val_np.shape[0]} rows, {len(np.unique(groups_val_np))} groups")
    dtrain = xgb.DMatrix(
        X_train_np,
        label= y_train_np,
        feature_names=feature_cols
    )
    dtrain.set_group(group_sizes_train)

    dval = xgb.DMatrix(
        X_val_np,
        label=y_val_np,
        feature_names=feature_cols

    )
    dval.set_group(group_sizes_val)

    return {
        "dtrain":dtrain,
        "dval":dval,
        "X_train_np": X_train_np,
        "y_train_np": y_train_np,
        "groups_train_np": groups_train_np,
        "X_val_np": X_val_np,
        "y_val_np": y_val_np,
        "groups_val_np": groups_val_np,
        "group_sizes_train": group_sizes_train,
        "group_sizes_val": group_sizes_val,
        "val_ids": val_ids
    }
    
def prepare_prediction(
    result: dict,
    split_label: str,
    feature_cols: list,
):
    """
    直接準備所有資料用於預測，不分train/val。
    """
    
    df = result["split_data"][split_label]
   # 只保留存在的欄，且按df順序
    final_features = [c for c in df.columns if c in feature_cols]

    # 全部 rows
    groups_np = df.select("ranker_id").to_numpy().flatten()
    rows_np = df.select("global_row_nr").to_numpy().flatten()

    group_sizes = (
        pl.DataFrame({"ranker_id": groups_np})
        .group_by("ranker_id", maintain_order=True)
        .agg(pl.len())['len']
        .to_numpy()
    )

    print(f"✅ {split_label} 分組大小: {len(group_sizes)}")
    return {
        "groups_np": groups_np,
        "global_row_np": rows_np,
        "group_sizes": group_sizes,
        "feature_cols": final_features,
    }
    
    
import os
import xgboost as xgb
import pandas as pd

def export_xgb_feature_importance(
    model_dir: str,
    label: str,
) -> pd.DataFrame:
    """
    輸出XGBoost模型的特徵重要性(不用feature_names)，
    自動從模型提取名稱並儲存CSV。

    參數:
    - model_dir: 模型資料夾
    - label: 分群名稱
    - top_n: 要顯示前幾名 (預設50)

    回傳:
    - 排序後的DataFrame
    """
    # 讀取模型
    model_path = os.path.join(model_dir, f"xgb_ranker_{label}.bin")
    booster = xgb.Booster()
    booster.load_model(model_path)
    print(f"✅ 已讀取模型 {model_path}")

    # 取重要性
    importance_types = ["weight", "gain", "cover"]
    importance_all = {}

    all_features = set()
    for imp_type in importance_types:
        imp_raw = booster.get_score(importance_type=imp_type)
        all_features.update(imp_raw.keys())
        sorted_imp = sorted(imp_raw.items(), key=lambda x: x[1], reverse=True)
        importance_all[imp_type] = sorted_imp
    all_features = booster.feature_names

    # 建完整特徵表
    df_all = pd.DataFrame({"feature": sorted(all_features)})

    # 個別DataFrame
    df_weight = pd.DataFrame(importance_all["weight"], columns=["feature", "weight"])
    df_weight["weight_rank_pos"] = df_weight.index

    df_gain = pd.DataFrame(importance_all["gain"], columns=["feature", "gain"])
    df_gain["gain_rank_pos"] = df_gain.index

    df_cover = pd.DataFrame(importance_all["cover"], columns=["feature", "cover"])
    df_cover["cover_rank_pos"] = df_cover.index

    # 合併
    df_merged = (
        df_all
        .merge(df_weight, on="feature", how="left")
        .merge(df_gain, on="feature", how="left")
        .merge(df_cover, on="feature", how="left")
    )

    # 補rank_pos
    df_merged["weight_rank_pos"] = df_merged["weight_rank_pos"].fillna(9999)
    df_merged["gain_rank_pos"] = df_merged["gain_rank_pos"].fillna(9999)
    df_merged["cover_rank_pos"] = df_merged["cover_rank_pos"].fillna(9999)

    # 最小排名
    df_merged["min_rank"] = df_merged[["weight_rank_pos", "gain_rank_pos", "cover_rank_pos"]].min(axis=1)

    # 排序
    df_merged_sorted = df_merged.sort_values("min_rank")

    # 建立輸出資料夾
    model_importance_dir = os.path.join(model_dir, "model_importance")
    os.makedirs(model_importance_dir, exist_ok=True)

    # 輸出CSV
    csv_path = os.path.join(model_importance_dir, f"feature_importance_{label}_all_features.csv")
    df_merged_sorted.to_csv(csv_path, index=False)
    print(f"✅ 已輸出所有特徵重要性到 {csv_path}")

    return df_merged_sorted


import os
import pandas as pd

def export_common_feature_ranks(
    labels: list,
    model_importance_dir: str
) -> pd.DataFrame:
    """
    讀取多個模型的特徵重要性CSV，找共同特徵，
    計算平均rank/最大rank/最小rank，輸出彙總表。

    參數:
    - labels: ["small", "medium", ...]
    - model_importance_dir: 模型重要性CSV目錄
    - top_n: 要顯示的Top N (預設50)

    回傳:
    - 排序後DataFrame
    """
    dfs = {}
    for label in labels:
        csv_path = os.path.join(model_importance_dir, f"feature_importance_{label}_all_features.csv")
        df = pd.read_csv(csv_path)
        dfs[label] = df[["feature", "min_rank"]].copy()
        dfs[label].rename(columns={"min_rank": f"min_rank_{label}"}, inplace=True)
        print(f"✅ 已讀 {label}: {len(df)} rows")

    # 依序 inner merge
    df_merged = dfs[labels[0]]
    for label in labels[1:]:
        df_merged = df_merged.merge(dfs[label], on="feature", how="inner")

    print(f"\n🎯 所有模型共同出現特徵: {len(df_merged)}")

    # 計算綜合排名
    rank_cols = [f"min_rank_{label}" for label in labels]
    df_merged["avg_rank"] = df_merged[rank_cols].mean(axis=1)
    df_merged["max_rank"] = df_merged[rank_cols].max(axis=1)
    df_merged["min_rank_overall"] = df_merged[rank_cols].min(axis=1)

    # 排序
    df_sorted = df_merged.sort_values("avg_rank")


    # 輸出CSV
    csv_common = os.path.join(model_importance_dir, "common_features_with_ranks.csv")
    df_sorted.to_csv(csv_common, index=False)
    print(f"\n✅ 已輸出共同特徵到 {csv_common}")

    return df_sorted


import os
import pandas as pd

def load_used_features_from_importance_csv(
    model_importance_dir: str,
    labels: list
) -> dict:
    """
    讀取模型feature importance CSV，
    回傳每個label實際有用到的特徵list。

    參數:
    - model_importance_dir: 存放CSV的資料夾
    - labels: label list

    回傳:
    - dict {label: features list}
    """
    if labels is None:
        csv_path = os.path.join(model_importance_dir, f"feature_importance.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到檔案: {csv_path}")

        df = pd.read_csv(csv_path)

        # 挑出min_rank < 9999代表有用到的feature
        result = df["feature"].tolist()

        print(f"✅ 共 {len(result)} 個用到的特徵")
    else:
        result = {}

        for label in labels:
            csv_path = os.path.join(model_importance_dir, f"feature_importance_{label}_all_features.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"找不到檔案: {csv_path}")

            df = pd.read_csv(csv_path)

            # 挑出min_rank < 9999代表有用到的feature
            used_features = df["feature"].tolist()

            print(f"✅ {label}: 共 {len(used_features)} 個用到的特徵")

            result[label] = used_features

    return result
