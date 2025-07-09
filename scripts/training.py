import os
import pandas as pd

def load_label_features(
    model_dir: str,
    split_labels: list,
    top_n: int = None
):
    """
    從模型資料夾讀取每個分群的特徵重要性檔案，回傳每個label的features list。
    
    參數:
    - model_dir: 模型資料夾
    - split_labels: 分群名稱list
    - top_n: 如果指定，只取前N個特徵；否則用第一個min_rank=9999為止

    回傳:
    - dict(label -> features list)
    """
    label_features = {}

    for label in split_labels:
        model_importance_dir = os.path.join(model_dir, "model_importance")
        csv_path = os.path.join(model_importance_dir, f"feature_importance_{label}_all_features.csv")

        df = pd.read_csv(csv_path)

        if top_n is not None:
            # 直接取前top_n
            feats = df.iloc[:top_n]["feature"].tolist()
            print(f"\n✅ {label}: 取前 {top_n} 個特徵")
        else:
            idx_first_unused = df[df["min_rank"] == 9999].index.min()
            feats = df.iloc[:idx_first_unused]["feature"].tolist()
            print(f"\n✅ {label}: 第 {idx_first_unused} 名後都是完全未使用的特徵")
            print("✅ 第一個未使用特徵：")
            print(df.iloc[idx_first_unused])

        label_features[label] = feats

    # 印出所有分群features數量
    for label in split_labels:
        print(f"{label}: {len(label_features[label])} features")

    return label_features


import polars as pl

def split_data_by_group_size(
    df: pl.DataFrame,
    bins: list,
    labels: list,
    label_features: dict = None
):
    """
    一次切分所有分群，每個分群用各自features。
    
    df: polars.DataFrame
    bins: 分群邊界
    labels: 分群名稱
    label_features: dict(label -> feature list)，每個label自己的feature欄位。
                   如果為None，則所有分群都用df的所有columns。
    """
    df = df.with_row_count("global_row_nr")

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

        # 如果沒給 label_features，就用全部 columns
        if label_features is None:
            feats = [c for c in df.columns if c not in ("group_category")]
        else:
            feats = label_features.get(lbl, [])

        base_cols = ["selected", "ranker_id", "global_row_nr", "group_category"]
        all_cols = feats + base_cols
        all_cols = list(dict.fromkeys(all_cols))  # 去重防止重複

        subset = subset.select([c for c in all_cols if c in subset.columns])

        mem_mb = subset.estimated_size() / (1024 * 1024)
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
