import polars as pl


def split_data_by_group_size(
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
        .group_by("ranker_id")
        .agg(pl.len().alias("size"))
        .sort("ranker_id")["size"]
        .to_numpy()
    )
    group_sizes_val = (
        pl.DataFrame({"ranker_id": groups_val_np})
        .group_by("ranker_id")
        .agg(pl.len().alias("size"))
        .sort("ranker_id")["size"]
        .to_numpy()
    )

    # 輸出
    print(f"✅ Train: {X_train_np.shape[0]} rows, {len(np.unique(groups_train_np))} groups")
    print(f"✅ Val: {X_val_np.shape[0]} rows, {len(np.unique(groups_val_np))} groups")

    return {
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

    # 全部 rows
    X_np = df.select(feature_cols).to_numpy()
    y_np = df.select("selected").to_numpy().flatten()
    groups_np = df.select("ranker_id").to_numpy().flatten()
    rows_np = df.select("global_row_nr").to_numpy().flatten()

    group_sizes = (
        pl.DataFrame({"ranker_id": groups_np})
        .group_by("ranker_id")
        .agg(pl.len().alias("size"))
        .sort("ranker_id")["size"]
        .to_numpy()
    )
    print(f"✅ {split_label} 分組大小: {len(group_sizes)}")
    return {
        "X_np": X_np,
        "y_np": y_np,
        "groups_np": groups_np,
        "global_row_np": rows_np,
        "group_sizes": group_sizes,
    }