import polars as pl


def compute_hitrate_at_3(groups_np, labels_np, preds_np,top =3, min_group_size=10, max_group_size=None):
    """
    計算 HitRate@3，可自訂 group size 範圍
    參數:
        groups_np: np.array of ranker_id
        labels_np: np.array of true labels (0/1)
        preds_np: np.array of predicted scores
        min_group_size: int, 最小 group size (預設=10)
        max_group_size: int or None, 最大 group size (預設=None，不限)
    回傳:
        hitrate: float
    """
    # 建立 DataFrame
    val_df = pl.DataFrame({
        "ranker_id": groups_np,
        "label": labels_np,
        "score": preds_np
    })

    # 排序
    val_df = val_df.sort(["ranker_id", "score"], descending=[False, True])

    # rank_in_group: 分數最高 rank=1
    val_df = val_df.with_columns([
        pl.col("score").rank(method="ordinal", descending=True).over("ranker_id").alias("rank_in_group")
    ])

    # 計算每組大小
    group_size_df = (
        val_df.group_by("ranker_id")
              .agg(pl.len().alias("group_size"))
    )

    # 對每組找 label=1 的 row
    hit_df = (
        val_df.filter(pl.col("label") == 1)
              .join(group_size_df, on="ranker_id")
              .with_columns([
                  (pl.col("rank_in_group") <=top).cast(pl.Int8).alias("is_hit")
              ])
    )

    # 篩選 group size
    cond = pl.col("group_size") >= min_group_size
    if max_group_size is not None:
        cond = cond & (pl.col("group_size") <= max_group_size)
    hit_filtered = hit_df.filter(cond)

    # 計算
    num_groups = hit_filtered.height
    num_hits = hit_filtered["is_hit"].sum()
    if num_groups > 0:
        hitrate = num_hits / num_groups
    else:
        hitrate = 0.0

    # 輸出
    print(f"✅ HitRate@3 (groups size in [{min_group_size}, {max_group_size or 'inf'}]): {hitrate:.4f}")

    return hitrate