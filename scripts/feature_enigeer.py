import os
import polars as pl

def build_price_features(
    df: pl.DataFrame,
    output_dir: str = None
) -> pl.DataFrame:
    """
    給一個 DataFrame，產生價格特徵。
    如果 output_dir 有給，會把結果存成 output_dir/price_features.parquet。

    回傳：僅包含 Id + 新增特徵
    """
    price_features = df.select([
        pl.col("Id"),
        
        (pl.col("totalPrice") / (pl.col("taxes") + 1)).alias("price_per_tax"),
        (pl.col("taxes") / (pl.col("totalPrice") + 1)).alias("tax_rate"),
        pl.col("totalPrice").log1p().alias("log_price"),

        pl.col("totalPrice")
          .rank(method="dense", descending=False)
          .over("ranker_id")
          .alias("totalPrice_rank"),

        (
            pl.col("totalPrice") == pl.col("totalPrice").min().over("ranker_id")
        ).cast(pl.Int8).alias("is_cheapest"),

        (
            (pl.col("totalPrice") - pl.col("totalPrice").median().over("ranker_id")) /
            (pl.col("totalPrice").std().over("ranker_id") + 1)
        ).alias("price_from_median_zscore"),

        (
            pl.col("totalPrice")
              .rank("average")
              .over("ranker_id")
            / pl.col("totalPrice").count().over("ranker_id")
        ).alias("price_percentile")
    ])
    
    print("✅ 已完成價格特徵工程")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "price_features.parquet")
        price_features.write_parquet(output_path)
        print(f"✅ 已儲存 Parquet: {output_path}")
    
    return price_features



import os
import polars as pl

def build_duration_features(
    df: pl.DataFrame,
    output_dir: str = None
) -> pl.DataFrame:
    """
    對 Duration/Price per Duration 做特徵工程:
    - 文字 duration 轉換成分鐘
    - total_duration
    - ranker_id 分群排名
    - price_per_duration & 排名

    如果 output_dir 給定，會輸出 duration_features.parquet
    """
    duration_cols = [
        "legs0_duration",
        "legs1_duration",
        "legs0_segments0_duration",
        "legs0_segments1_duration",
        "legs0_segments2_duration",
        "legs0_segments3_duration",
        "legs1_segments0_duration",
        "legs1_segments1_duration",
        "legs1_segments2_duration",
        "legs1_segments3_duration"
    ]

    # duration欄位轉分鐘
    duration_exprs = [
        pl.when(pl.col(c).is_in([None, "missing"]))
          .then(0)
          .otherwise(
              pl.col(c).str.extract(r"^(\d+):", 1).cast(pl.Int64) * 60 +
              pl.col(c).str.extract(r":(\d+):", 1).cast(pl.Int64)
          )
          .alias(c)
        for c in duration_cols if c in df.columns
    ]

    df = df.with_columns(duration_exprs)

    # 加總 total_duration
    if all(c in df.columns for c in ["legs0_duration", "legs1_duration"]):
        df = df.with_columns([
            (pl.col("legs0_duration") + pl.col("legs1_duration")).alias("total_duration")
        ])

    # rank表達式
    rank_exprs = [
        pl.col(c)
          .rank(method="dense", descending=False)
          .over("ranker_id")
          .cast(pl.Int32)
          .alias(f"{c}_rank")
        for c in (duration_cols + ["total_duration"]) if c in df.columns
    ]
    df = df.with_columns(rank_exprs)

    # price_per_duration
    df = df.with_columns([
        (pl.col("totalPrice") / (pl.col("total_duration") + 1)).alias("price_per_duration")
    ])

    # price_per_duration_rank
    df = df.with_columns([
        pl.col("price_per_duration")
          .rank(method="dense", descending=False)
          .over("ranker_id")
          .alias("price_per_duration_rank")
    ])

    print("✅ 已完成 Duration 特徵工程 (含排名與 price_per_duration)")

    # 只保留 Id 與新特徵
    keep_cols = ["Id"] + [
        c for c in df.columns
        if c not in ["ranker_id", "totalPrice"] and (
            c.endswith("_duration") or
            c.endswith("_rank") or
            c in ["total_duration", "price_per_duration", "price_per_duration_rank"]
        )
    ]

    duration_features = df.select(keep_cols)

    # 輸出 parquet
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "duration_features.parquet")
        duration_features.write_parquet(output_path)
        print(f"✅ 已輸出 Parquet: {output_path}")

    return duration_features
