import polars as pl
from typing import List

def compute_ranked_average_selected_from_submissions(submission_paths: List[str]) -> pl.DataFrame:
    """
    輸入多個 submission.parquet，計算 selected 平均，再依 ranker_id 排名。

    參數:
    - submission_paths: List of parquet 檔案路徑

    回傳:
    - Polars DataFrame：Id, ranker_id, __index_level_0__, selected
    """
    if not submission_paths:
        raise ValueError("請至少提供一個 submission 檔案路徑")

    dfs = []
    for i, path in enumerate(submission_paths):
        print(f"✅ 讀取: {path}")
        df = pl.read_parquet(path)
        df = df.select([
            "Id",
            "ranker_id",
            "__index_level_0__",
            pl.col("selected").alias(f"selected_{i}")
        ])
        dfs.append(df)

    # 依 Id, ranker_id, __index_level_0__ join
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.join(
            df, 
            on=["Id", "ranker_id", "__index_level_0__"],
            how="inner"
        )

    # 計算平均
    selected_cols = [f"selected_{i}" for i in range(len(dfs))]
    merged_df = merged_df.with_columns([
        pl.mean_horizontal(selected_cols).alias("selected")
    ])

    # 排名
    merged_df = merged_df.with_columns(
        pl.col("selected")
          .rank(method="ordinal", descending=True)
          .over("ranker_id")
          .alias("selected")
    )

    # 排序
    merged_df = merged_df.sort("Id")

    print(f"✅ 完成 {len(merged_df)} 筆平均+排名計算")
    return merged_df.select(["Id", "ranker_id", "selected" , "__index_level_0__"])
