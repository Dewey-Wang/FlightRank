import polars as pl
import xgboost as xgb

def predict_and_get_topk(
    xgb_model: xgb.Booster,
    data: pl.DataFrame,
    topk: int
) -> pl.DataFrame:
    """
    輸入 xgb 模型與資料，預測並標記每個 ranker_id Top-K。
    回傳 Id + TopK flag，並顯示 TopK 記錄數。
    """
    # 預測
    dmatrix = xgb.DMatrix(data.drop("Id", "ranker_id").to_pandas())
    preds = xgb_model.predict(dmatrix)
    del dmatrix

    # 先只保留必要欄位
    result_df = data.select([
        "Id",
        "ranker_id"
    ]).with_columns(
        pl.Series("prediction", preds)
    )

    # 排序 rank
    result_df = (
        result_df
        .with_columns(
            pl.col("prediction")
            .rank("dense", descending=True)
            .over("ranker_id")
            .alias("prediction_rank")
        )
    )

    # 標記 TopK
    result_df = result_df.with_columns(
        (pl.col("prediction_rank") <= topk).alias(f"top{topk}")
    )

    # 統計
    n_topk = result_df.filter(pl.col(f"top{topk}") == True).height
    print(f"✅ Top{topk} 數量: {n_topk}")

    # 只保留 Id 和 flag
    output_df = result_df.select([
        "Id",
        f"top{topk}"
    ])

    # 主動刪除不用的中間變數
    del result_df, preds

    return output_df
