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
        output_path = os.path.join(output_dir, "1_price_features.parquet")
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
        output_path = os.path.join(output_dir, "2_duration_features.parquet")
        duration_features.write_parquet(output_path)
        print(f"✅ 已輸出 Parquet: {output_path}")

    return duration_features



import polars as pl
import os

def build_frequent_flyer_match_features(
    df: pl.DataFrame,
    output_dir: str = None,
    output_filename: str = "3_frequent_flyer_features.parquet"
) -> pl.DataFrame:
    """
    將 frequentFlyer 與各段航段 carrier_code 比對，建立下列特徵：
    - has_frequentFlyer
    - n_ff_programs
    - carrier match flags
    - matched/unmatched duration
    - 各種排名
    """
    # 0. FrequentFlyer 衍生特徵
    df = df.with_columns([
        pl.col("frequentFlyer").cast(pl.Utf8).fill_null("missing").alias("frequentFlyer")
    ])
    
    df = df.with_columns([
        (
            (pl.col("frequentFlyer") != "") &
            (pl.col("frequentFlyer") != "missing")
        ).cast(pl.Int8).alias("has_frequentFlyer"),
        (
            pl.col("frequentFlyer").map_elements(
                lambda s: 0 if s in ("", "missing") else s.count("/") + 1,
                return_dtype=pl.Int32
            ).alias("n_ff_programs")
        )
    ])

    # 1. clean frequentFlyer
    cleaned_ff = (
        pl.col("frequentFlyer")
        .fill_null("")
        .str.replace_all("missing", "")
        .str.split("/")
    )

    # 2. segments
    segments = [
        "legs0_segments0",
        "legs0_segments1",
        "legs0_segments2",
        "legs0_segments3",
        "legs1_segments0",
        "legs1_segments1",
        "legs1_segments2",
        "legs1_segments3"
    ]

    # 3. 是否 in_ff
    exprs = []
    for seg in segments:
        for carrier_type in ["marketingCarrier_code", "operatingCarrier_code"]:
            carrier_col = f"{seg}_{carrier_type}"
            exprs.append(
                pl.col(carrier_col)
                .fill_null("")
                .is_in(cleaned_ff)
                .cast(pl.Int8)
                .alias(f"{carrier_col}_in_ff")
            )

    df = df.with_columns(exprs)

    # 4. duration欄位轉分鐘
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

    # 5. total_duration
    if all(c in df.columns for c in ["legs0_duration", "legs1_duration"]):
        df = df.with_columns([
            (pl.col("legs0_duration") + pl.col("legs1_duration")).alias("total_duration")
        ])

    # 6. 累積matched duration
    legs0_matched_duration_sum = pl.sum_horizontal([
        pl.col(f"legs0_segments{i}_duration") *
        (
            pl.col(f"legs0_segments{i}_marketingCarrier_code_in_ff") |
            pl.col(f"legs0_segments{i}_operatingCarrier_code_in_ff")
        ).cast(pl.Int8)
        for i in range(4)
    ]).alias("legs0_matched_duration_sum")

    legs1_matched_duration_sum = pl.sum_horizontal([
        pl.col(f"legs1_segments{i}_duration") *
        (
            pl.col(f"legs1_segments{i}_marketingCarrier_code_in_ff") |
            pl.col(f"legs1_segments{i}_operatingCarrier_code_in_ff")
        ).cast(pl.Int8)
        for i in range(4)
    ]).alias("legs1_matched_duration_sum")

    df = df.with_columns([
        legs0_matched_duration_sum,
        legs1_matched_duration_sum,
    ])
    df = df.with_columns([

        (pl.col("legs0_matched_duration_sum") + pl.col("legs1_matched_duration_sum")).alias("all_matched_duration_sum"),
    ])
    # unmatched
    df = df.with_columns([
        (pl.col("total_duration") - pl.col("all_matched_duration_sum")).alias("unmatched_duration")
    ])

    # 7. 排名
    rank_exprs = [
        pl.col("legs0_matched_duration_sum")
          .rank(method="dense", descending=True)
          .over("ranker_id")
          .cast(pl.Int32)
          .alias("legs0_matched_duration_sum_rank"),

        pl.col("legs1_matched_duration_sum")
          .rank(method="dense", descending=True)
          .over("ranker_id")
          .cast(pl.Int32)
          .alias("legs1_matched_duration_sum_rank"),

        pl.col("unmatched_duration")
          .rank(method="dense", descending=False)
          .over("ranker_id")
          .cast(pl.Int32)
          .alias("unmatched_duration_rank")
    ]

    duration_rank_exprs = [
        pl.col(c)
          .rank(method="dense", descending=False)
          .over("ranker_id")
          .cast(pl.Int32)
          .alias(f"{c}_rank")
        for c in (duration_cols + ["total_duration"]) if c in df.columns
    ]

    df = df.with_columns(rank_exprs + duration_rank_exprs)

    # 8. 輸出
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        df.write_parquet(output_path)
        print(f"✅ 已儲存 Parquet: {output_path}")

    print("✅ 已完成 frequentFlyer 特徵 + match 特徵 + duration 特徵生成")
    return df


import os
import polars as pl

def build_baggage_fee_features(
    df: pl.DataFrame,
    output_dir: str = None,
    output_filename: str = "4_baggage_fee_features.parquet"
) -> pl.DataFrame:
    """
    建立行李與費用相關特徵:
    - baggage_total: legs0 + legs1 行李數
    - total_fees: miniRules0 + miniRules1 的費用
    - has_baggage: 是否有任何行李
    - has_fees: 是否有任何費用
    - price_per_fee: totalPrice / (total_fees + 1)
    - price_minus_fee: totalPrice - total_fees
    - groupby ranker_id 排名
    """

    # 行李總數
    baggage_total = (
        pl.col("legs0_segments0_baggageAllowance_quantity").fill_null(0) +
        pl.col("legs1_segments0_baggageAllowance_quantity").fill_null(0)
    ).alias("baggage_total")

    # 費用總額
    total_fees = (
        pl.col("miniRules0_monetaryAmount").fill_null(0) +
        pl.col("miniRules1_monetaryAmount").fill_null(0)
    ).alias("total_fees")

    # 是否有行李
    has_baggage = (
        (pl.col("baggage_total") > 0)
    ).cast(pl.Int32).alias("has_baggage")

    # 是否有費用
    has_fees = (
        (pl.col("total_fees") > 0)
    ).cast(pl.Int32).alias("has_fees")

    # totalPrice / (total_fees + 1)
    price_per_fee = (
        (pl.col("totalPrice") / (pl.col("total_fees") + 1))
        .alias("price_per_fee")
    )

    # totalPrice - total_fees
    price_minus_fee = (
        (pl.col("totalPrice") - pl.col("total_fees"))
        .alias("price_minus_fee")
    )

    # 加入主要欄位
    df = df.with_columns([
        baggage_total,
        total_fees,
    ])
    df = df.with_columns([
        has_baggage,
        has_fees,
        price_per_fee,
        price_minus_fee
    ])
    # 排名
    rank_exprs = [
        # baggage_total (數字越大 rank越低)
        pl.col("baggage_total")
          .rank(method="dense", descending=True)
          .over("ranker_id")
          .cast(pl.Int32)
          .alias("baggage_total_rank"),

        # price_per_fee (數字越大 rank越低)
        pl.col("price_per_fee")
          .rank(method="dense", descending=True)
          .over("ranker_id")
          .cast(pl.Int32)
          .alias("price_per_fee_rank"),

        # price_minus_fee (數字越大 rank越低)
        pl.col("price_minus_fee")
          .rank(method="dense", descending=True)
          .over("ranker_id")
          .cast(pl.Int32)
          .alias("price_minus_fee_rank"),

        # total_fees (數字越小 rank越低)
        pl.col("total_fees")
          .rank(method="dense", descending=False)
          .over("ranker_id")
          .cast(pl.Int32)
          .alias("total_fees_rank")
    ]

    df = df.with_columns(rank_exprs)

    # 輸出 parquet
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        df.write_parquet(output_path)
        print(f"✅ 已儲存 Parquet: {output_path}")

    return df



import os
import polars as pl

def build_cabin_features(
    df: pl.DataFrame,
    output_dir: str = None,
    output_filename: str = "5_cabin_features.parquet"
) -> pl.DataFrame:
    """
    建立艙等特徵，包含:
    - 平均艙等
    - 最長segment艙等
    - 是否最長segment艙等相同
    - 加權平均艙等
    - legs0/legs1艙等是否一致

    會先將duration欄位轉分鐘。
    """

    # 欄位
    legs0_cabin_cols = [
        "legs0_segments0_cabinClass",
        "legs0_segments1_cabinClass",
        "legs0_segments2_cabinClass",
        "legs0_segments3_cabinClass"
    ]
    legs1_cabin_cols = [
        "legs1_segments0_cabinClass",
        "legs1_segments1_cabinClass",
        "legs1_segments2_cabinClass",
        "legs1_segments3_cabinClass"
    ]
    legs0_duration_cols = [
        "legs0_segments0_duration",
        "legs0_segments1_duration",
        "legs0_segments2_duration",
        "legs0_segments3_duration"
    ]
    legs1_duration_cols = [
        "legs1_segments0_duration",
        "legs1_segments1_duration",
        "legs1_segments2_duration",
        "legs1_segments3_duration"
    ]

    duration_cols = legs0_duration_cols + legs1_duration_cols

    # Step1: duration欄位轉分鐘
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

    # Step2: cabin欄位轉Int64
    for c in legs0_cabin_cols + legs1_cabin_cols:
        df = df.with_columns(pl.col(c).cast(pl.Int64))

    # 平均艙等
    legs0_mean = (
        pl.concat_list([pl.col(c) for c in legs0_cabin_cols])
        .list.eval(pl.element().filter(pl.element() > 0))
        .list.mean()
        .fill_null(0)
        .alias("legs0_mean_cabin")
    )
    legs1_mean = (
        pl.concat_list([pl.col(c) for c in legs1_cabin_cols])
        .list.eval(pl.element().filter(pl.element() > 0))
        .list.mean()
        .fill_null(0)
        .alias("legs1_mean_cabin")
    )

    is_same_cabin = (
        (
            pl.concat_list([pl.col(c) for c in legs0_cabin_cols]).list.unique().sort()
            ==
            pl.concat_list([pl.col(c) for c in legs1_cabin_cols]).list.unique().sort()
        )
        .cast(pl.Int8)
        .alias("is_legs0_legs1_cabin_same")
    )

    # 最長segment index
    def longest_segment_idx(durations):
        if all(d is None for d in durations):
            return None
        idx = max(
            ((i, int(d) if d is not None else -1) for i, d in enumerate(durations)),
            key=lambda x: x[1]
        )[0]
        return idx

    def max_duration_cabin(row, dur_cols, cabin_cols):
        durations = [row[c] for c in dur_cols]
        cabins = [row[c] for c in cabin_cols]
        idx = longest_segment_idx(durations)
        if idx is None:
            return 0
        return cabins[idx] if cabins[idx] is not None else 0

    df = df.with_columns([
        pl.struct(legs0_duration_cols + legs0_cabin_cols)
        .map_elements(lambda row: max_duration_cabin(row, legs0_duration_cols, legs0_cabin_cols))
        .alias("legs0_max_duration_cabin"),

        pl.struct(legs1_duration_cols + legs1_cabin_cols)
        .map_elements(lambda row: max_duration_cabin(row, legs1_duration_cols, legs1_cabin_cols))
        .alias("legs1_max_duration_cabin"),
    ])

    df = df.with_columns(
        (
            (pl.col("legs0_max_duration_cabin") == pl.col("legs1_max_duration_cabin"))
            .cast(pl.Int8)
            .alias("is_max_duration_cabin_same")
        )
    )

    def weighted_mean(durations, cabins):
        pairs = [(d, c) for d, c in zip(durations, cabins) if d is not None and c not in (0, None)]
        if not pairs:
            return 0
        num = sum(d * c for d, c in pairs)
        denom = sum(d for d, _ in pairs)
        return num / denom if denom > 0 else 0

    df = df.with_columns([
        pl.struct(legs0_duration_cols + legs0_cabin_cols)
        .map_elements(lambda row: weighted_mean(
            [row[c] for c in legs0_duration_cols],
            [row[c] for c in legs0_cabin_cols]
        ))
        .alias("legs0_weighted_mean_cabin"),

        pl.struct(legs1_duration_cols + legs1_cabin_cols)
        .map_elements(lambda row: weighted_mean(
            [row[c] for c in legs1_duration_cols],
            [row[c] for c in legs1_cabin_cols]
        ))
        .alias("legs1_weighted_mean_cabin"),
    ])

    all_duration_cols = legs0_duration_cols + legs1_duration_cols
    all_cabin_cols = legs0_cabin_cols + legs1_cabin_cols

    df = df.with_columns(
        pl.struct(all_duration_cols + all_cabin_cols)
        .map_elements(lambda row: weighted_mean(
            [row[c] for c in all_duration_cols],
            [row[c] for c in all_cabin_cols]
        ))
        .alias("total_weighted_mean_cabin")
    )

    df = df.with_columns([
        legs0_mean,
        legs1_mean,
        is_same_cabin
    ])

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        df.write_parquet(output_path)
        print(f"✅ 已儲存 Parquet: {output_path}")

    return df

import polars as pl
import os

def build_time_features(
    df: pl.DataFrame,
    output_dir: str = None,
    output_filename: str = "6_time_features.parquet"
) -> pl.DataFrame:
    """
    建立與時間相關的特徵:
    - 時間分段
    - 是否週末
    - 是否 round trip
    - 出發到抵達的天數
    - 預訂到出發的天數
    """
    time_cols = ["legs0_departureAt", "legs0_arrivalAt", "legs1_departureAt", "legs1_arrivalAt"]
    time_exprs = []

    for col in time_cols:
        if col in df.columns:
            cleaned_col = (
                pl.when(pl.col(col) == "missing")
                  .then(None)
                  .otherwise(pl.col(col))
            )

            dt = cleaned_col.str.to_datetime(strict=False)
            h = dt.dt.hour()

            period = (
                pl.when(h.is_between(0,5)).then(0)
                .when(h.is_between(6,11)).then(1)
                .when(h.is_between(12,17)).then(2)
                .when(h.is_between(18,23)).then(3)
            )

            is_weekend = (
                (dt.dt.weekday() >= 5)
            ).cast(pl.Int32).fill_null(-1)

            time_exprs.extend([
                h.fill_null(-1).alias(f"{col}_hour"),
                dt.dt.weekday().fill_null(-1).alias(f"{col}_weekday"),
                (
                    ((h >= 6) & (h <= 9)) | ((h >= 17) & (h <= 20))
                ).cast(pl.Int32).fill_null(-1).alias(f"{col}_business_time"),
                period.fill_null(-1).alias(f"{col}_day_period"),
                is_weekend.alias(f"{col}_is_weekend")
            ])

    # 是否 round trip
    round_trip_flag = (
        (
            (pl.col("legs1_departureAt").is_not_null() & (pl.col("legs1_departureAt") != "missing"))
            |
            (pl.col("legs1_arrivalAt").is_not_null() & (pl.col("legs1_arrivalAt") != "missing"))
        )
        .cast(pl.Int8)
        .alias("is_round_trip")
    )

    # legs0_departureAt datetime
    depart_dt = (
        pl.when(pl.col("legs0_departureAt") == "missing")
          .then(None)
          .otherwise(pl.col("legs0_departureAt"))
    ).str.to_datetime(strict=False)

    # legs1_arrivalAt datetime
    arrive_dt = (
        pl.when(pl.col("legs1_arrivalAt") == "missing")
          .then(None)
          .otherwise(pl.col("legs1_arrivalAt"))
    ).str.to_datetime(strict=False)

    # 出發 ~ 抵達天數
    duration_ms_arrive = (arrive_dt - depart_dt).dt.total_milliseconds()
    days_between = (
        (duration_ms_arrive / (1000 * 60 * 60 * 24))
        .floor()
        .cast(pl.Int32)
        .fill_null(0)
        .alias("days_between_departure_arrival")
    )

    # requestDate ~ 出發天數
    request_dt = pl.col("requestDate")
    duration_ms_request = (depart_dt - request_dt).dt.total_milliseconds()
    days_before_departure = (
        (duration_ms_request / (1000 * 60 * 60 * 24))
        .floor()
        .cast(pl.Int32)
        .fill_null(-1)
        .alias("days_before_departure")
    )

    # 加入全部特徵
    df = df.with_columns(
        time_exprs +
        [round_trip_flag, days_between, days_before_departure]
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        df.write_parquet(output_path)
        print(f"✅ 已儲存 Parquet: {output_path}")

    print("✅ 所有時間特徵已生成完成")
    return df



import polars as pl
import os

def build_corporate_access_route_features(
    df: pl.DataFrame,
    output_dir: str = None,
    output_filename: str = "7_corporate_access_route_features.parquet"
) -> pl.DataFrame:
    """
    建立以下特徵：
    - 是否有 corporate tariff
    - 是否有 access TP
    - 是否為熱門航線
    """
    df = df.with_columns([
        pl.col("corporateTariffCode").is_not_null().cast(pl.Int32).alias("has_corporate_tariff"),
        (pl.col("pricingInfo_isAccessTP") == 1).cast(pl.Int32).alias("has_access_tp"),
        pl.col("searchRoute").is_in([
            "MOWLED/LEDMOW",
            "LEDMOW/MOWLED",
            "MOWLED",
            "LEDMOW",
            "MOWAER/AERMOW"
        ]).cast(pl.Int32).alias("is_popular_route")
    ])

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        df.write_parquet(output_path)
        print(f"✅ 已儲存 Parquet: {output_path}")

    print("✅ 已完成 corporate/access/route 特徵生成")
    return df


import polars as pl
import os

def build_transfer_features(
    df: pl.DataFrame,
    output_dir: str = None,
    output_filename: str = "8_transfer_features.parquet"
) -> pl.DataFrame:
    """
    建立轉機相關特徵，包含：
    - legs0/legs1/總轉機次數
    - duration_ratio
    - 是否直飛
    - groupby ranker_id 排名
    - 是否最少轉機
    """
    # 先把 duration 欄位轉分鐘
    for dur_col in ["legs0_duration", "legs1_duration"]:
        if dur_col in df.columns and df[dur_col].dtype == pl.Utf8:
            df = df.with_columns(
                pl.when(pl.col(dur_col).is_in([None, "missing"]))
                .then(0)
                .otherwise(
                    pl.col(dur_col).str.extract(r"^(\d+):", 1).cast(pl.Int64) * 60 +
                    pl.col(dur_col).str.extract(r":(\d+):", 1).cast(pl.Int64)
                )
                .alias(dur_col)
            )

    # Legs0 segments1~3
    legs0_segment_cols = [
        "legs0_segments1_departureFrom_airport_iata",
        "legs0_segments2_departureFrom_airport_iata",
        "legs0_segments3_departureFrom_airport_iata"
    ]

    legs1_segment_cols = [
        "legs1_segments1_departureFrom_airport_iata",
        "legs1_segments2_departureFrom_airport_iata",
        "legs1_segments3_departureFrom_airport_iata"
    ]

    # legs0轉機次數
    legs0_num_segments = (
        pl.sum_horizontal([
            ((pl.col(c).is_not_null()) & (pl.col(c) != "missing")).cast(pl.Int8)
            for c in legs0_segment_cols if c in df.columns
        ])
        .alias("legs0_num_transfers")
    )

    # legs1轉機次數
    legs1_num_segments = (
        pl.sum_horizontal([
            ((pl.col(c).is_not_null()) & (pl.col(c) != "missing")).cast(pl.Int8)
            for c in legs1_segment_cols if c in df.columns
        ])
        .alias("legs1_num_transfers")
    )

    df = df.with_columns([
        legs0_num_segments,
        legs1_num_segments
    ])

    # legs0 + legs1總轉機次數
    df = df.with_columns([
        (pl.col("legs0_num_transfers") + pl.col("legs1_num_transfers")).alias("total_num_transfers"),
        pl.when(pl.col("legs1_duration").fill_null(0) > 0)
            .then(pl.col("legs0_duration") / (pl.col("legs1_duration") + 1))
            .otherwise(1.0)
            .alias("duration_ratio")
    ])

    # 是否直飛
    df = df.with_columns([
        (pl.col("legs0_num_transfers") == 0).cast(pl.Int8).alias("legs0_is_direct"),
        (pl.col("legs1_num_transfers") == 0).cast(pl.Int8).alias("legs1_is_direct"),
        (
            (pl.col("legs0_num_transfers") == 0) & (pl.col("legs1_num_transfers") == 0)
        ).cast(pl.Int8).alias("both_legs_direct")
    ])

    # 排名
    df = df.with_columns([
        pl.col("legs0_num_transfers").rank(method="dense", descending=False).over("ranker_id").alias("legs0_num_transfers_rank"),
        pl.col("legs1_num_transfers").rank(method="dense", descending=False).over("ranker_id").alias("legs1_num_transfers_rank"),
        pl.col("total_num_transfers").rank(method="dense", descending=False).over("ranker_id").alias("total_num_transfers_rank")
    ])

    # 是否最少轉機
    df = df.with_columns([
        (pl.col("legs0_num_transfers") == pl.col("legs0_num_transfers").min().over("ranker_id"))
            .cast(pl.Int8).alias("legs0_is_min_transfers"),
        (pl.col("legs1_num_transfers") == pl.col("legs1_num_transfers").min().over("ranker_id"))
            .cast(pl.Int8).alias("legs1_is_min_transfers"),
        (pl.col("total_num_transfers") == pl.col("total_num_transfers").min().over("ranker_id"))
            .cast(pl.Int8).alias("total_is_min_transfers")
    ])

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        df.write_parquet(output_path)
        print(f"✅ 已儲存 Parquet: {output_path}")

    print("✅ 已完成轉機特徵生成")
    return df



from typing import Optional
import os
def build_carrier_consistency_features(
    df: pl.DataFrame,
    output_dir: str = None,
    output_filename: str = "9_carrier_consistency_features.parquet",
    transform_config: Optional[dict] = None
) -> pl.DataFrame:
    """
    建立 legs0/legs1 主 Carrier 一致性特徵 (自動先計算轉機次數)。
    可選: 使用transform_config進行共用Carrier/Departure編碼。
    """

    # legs0轉機判斷欄
    legs0_segment_cols = [
        "legs0_segments1_departureFrom_airport_iata",
        "legs0_segments2_departureFrom_airport_iata",
        "legs0_segments3_departureFrom_airport_iata"
    ]
    legs1_segment_cols = [
        "legs1_segments1_departureFrom_airport_iata",
        "legs1_segments2_departureFrom_airport_iata",
        "legs1_segments3_departureFrom_airport_iata"
    ]

    # legs0轉機次數
    legs0_num_segments = (
        pl.sum_horizontal([
            ((pl.col(c).is_not_null()) & (pl.col(c) != "missing")).cast(pl.Int8)
            for c in legs0_segment_cols if c in df.columns
        ])
        .alias("legs0_num_transfers")
    )
    # legs1轉機次數
    legs1_num_segments = (
        pl.sum_horizontal([
            ((pl.col(c).is_not_null()) & (pl.col(c) != "missing")).cast(pl.Int8)
            for c in legs1_segment_cols if c in df.columns
        ])
        .alias("legs1_num_transfers")
    )

    df = df.with_columns([
        legs0_num_segments,
        legs1_num_segments
    ])

    # legs0主carrier
    legs0_marketing_cols = [
        f"legs0_segments{s}_marketingCarrier_code"
        for s in range(4) if f"legs0_segments{s}_marketingCarrier_code" in df.columns
    ]
    legs1_marketing_cols = [
        f"legs1_segments{s}_marketingCarrier_code"
        for s in range(4) if f"legs1_segments{s}_marketingCarrier_code" in df.columns
    ]

    legs0_main_carrier = (
        pl.coalesce([pl.col(c) for c in legs0_marketing_cols])
        .alias("legs0_main_carrier")
    )
    legs1_main_carrier = (
        pl.coalesce([pl.col(c) for c in legs1_marketing_cols])
        .alias("legs1_main_carrier")
    )

    df = df.with_columns([
        legs0_main_carrier,
        legs1_main_carrier
    ])

    # 兩腿一致性標記
    legs0_all_same = (
        pl.when(pl.col("legs0_num_transfers") == 0)
        .then(1)
        .otherwise(
            pl.all_horizontal([
                (pl.col(c) == pl.col("legs0_main_carrier")) & pl.col(c).is_not_null()
                for c in legs0_marketing_cols
            ]).cast(pl.Int8)
        )
        .alias("legs0_all_segments_carrier_same")
    )
    legs1_all_same = (
        pl.when(pl.col("legs1_num_transfers") == 0)
        .then(1)
        .otherwise(
            pl.all_horizontal([
                (pl.col(c) == pl.col("legs1_main_carrier")) & pl.col(c).is_not_null()
                for c in legs1_marketing_cols
            ]).cast(pl.Int8)
        )
        .alias("legs1_all_segments_carrier_same")
    )

    df = df.with_columns([
        legs0_all_same,
        legs1_all_same
    ])

    both_legs_all_same = (
        (
            (pl.col("legs0_all_segments_carrier_same") == 1) &
            (pl.col("legs1_all_segments_carrier_same") == 1) &
            (pl.col("legs0_main_carrier") == pl.col("legs1_main_carrier")) &
            pl.col("legs0_main_carrier").is_not_null() &
            pl.col("legs1_main_carrier").is_not_null()
        ).cast(pl.Int8)
        .alias("both_legs_carrier_all_same")
    )

    df = df.with_columns([
        both_legs_all_same
    ])

    # ✅ 如果提供transform_config就做共用encoding
    if transform_config:
        carrier_enc = transform_config["label_encoders"]["carrier_cols"]
        mapping_df = pl.DataFrame({
            "value": carrier_enc["values"],
            "rank_id": carrier_enc["codes"]
        })
        # 需要共用encoding的欄位
        cols_to_encode = ["legs0_main_carrier", "legs1_main_carrier"]
        # departure欄位
        departure_cols = [
            "legs0_segments0_departureFrom_airport_iata",
            "legs1_segments0_departureFrom_airport_iata"
        ]
        cols_to_encode += [c for c in departure_cols if c in df.columns]
        print(f"✅ 正在共用carrier encoding處理 {cols_to_encode}")

        for col in cols_to_encode:
            df_col = (
                df.select([col])
                .with_columns(pl.col(col).cast(pl.Utf8))
                .join(
                    mapping_df.rename({"value": col}),
                    on=col,
                    how="left"
                )
                .with_columns(
                    pl.col("rank_id").fill_null(-1).cast(pl.Int32).alias(f"{col}_encoded")
                )
                .drop([col, "rank_id"])      # <<=== 這行就會確保 rank_id 永遠不留
                .rename({f"{col}_encoded": col})
            )
            df = df.with_columns(df_col)


    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        df.write_parquet(output_path)
        print(f"✅ 已儲存 Parquet: {output_path}")

    print("✅ 已完成主Carrier一致性與轉機次數特徵")
    return df




import polars as pl
import pickle
import os
from typing import Optional, Dict, Tuple

def build_label_encoding_features(
    df: pl.DataFrame,
    output_dir: Optional[str] = None,
    transform_config: Optional[Dict] = None,
    id_col: str = "Id"
) -> Tuple[pl.DataFrame, dict]:
    """
    快速批次Label Encoding，保留Id供後續join。
    對airport_cols共用同一個encoding，carrier_cols共用同一個encoding。
    """
    all_cols = df.columns
    label_enc_cols = []

    if id_col not in all_cols:
        raise ValueError(f"'{id_col}' 不存在於df.columns，無法作為主鍵")

    # Collect target columns
    aircraft_cols = [c for c in all_cols if c.endswith("_aircraft_code")]
    label_enc_cols += aircraft_cols

    flightnum_cols = [c for c in all_cols if c.endswith("_flightNumber")]
    label_enc_cols += flightnum_cols

    airport_cols = [c for c in all_cols if "_arrivalTo_airport_" in c or "_departureFrom_airport_" in c]
    label_enc_cols += airport_cols

    carrier_cols = [c for c in all_cols if c.endswith("_marketingCarrier_code") or c.endswith("_operatingCarrier_code")]
    label_enc_cols += carrier_cols

    if "searchRoute" in all_cols:
        label_enc_cols.append("searchRoute")

    # 還原模式
    if transform_config:
        label_encoders = transform_config["label_encoders"]
        df_result = df.select([id_col])
        for key, enc in label_encoders.items():
            cols = enc["columns"]
            mapping_df = pl.DataFrame({
                "value": enc["values"],
                "rank_id": enc["codes"]
            })
            for col in cols:
                df_col = (
                    df.select([id_col, col])
                    .with_columns(pl.col(col).cast(pl.Utf8))
                    .join(mapping_df.rename({"value": col}), on=col, how="left")
                    .with_columns(
                        pl.col("rank_id").fill_null(-1).cast(pl.Int32).alias(col)
                    )
                    .drop("rank_id")
                )
                df_result = df_result.join(df_col, on=id_col, how="left")
        print("✅ 已完成還原模式 Label Encoding (使用transform_config)")
                # ✅ 如果指定輸出目錄，儲存還原後的df
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            restored_path = os.path.join(output_dir, "10_df_restored_features.parquet")
            df_result.write_parquet(restored_path)
            print(f"✅ 已儲存還原後特徵: {restored_path}")
            
        return df_result, transform_config

    # 新訓練模式
    label_encoders = {}
    df_encoded = df.select([id_col])

    def encode_shared(cols: list, encoder_name: str):
        """
        對多個欄位共用同一個encoding
        """
        mapping_df = (
            df.select(cols)
            .melt()
            .select(pl.col("value").unique())
            .with_columns(
                (pl.col("value").rank("dense") - 1).fill_null(-1).cast(pl.Int32).alias("rank_id")
            )
            .sort("rank_id")
        )
        label_encoders[encoder_name] = {
            "columns": cols,
            "values": mapping_df["value"].to_list(),
            "codes": mapping_df["rank_id"].to_list()
        }
        for col in cols:
            encoded = (
                df.select([id_col, col])
                .with_columns(pl.col(col).cast(pl.Utf8))
                .join(mapping_df.rename({"value": col}), on=col, how="left")
                .with_columns(
                    pl.col("rank_id").fill_null(-1).cast(pl.Int32).alias(col)
                )
                .drop("rank_id")
            )
            nonlocal df_encoded
            df_encoded = df_encoded.join(encoded, on=id_col, how="left")

    def encode_individual(col: str):
        """
        對單個欄位encoding
        """
        mapping_df = (
            df.select(pl.col(col))
            .unique()
            .with_columns(
                (pl.col(col).rank("dense") - 1).fill_null(-1).cast(pl.Int32).alias("rank_id")
            )
            .sort("rank_id")
        )
        label_encoders[col] = {
            "columns": [col],
            "values": mapping_df[col].to_list(),
            "codes": mapping_df["rank_id"].to_list()
        }
        encoded = (
            df.select([id_col, col])
            .with_columns(pl.col(col).cast(pl.Utf8))
            .join(mapping_df, on=col, how="left")
            .with_columns(
                pl.col("rank_id").fill_null(-1).cast(pl.Int32).alias(col)
            )
            .drop("rank_id")
        )
        nonlocal df_encoded
        df_encoded = df_encoded.join(encoded, on=id_col, how="left")

    # 先對共用欄位做encoding
    if airport_cols:
        encode_shared(airport_cols, "airport_cols")
    if carrier_cols:
        encode_shared(carrier_cols, "carrier_cols")
    if aircraft_cols:
        encode_shared(aircraft_cols, "aircraft_cols")
    if flightnum_cols:
        encode_shared(flightnum_cols, "flightnum_cols")

    # 再對其他欄位做encoding
    other_cols = []
    if "searchRoute" in all_cols:
        other_cols.append("searchRoute")


    for c in other_cols:
        encode_individual(c)
    # 輸出transform_config
    config = {
        "label_encoders": label_encoders
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # 儲存config
        config_path = os.path.join(output_dir, "transform_config_rank.pkl")
        with open(config_path, "wb") as f:
            pickle.dump(config, f)
        print(f"✅ 已儲存 transform_config: {config_path}")
        # 儲存encoded df
        encoded_path = os.path.join(output_dir, "10_df_encoded_features.parquet")
        df_encoded.write_parquet(encoded_path)
        print(f"✅ 已儲存編碼後特徵: {encoded_path}")

    print("✅ 新訓練Label Encoding完成 (保留Id，共用airport/carrier編碼)")
    return df_encoded, config





def clean_fill_and_cast_columns(
    df: pl.DataFrame,
    test: bool = False
) -> pl.DataFrame:
    """
    清理資料：
    - 將所有空字串視為null，再填入'missing'
    - 數值欄填0
    - Boolean欄轉0/1
    - 如果 test=True，duration_cols 全部轉成字串並填'missing'
    """

    # 找字串欄
    str_cols = [c for c in df.columns if df[c].dtype in (pl.Utf8, pl.String)]
    # 找數值欄
    numeric_cols = [c for c in df.columns if df[c].dtype.is_numeric()]
    # 找布林欄
    bool_cols = [c for c in df.columns if df[c].dtype == pl.Boolean]

    print(f"✅ 共找到 {len(str_cols)} 個字串欄位")
    print(f"✅ 共找到 {len(numeric_cols)} 個數值欄位")
    print(f"✅ 共找到 {len(bool_cols)} 個布林欄位")

    # 把空字串變null
    string_exprs = [
        pl.when(pl.col(c).str.strip_chars() == "")
          .then(None)
          .otherwise(pl.col(c))
          .alias(c)
        for c in str_cols
    ]
    df = df.with_columns(string_exprs)

    # 填補缺失
    df = df.with_columns(
        [pl.col(c).fill_null("missing") for c in str_cols] +
        [pl.col(c).fill_null(0) for c in numeric_cols]
    )

    # 布林轉0/1
    df = df.with_columns([
        pl.col(c).cast(pl.Int8).alias(c) for c in bool_cols
    ])

    # ✅ 如果 test=True，處理 duration_cols
    if test:
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
        duration_cols_exist = [c for c in duration_cols if c in df.columns]
        if duration_cols_exist:
            duration_exprs = [
                (
                    pl.col(c)
                    .cast(pl.Utf8)
                    .fill_null("missing")
                    .alias(c)
                )
                for c in duration_cols_exist
            ]
            df = df.with_columns(duration_exprs)
            print(f"✅ test=True: 已將 {len(duration_cols_exist)} 個duration欄位轉str並填'missing'")

    print("✅ 已完成空字串處理、缺失補值、布林轉0/1")
    return df



import polars as pl
import os
import glob
import re

def merge_original_with_extra_features(
    base_parquet_path: str,
    extra_features_dir: str,
    id_col: str = "Id"
) -> pl.DataFrame:
    """
    把原始Parquet和指定資料夾中所有Parquet檔案依據Id合併。
    如果feature名稱重複，以新檔案的值覆蓋。

    參數:
    - base_parquet_path: 原始資料路徑，例如 "data/train.parquet"
    - extra_features_dir: 額外特徵資料夾路徑，例如 "data/extra_features/train/"
    - id_col: 主鍵欄位，預設 "Id"

    回傳:
    - 合併後的DataFrame
    """

    print(f"✅ 讀取原始資料: {base_parquet_path}")
    df = pl.read_parquet(base_parquet_path)

    if id_col not in df.columns:
        raise ValueError(f"'{id_col}' 欄位不存在於原始資料！")
    
    original_cols = set(df.columns)
    bool_cols = [c for c in df.columns if df[c].dtype == pl.Boolean]

    # 布林轉0/1
    df = df.with_columns([
        pl.col(c).cast(pl.Int8).alias(c) for c in bool_cols
    ])
    # 搜尋所有 parquet
    pattern = os.path.join(extra_features_dir, "**/*.parquet")
    extra_files = glob.glob(pattern, recursive=True)

    def extract_number(file_path):
        base = os.path.basename(file_path)
        m = re.match(r"(\d+)_", base)
        return int(m.group(1)) if m else float("inf")

    extra_files = sorted(extra_files, key=extract_number)


    if not extra_files:
        print(f"⚠️ 找不到任何Parquet檔於 {extra_features_dir}")
        return df

    print(f"✅ 共找到 {len(extra_files)} 個 Parquet 要合併")

    # 逐一合併
    for i, file_path in enumerate(extra_files):
        print(f"🔹 合併第 {i+1}/{len(extra_files)} 個: {file_path}")

        df_extra = pl.read_parquet(file_path)

        if id_col not in df_extra.columns:
            raise ValueError(f"'{id_col}' 欄位不存在於 {file_path}")

        # 若df_extra有和df重複的欄，先移除df裡的
        overlap_cols = set(df.columns) & set(df_extra.columns) - {id_col}
        if overlap_cols:
            print(f"⚠️ {len(overlap_cols)} 個特徵將被新檔案覆蓋: {list(overlap_cols)}")
            df = df.drop(overlap_cols)

        df = df.join(df_extra, on=id_col, how="left")

    merged_cols = set(df.columns)
    added_cols = merged_cols - original_cols

    print("✅ 已完成所有檔案合併")
    print(f"✅ 共新增 {len(added_cols)} 個新特徵")
    if added_cols:
        print(f"🔹 新增欄位: {sorted(added_cols)}")

    return df

import pickle
import os
import polars as pl
from typing import Optional
import json

def enrich_flight_view_features(
    df: pl.DataFrame,
    output_dir: Optional[str] = None,
    output_filename: str = "11_flight_view_features.parquet",
    transform_config: Optional[dict] = None
) -> tuple[pl.DataFrame, dict]:
    def make_leg_segment_keys(leg_prefix):
        keys = []
        for i in range(4):
            key_name = f"{leg_prefix}_segments{i}_key"
            dep = pl.col(f"{leg_prefix}_segments{i}_departureFrom_airport_iata").fill_null("missing")
            arr = pl.col(f"{leg_prefix}_segments{i}_arrivalTo_airport_iata").fill_null("missing")
            keys.append((dep + "-" + arr).alias(key_name))
        return keys

    df = df.with_columns(make_leg_segment_keys("legs0") + make_leg_segment_keys("legs1"))

    all_segments = [f"legs0_segments{i}_key" for i in range(4)] + [f"legs1_segments{i}_key" for i in range(4)]

    if transform_config is None:
        segment_counts = (
            df.melt(id_vars=[], value_vars=all_segments)
            .filter(pl.col("value") != "missing-missing")
            .group_by("value")
            .agg(pl.count().alias("segment_view_count"))
        )
        segment_counts_dict = segment_counts.to_dict(as_series=False)
    else:
        segment_counts = pl.DataFrame(transform_config["segment_counts"])

    for seg_col in all_segments:
        df = df.join(
            segment_counts,
            left_on=seg_col,
            right_on="value",
            how="left"
        ).with_columns(
            pl.col("segment_view_count").fill_null(0).alias(f"{seg_col}_view_count")
        ).drop("segment_view_count")

    def make_leg_full_key(leg_prefix):
        seg_keys = [f"{leg_prefix}_segments{i}_key" for i in range(4)]
        return pl.concat_str([pl.col(k) for k in seg_keys], separator="|").alias(f"{leg_prefix}_key")

    df = df.with_columns([
        make_leg_full_key("legs0"),
        make_leg_full_key("legs1"),
        (
            pl.concat_str([
                pl.concat_str([pl.col(f"legs0_segments{i}_key") for i in range(4)], separator="|"),
                pl.lit("||"),
                pl.concat_str([pl.col(f"legs1_segments{i}_key") for i in range(4)], separator="|"),
            ], separator="")
        ).alias("all_key")
    ])

    if transform_config is None:
        leg0_counts = df.group_by("legs0_key").agg(pl.count().alias("leg0_flight_view_count"))
        leg1_counts = df.group_by("legs1_key").agg(pl.count().alias("leg1_flight_view_count"))
        all_counts = df.group_by("all_key").agg(pl.count().alias("all_flight_view_count"))
        leg0_counts_dict = leg0_counts.to_dict(as_series=False)
        leg1_counts_dict = leg1_counts.to_dict(as_series=False)
        all_counts_dict = all_counts.to_dict(as_series=False)
    else:
        leg0_counts = pl.DataFrame(transform_config["leg0_counts"])
        leg1_counts = pl.DataFrame(transform_config["leg1_counts"])
        all_counts = pl.DataFrame(transform_config["all_counts"])

    df = df.join(leg0_counts, on="legs0_key", how="left")
    df = df.join(leg1_counts, on="legs1_key", how="left")
    df = df.join(all_counts, on="all_key", how="left")

    ranker_stats = df.group_by("ranker_id").agg([
        pl.max("leg0_flight_view_count").alias("leg0_view_max"),
        pl.max("leg1_flight_view_count").alias("leg1_view_max"),
        pl.max("all_flight_view_count").alias("all_view_max"),
    ])

    df = df.join(ranker_stats, on="ranker_id", how="left")

    df = df.with_columns([
        (pl.col("leg0_flight_view_count") / (pl.col("leg0_view_max") + 1e-5)).alias("leg0_view_norm"),
        (pl.col("leg1_flight_view_count") / (pl.col("leg1_view_max") + 1e-5)).alias("leg1_view_norm"),
        (pl.col("all_flight_view_count") / (pl.col("all_view_max") + 1e-5)).alias("all_view_norm"),
    ])

    ranker_stats_mean = df.group_by("ranker_id").agg([
        pl.mean("leg0_flight_view_count").alias("leg0_view_mean"),
        pl.mean("leg1_flight_view_count").alias("leg1_view_mean"),
        pl.mean("all_flight_view_count").alias("all_view_mean"),
    ])

    df = df.join(ranker_stats_mean, on="ranker_id", how="left")

    df = df.with_columns([
        (pl.col("leg0_flight_view_count") - pl.col("leg0_view_mean")).alias("leg0_view_diff_mean"),
        (pl.col("leg1_flight_view_count") - pl.col("leg1_view_mean")).alias("leg1_view_diff_mean"),
        (pl.col("all_flight_view_count") - pl.col("all_view_mean")).alias("all_view_diff_mean"),
    ])
    
    rank_features = [
        "leg0_flight_view_count",
        "leg1_flight_view_count",
        "all_flight_view_count",
    ] + [f"legs0_segments{i}_key_view_count" for i in range(4)] + [f"legs1_segments{i}_key_view_count" for i in range(4)]

    rank_exprs = []
    for col in rank_features:
        rank_exprs.append(
            pl.col(col).rank(method="dense").over("ranker_id").alias(f"{col}_rank")
        )

    df = df.with_columns(rank_exprs)

    output_config = None
    if transform_config is None:
        output_config = {
            "segment_counts": segment_counts_dict,
            "leg0_counts": leg0_counts_dict,
            "leg1_counts": leg1_counts_dict,
            "all_counts": all_counts_dict
        }

    # 最後要 drop 的 columns
    columns_to_drop = (
        [
            "leg0_view_max", "leg1_view_max", "all_view_max",
            "leg0_view_mean", "leg1_view_mean", "all_view_mean",
            "legs0_key", "legs1_key", "all_key"
        ]
        + [f"legs0_segments{i}_key" for i in range(4)]
        + [f"legs1_segments{i}_key" for i in range(4)]
        + [
            f"{leg}_segments{i}_{x}_airport_iata"
            for leg in ["legs0", "legs1"]
            for i in range(4)
            for x in ["departureFrom", "arrivalTo"]
        ]
    )
    df = df.drop(columns_to_drop)

    # 保留 Id 與所有新欄位
    keep_cols = ["Id"] + [col for col in df.columns if col != "Id"]
    df = df.select(keep_cols)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df.write_parquet(os.path.join(output_dir, output_filename))
        print(f"✅ 已儲存flight view特徵: {os.path.join(output_dir, output_filename)}")
        if transform_config is None and output_config is not None:
            config_path = os.path.join(output_dir, "transform_flight_view_key_config.pkl")
            with open(config_path, "wb") as f:
                pickle.dump(output_config, f)
            print(f"✅ 已儲存 transform_config: {config_path}")

    return df, output_config


import polars as pl
import os
import pickle
from typing import Optional, Tuple, Dict

def build_company_loo_features(
    df: pl.DataFrame,
    output_dir: Optional[str] = None,
    transform_dict: Optional[Dict] = None
) -> Tuple[pl.DataFrame, Optional[Dict]]:
    """
    建立公司 LOO aggregation 特徵：
    - 所有 mean 特徵：selected==1 且排除同 ranker_id
    - mode 特徵：selected==1，不做 LOO
    - 出現次數：所有紀錄，不做 LOO
    - 當 companyID 未出現，使用全體均值 fallback
    """
    save_transform = transform_dict
    target_col = "selected"
    company_col = "companyID"
    ranker_col = "ranker_id"

    df = df.with_columns([
        pl.col(target_col).cast(pl.Int8)
    ])

    # Duration轉分鐘
    duration_cols = ["legs0_duration", "legs1_duration"]
    duration_exprs = [
        pl.when(pl.col(c).is_in([None, "missing"]))
        .then(None)
        .otherwise(
            pl.col(c).str.extract(r"^(\d+):", 1).cast(pl.Int64) * 60 +
            pl.col(c).str.extract(r":(\d+):", 1).cast(pl.Int64)
        )
        .alias(c)
        for c in duration_cols
    ]
    df = df.with_columns(duration_exprs)

    # 時間特徵
    time_cols = ["legs0_departureAt", "legs0_arrivalAt", "legs1_departureAt", "legs1_arrivalAt"]
    time_exprs = []
    for col in time_cols:
        cleaned_col = (
            pl.when(pl.col(col).is_in(["missing", None, ""]))
            .then(None)
            .otherwise(pl.col(col))
        )
        dt = cleaned_col.str.to_datetime(strict=False)
        h = dt.dt.hour()
        time_exprs.append(
            h.fill_null(-1).alias(f"{col}_hour")
        )
    df = df.with_columns(time_exprs)

    # Cabin class
    if "legs0_segments0_cabinClass" in df.columns:
        df = df.with_columns(
            pl.col("legs0_segments0_cabinClass").cast(pl.Float32).alias("cabin_class")
        )
    else:
        df = df.with_columns(
            pl.lit(None).alias("cabin_class")
        )

    # Transfer
    df = df.with_columns([
        pl.sum_horizontal([
            ((pl.col(f"legs0_segments{i}_departureFrom_airport_iata").is_not_null()) &
             (pl.col(f"legs0_segments{i}_departureFrom_airport_iata") != "missing")).cast(pl.Int8)
            for i in range(1, 4)
        ]).alias("legs0_num_transfers"),
        pl.sum_horizontal([
            ((pl.col(f"legs1_segments{i}_departureFrom_airport_iata").is_not_null()) &
             (pl.col(f"legs1_segments{i}_departureFrom_airport_iata") != "missing")).cast(pl.Int8)
            for i in range(1, 4)
        ]).alias("legs1_num_transfers")
    ])
    df = df.with_columns([
        (pl.col("legs0_num_transfers") + pl.col("legs1_num_transfers")).fill_null(0).cast(pl.Int64).alias("total_num_transfers"),
        ((pl.col("legs0_num_transfers") + pl.col("legs1_num_transfers")) > 0).cast(pl.Int8).alias("has_transfer")
    ])

    agg_cols = [
        "totalPrice", "taxes",
        "legs0_duration", "legs1_duration",
        "cabin_class",
        "total_num_transfers"
    ] + [f"{c}_hour" for c in time_cols]
    
    stats_cols = [company_col] + [f"{c}_mean" for c in agg_cols] + ["selected_count"]

    if transform_dict is None:

        # selected==1 mean
        all_stats = (
            df.filter(pl.col(target_col) == 1)
            .group_by(company_col)
            .agg([
                *(pl.mean(c).alias(f"{c}_mean") for c in agg_cols),
                pl.count().alias("selected_count")
            ])
        )

        # 全體均值 fallback
        global_mean_row = (
            df.filter(pl.col(target_col) == 1)
            .select([
                pl.lit(-1).alias(company_col),
                *(pl.mean(c).alias(f"{c}_mean") for c in agg_cols),
                pl.count().alias("selected_count")
            ])
        )
        # 確保欄位名稱和順序一致
        global_mean_row = global_mean_row.select(all_stats.columns)

        # 強制同順序


        # mode
        def mode_table(col, alias, dtype):
            m = (
                df.filter(pl.col(target_col)==1)
                .group_by(company_col)
                .agg([
                    pl.col(col)
                    .value_counts(sort=True)
                    .struct.field(col)
                    .first()
                    .cast(dtype)
                    .alias(alias)
                ])
            )
            global_mode = (
                df.filter(pl.col(target_col)==1)
                .select([
                    pl.col(col)
                    .value_counts(sort=True)
                    .struct.field(col)
                    .first()
                    .cast(dtype)
                    .alias(alias)
                ])
                .with_columns(pl.lit(-1).alias(company_col))
            )
            return m, global_mode

        cabin_mode, global_cabin = mode_table("cabin_class","mode_cabin_class",pl.Int32)
        transfer_mode, global_transfer = mode_table("has_transfer","mode_has_transfer",pl.Int8)
        transfer_num_mode, global_transfer_num = mode_table("total_num_transfers","mode_transfer_num",pl.Int64)

        ranker_stats = (
            df.filter(pl.col(target_col) == 1)
            .group_by([ranker_col, company_col])
            .agg([
                *(pl.sum(c).alias(f"{c}_sum") for c in agg_cols),
                pl.count().alias("count")
            ])
        )
        total_counts = (
            df.group_by(company_col)
            .agg(pl.count().alias("total_occurrences"))
        )

        transform_dict = {
            "all_stats": all_stats.to_dict(as_series=False),
            "global_mean": global_mean_row.to_dict(as_series=False),
            "cabin_mode": cabin_mode.to_dict(as_series=False),
            "global_cabin": global_cabin.to_dict(as_series=False),
            "transfer_mode": transfer_mode.to_dict(as_series=False),
            "global_transfer": global_transfer.to_dict(as_series=False),
            "transfer_num_mode": transfer_num_mode.to_dict(as_series=False),
            "global_transfer_num": global_transfer_num.to_dict(as_series=False),
            "ranker_stats": ranker_stats.to_dict(as_series=False),
            "total_counts": total_counts.to_dict(as_series=False)
        }
    else:
        all_stats = pl.DataFrame(transform_dict["all_stats"])
        global_mean_row = pl.DataFrame(transform_dict["global_mean"])
        global_mean_row = global_mean_row.select(stats_cols)

        cabin_mode = pl.DataFrame(transform_dict["cabin_mode"])
        global_cabin = pl.DataFrame(transform_dict["global_cabin"])
        cabin_mode = cabin_mode.select([company_col, "mode_cabin_class"])
        global_cabin = global_cabin.select([company_col, "mode_cabin_class"])

        transfer_mode = pl.DataFrame(transform_dict["transfer_mode"])
        global_transfer = pl.DataFrame(transform_dict["global_transfer"])
        transfer_mode = transfer_mode.select([company_col, "mode_has_transfer"])
        global_transfer = global_transfer.select([company_col, "mode_has_transfer"])

        transfer_num_mode = pl.DataFrame(transform_dict["transfer_num_mode"])
        global_transfer_num = pl.DataFrame(transform_dict["global_transfer_num"])
        transfer_num_mode = transfer_num_mode.select([company_col, "mode_transfer_num"])
        global_transfer_num = global_transfer_num.select([company_col, "mode_transfer_num"])

        ranker_stats = pl.DataFrame(transform_dict["ranker_stats"])
        total_counts = pl.DataFrame(transform_dict["total_counts"])

        # concat global fallback row
        all_stats = pl.concat([all_stats, global_mean_row])
        cabin_mode = pl.concat([cabin_mode, global_cabin])
        transfer_mode = pl.concat([transfer_mode, global_transfer])
        transfer_num_mode = pl.concat([transfer_num_mode, global_transfer_num])


    # join
    df = df.join(all_stats, on=company_col, how="left")
    df = df.join(cabin_mode, on=company_col, how="left")
    df = df.join(transfer_mode, on=company_col, how="left")
    df = df.join(transfer_num_mode, on=company_col, how="left")
    df = df.join(ranker_stats, on=[ranker_col, company_col], how="left")
    df = df.join(total_counts, on=company_col, how="left")

    # LOO mean
    new_cols = []
    for c in agg_cols:
        new_cols.append(
                pl.col(f"{c}_mean").alias(f"{company_col}_loo_mean_{c}")
        )
    new_cols.append(
            pl.col("selected_count").alias(f"{company_col}_loo_selected_count")
        )

    # mode 和 occurrence不變
    new_cols.append(pl.col("mode_cabin_class").alias(f"{company_col}_mode_cabin_class"))
    new_cols.append(pl.col("mode_has_transfer").alias(f"{company_col}_mode_has_transfer"))
    new_cols.append(pl.col("mode_transfer_num").alias(f"{company_col}_mode_transfer_num"))
    new_cols.append(pl.col("total_occurrences").alias(f"{company_col}_total_occurrences"))

    df = df.with_columns(new_cols)

    kept_cols = ["Id"] + [c.meta.output_name() for c in new_cols]
    df = df.select(kept_cols)

    # 儲存
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df_path = os.path.join(output_dir, "12_companyID_features.parquet")
        df.write_parquet(df_path)
        print(f"✅ 已儲存 transform_dict: {df_path}")
        if save_transform is None:
            config_path = os.path.join(output_dir, "transform_dict_companyID.pkl")
            with open(config_path, "wb") as f:
                pickle.dump(transform_dict, f)
            print(f"✅ 已儲存 transform_dict: {config_path}")

    return df, transform_dict

import os
import pickle
import polars as pl
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

def build_cluster_transform_dict(
    transform_path: str,
    output_path: str,
    k: int = 3
):
    """
    從 transform_dict_companyID.pkl 讀取，進行 KMeans clustering，生成 cluster summary，並存成 transform_dict_cluster.pkl
    """
    # === 讀取 transform_dict ===
    with open(transform_path, "rb") as f:
        transform_dict = pickle.load(f)

    all_stats = pl.DataFrame(transform_dict["all_stats"])
    cabin_mode = pl.DataFrame(transform_dict["cabin_mode"])
    transfer_mode = pl.DataFrame(transform_dict["transfer_mode"])
    transfer_num_mode = pl.DataFrame(transform_dict["transfer_num_mode"])
    total_counts = pl.DataFrame(transform_dict["total_counts"])
    global_mean = pl.DataFrame(transform_dict["global_mean"])

    # 加上 fallback row
    all_stats = pl.concat([all_stats, global_mean])

    # 合併成 summary
    company_summary = (
        all_stats
        .join(cabin_mode, on="companyID", how="left")
        .join(transfer_mode, on="companyID", how="left")
        .join(transfer_num_mode, on="companyID", how="left")
        .join(total_counts, on="companyID", how="left")
    )

    # Null -> 0
    company_summary_filled = company_summary.fill_null(0)

    # Numeric columns
    exclude_cols = {"companyID"}
    numeric_cols = [
        c for c, dtype in company_summary_filled.schema.items()
        if c not in exclude_cols and dtype in pl.NUMERIC_DTYPES
    ]

    # Scaling
    X_np = company_summary_filled.select(numeric_cols).to_numpy()
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_np)

    # Clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)

    company_summary_clustered = company_summary_filled.with_columns(
        pl.Series("cluster_label", labels)
    )

    # 聚合每個 cluster
    base_features = [c for c in company_summary_clustered.columns if c not in {"companyID", "cluster_label"}]
    agg_exprs = [
        pl.col(feat).mean().alias(f"{feat}_mean") for feat in base_features
    ]

    cluster_summary = (
        company_summary_clustered
        .group_by("cluster_label")
        .agg(agg_exprs)
        .sort("cluster_label")
    )

    # Merge cluster summary back
    company_with_cluster_features = (
        company_summary_clustered
        .join(cluster_summary, on="cluster_label", how="left")
    )

    final_df = company_with_cluster_features.select(
        ["companyID", "cluster_label"] + cluster_summary.drop("cluster_label").columns
    )

    # 儲存 transform_dict
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cluster_transform_dict = {
        "cluster_summary": final_df.to_dict(as_series=False)
    }

    with open(output_path, "wb") as f:
        pickle.dump(cluster_transform_dict, f)

    print(f"✅ 已儲存 transform_dict: {output_path}")
    
    return cluster_transform_dict


import os
import pickle
import polars as pl

import os
import pickle
import polars as pl
import os
import pickle
import polars as pl

def add_cluster_features_and_save(
    df: pl.DataFrame,
    transform_dict_path: str,
    output_dir: str
):
    """
    根據 transform_dict (cluster) 對應 companyID 加入 cluster features，若找不到則用 fallback (-1)。
    
    Args:
        df: 要加上 features 的 DataFrame (必須有 "companyID")
        transform_dict_path: transform_dict_cluster.pkl 路徑
        output_dir: 輸出目錄
    """
    if "companyID" not in df.columns:
        raise ValueError("❌ DataFrame 缺少 'companyID' 欄位")
    
    # 讀取 transform_dict
    with open(transform_dict_path, "rb") as f:
        transform_dict = pickle.load(f)
    
    cluster_features_df = pl.DataFrame(transform_dict["cluster_summary"])
    
    # fallback row
    fallback_row = cluster_features_df.filter(pl.col("companyID") == -1)
    
    # 對 df 先 left join cluster_features
    df_joined = df.join(
        cluster_features_df,
        on="companyID",
        how="left"
    )

    # 再依序對每個欄 coalesce() fallback
    feature_cols = [c for c in cluster_features_df.columns if c != "companyID"]
    for col in feature_cols:
        fallback_value = fallback_row[col].to_numpy()[0] if fallback_row.height else None
        df_joined = df_joined.with_columns(
            pl.col(col).fill_null(fallback_value)
        )

    # 輸出
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "13_cluster_features.parquet")
    df_joined.write_parquet(output_path)
    
    print(f"✅ 已儲存 cluster features: {output_path}")
    print(df_joined.head())
    
    return df_joined


import polars as pl
def drop_constant_numeric_columns(
    df: pl.DataFrame,
    threshold: float = 1.0
) -> pl.DataFrame:
    """
    檢查所有 numeric 欄位，若最常見值佔比 >= threshold，則移除該欄。
    """
    if not (0 < threshold <= 1.0):
        raise ValueError("threshold 必須在 (0, 1]")

    numeric_cols = [c for c, dtype in df.schema.items() if dtype in pl.NUMERIC_DTYPES]
    if not numeric_cols:
        print("⚠️ DataFrame 中沒有 numeric 欄位，無需檢查")
        return df

    columns_to_drop = []

    for col in numeric_cols:
        vc_df = (
            df.select(pl.col(col).value_counts())
            .unnest(col)
            .sort(by=["count", col], descending=[True, False])
        )
        most_common_count = vc_df[0, "count"]
        ratio = most_common_count / df.height
        if ratio >= threshold:
            print(f"🚮 欄位 {col} 最常見值佔比 {ratio:.4f} >= {threshold}, 將移除")
            columns_to_drop.append(col)

    if columns_to_drop:
        df = df.drop(columns_to_drop)
        print(f"✅ 已移除 {len(columns_to_drop)} 個幾乎無變化的 numeric 欄位: {columns_to_drop}")
    else:
        print("✅ 所有 numeric 欄位變異性足夠，無需刪除")
    print(f"目前有{len(df.columns)}")
    return df
