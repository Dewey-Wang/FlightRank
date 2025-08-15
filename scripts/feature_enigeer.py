
import polars as pl
from typing import List

def convert_prices_to_rub_by_currency(
    df: pl.DataFrame,
    cols_to_convert: List[str],
    *,
    kzt_to_rub: float,
    in_place: bool = True,
    strict_unique: bool = True,   # 每個 ranker_id 應只有一種幣別，不唯一就報錯
) -> pl.DataFrame:
    """
    依 ranker_id 取得唯一幣別（忽略 null/None/NaN/'missing'/''/'null'/'none'），
    設定匯率：RUB=1.0、KZT=kzt_to_rub，並將 cols_to_convert 轉為 RUB。
    - in_place=True 會直接覆蓋原欄位；未知/缺失匯率時保留原值（當作 *1）
    - in_place=False 會輸出為新欄位（<col>），若 to_RUB 缺失則結果為 null
    """
    if "ranker_id" not in df.columns or "currencyCode" not in df.columns:
        raise ValueError("需要存在 'ranker_id' 與 'currencyCode' 欄位")

    # 1) 正規化 currencyCode：轉小寫、去空白，忽略缺值/特殊字串
    #    只拿乾淨值進行唯一性與轉換
    cleaned = (
        df.select(
            "ranker_id",
            pl.col("currencyCode").cast(pl.Utf8).alias("currencyCode")
        )
        .with_columns(
            pl.col("currencyCode").str.strip_chars().str.to_lowercase().alias("__cur")
        )
        .with_columns(
            pl.when(
                (pl.col("__cur") == "") |
                pl.col("__cur").is_in(["missing", "none", "null", "nan"])
            )
            .then(None)
            .otherwise(pl.col("__cur"))
            .alias("currencyCode_clean")
        )
    )

    # 2) 依 ranker_id 決定唯一幣別（忽略清掉後的 null）
    cur = (
        cleaned
        .group_by("ranker_id")
        .agg([
            pl.col("currencyCode_clean").drop_nulls().n_unique().alias("n_currency"),
            pl.col("currencyCode_clean").drop_nulls().first().alias("currencyCode_clean"),
        ])
    )

    if strict_unique:
        bad = cur.filter(pl.col("n_currency") > 1)
        if bad.height > 0:
            raise ValueError(
                "發現某些 ranker_id 有多種 currencyCode，請先檢查：\n"
                + bad.head(20).to_pandas().__repr__()
            )

    # 3) 匯率表：rub=1.0, kzt=kzt_to_rub；其它/未知 -> None
    cur = (
        cur.with_columns(
            pl.when(pl.col("currencyCode_clean") == "rub").then(pl.lit(1.0))
             .when(pl.col("currencyCode_clean") == "kzt").then(pl.lit(float(kzt_to_rub)))
             .otherwise(None)
             .alias("to_RUB")
        )
        .select(["ranker_id", "to_RUB"])
    )

    # 4) 併回原表並轉換
    out = df.join(cur, on="ranker_id", how="left")

    # in_place 模式：未知匯率時保留原值 → 用 coalesce(to_RUB, 1.0)
    rate_col = pl.coalesce([pl.col("to_RUB"), pl.lit(1.0)]) if in_place else pl.col("to_RUB")

    converted = []
    for c in cols_to_convert:
        if c not in out.columns:
            continue
        # 轉成浮點後乘以匯率
        target_name = c if in_place else f"{c}_rub"
        converted.append(
            (pl.col(c).cast(pl.Float64, strict=False) * rate_col).alias(target_name)
        )

    out = out.with_columns(converted).drop("to_RUB")
    return out


import polars as pl
from typing import Optional
import polars as pl

def apply_mini_rules(
    df: pl.DataFrame,
    *,
    percent_scale: float = 100.0,
) -> pl.DataFrame:
    """
    將 miniRulesX_percentage * totalPrice/percent_scale 加到 miniRulesX_monetaryAmount，
    若任一參與欄位為 null 或 NaN -> 結果設為 NaN（Float64）。
    完成後把 miniRulesX_percentage 設為 0。回傳整個 DataFrame。
    """

    def _dtype_or(col: str, default: pl.DataType) -> pl.DataType:
        return df.schema[col] if col in df.schema else default

    # 取需要的欄位為 Float64（不存在則 None），不做 fill_null，保留缺失
    price = (pl.col("totalPrice") if "totalPrice" in df.columns else pl.lit(None, dtype=pl.Float64)).cast(pl.Float64, strict=False).alias("__price__")
    p0    = (pl.col("miniRules0_percentage") if "miniRules0_percentage" in df.columns else pl.lit(None, dtype=pl.Float64)).cast(pl.Float64, strict=False).alias("__p0__")
    p1    = (pl.col("miniRules1_percentage") if "miniRules1_percentage" in df.columns else pl.lit(None, dtype=pl.Float64)).cast(pl.Float64, strict=False).alias("__p1__")
    a0    = (pl.col("miniRules0_monetaryAmount") if "miniRules0_monetaryAmount" in df.columns else pl.lit(None, dtype=pl.Float64)).cast(pl.Float64, strict=False).alias("__a0__")
    a1    = (pl.col("miniRules1_monetaryAmount") if "miniRules1_monetaryAmount" in df.columns else pl.lit(None, dtype=pl.Float64)).cast(pl.Float64, strict=False).alias("__a1__")

    def bad(x: pl.Expr) -> pl.Expr:
        # True 若 x 為 null 或 NaN
        return (x.is_null() | x.is_nan()).fill_null(False)

    out = (
        df
        .with_columns([price, p0, p1, a0, a1])
        .with_columns([
            # 計算 rule0：若有缺失/NaN -> NaN，否則加總後四捨五入(到個位)但保留 Float64
            pl.when(bad(pl.col("__a0__")) | bad(pl.col("__price__")) | bad(pl.col("__p0__")))
              .then(pl.lit(float("nan")))
              .otherwise( (pl.col("__a0__") + pl.col("__price__") * (pl.col("__p0__") / percent_scale)).round(0) )
              .alias("miniRules0_monetaryAmount"),

            # 計算 rule1：同上
            pl.when(bad(pl.col("__a1__")) | bad(pl.col("__price__")) | bad(pl.col("__p1__")))
              .then(pl.lit(float("nan")))
              .otherwise( (pl.col("__a1__") + pl.col("__price__") * (pl.col("__p1__") / percent_scale)).round(0) )
              .alias("miniRules1_monetaryAmount"),

            # 將 percentage 欄位設為 0（型別沿用原本；若不存在就建立 Float64）
            pl.lit(0).cast(_dtype_or("miniRules0_percentage", pl.Float64)).alias("miniRules0_percentage"),
            pl.lit(0).cast(_dtype_or("miniRules1_percentage", pl.Float64)).alias("miniRules1_percentage"),
        ])
        .drop(["__price__", "__p0__", "__p1__", "__a0__", "__a1__"])
    )

    return out




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
    duration_cols = [
        "legs0_duration", "legs1_duration",
        "legs0_segments0_duration","legs0_segments1_duration",
        "legs0_segments2_duration","legs0_segments3_duration",
        "legs1_segments0_duration","legs1_segments1_duration",
        "legs1_segments2_duration","legs1_segments3_duration",
    ]

    # 文字 "H:MM:SS" -> 分鐘(Float64)；None/"missing" -> NaN
    def to_minutes_expr(col: str) -> pl.Expr:
        s = pl.col(col).cast(pl.Utf8)
        is_missing = s.eq("missing") | pl.col(col).is_null()
        hrs  = s.str.extract(r"^(\d+):", 1).cast(pl.Float64)
        mins = s.str.extract(r":(\d+):", 1).cast(pl.Float64)
        minutes = hrs * 60.0 + mins
        return pl.when(is_missing).then(pl.lit(float("nan"))).otherwise(minutes).alias(col)

    # 只轉存在的欄位
    conv_exprs = [to_minutes_expr(c) for c in duration_cols if c in df.columns]
    if conv_exprs:
        df = df.with_columns(conv_exprs)

    # total_duration：NaN 會自然傳播
    if {"legs0_duration","legs1_duration"}.issubset(df.columns):
        df = df.with_columns(
            (pl.col("legs0_duration") + pl.col("legs1_duration")).alias("total_duration")
        )

    # rank：有值才排，NaN 的列 → rank 也設 NaN（用 Float64）
    def rank_dense_min_nan(src: str, out: str) -> pl.Expr:
        v = pl.col(src)
        r = v.rank(method="dense", descending=False).over("ranker_id")
        return pl.when(v.is_nan()).then(pl.lit(float("nan"))).otherwise(r.cast(pl.Float64)).alias(out)

    rank_targets = [c for c in (duration_cols + ["total_duration"]) if c in df.columns]
    if rank_targets:
        df = df.with_columns([rank_dense_min_nan(c, f"{c}_rank") for c in rank_targets])

    # price_per_duration：只有 total_duration>0 且非 NaN
    if {"total_duration","totalPrice"}.issubset(df.columns):
        df = df.with_columns(
            pl.when(pl.col("total_duration").is_finite() & (pl.col("total_duration") > 0))
              .then(pl.col("totalPrice").cast(pl.Float64) / pl.col("total_duration"))
              .otherwise(pl.lit(float("nan")))
              .alias("price_per_duration")
        )
        df = df.with_columns(
            pl.when(pl.col("price_per_duration").is_nan()).then(pl.lit(float("nan")))
              .otherwise(pl.col("price_per_duration")
                         .rank(method="dense", descending=False)
                         .over("ranker_id")
                         .cast(pl.Float64))
              .alias("price_per_duration_rank")
        )

    # fastest：rank==1 → 1.0；rank 是 NaN → NaN；否則 0.0（全部用 Float64，不產生 NULL）
    fastest_exprs = []
    for base in ["legs0_duration","legs1_duration","total_duration"]:
        r = f"{base}_rank"
        if r in df.columns:
            fastest_exprs.append(
                pl.when(pl.col(r).is_nan()).then(pl.lit(float("nan")))
                 .when(pl.col(r) == 1).then(pl.lit(1.0))
                 .otherwise(0.0)
                 .alias(f"{base}_fastest")
            )
    if fastest_exprs:
        df = df.with_columns(fastest_exprs)

    # 只保留 Id + 本函式產生的欄位
    keep_cols = ["Id"] + [
        c for c in df.columns
        if c not in {"ranker_id","totalPrice"} and (
            c.endswith("_duration") or c.endswith("_rank") or c.endswith("_fastest")
            or c in {"total_duration","price_per_duration","price_per_duration_rank"}
        )
    ]
    out = df.select([c for c in keep_cols if c in df.columns])

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "2_duration_features.parquet")
        out.write_parquet(out_path, compression="zstd")
        print(f"✅ 已輸出 Parquet: {out_path}")

    return out



import polars as pl
import os
import re
from typing import List, Dict

def _parse_duration_to_min(expr: pl.Expr) -> pl.Expr:
    """把 'HH:MM:SS' 轉成分鐘；None/missing -> 0。"""
    return (
        pl.when(expr.is_in([None, "missing"]))
        .then(0)
        .otherwise(
            expr.cast(pl.Utf8)
               .str.extract(r"^(\d+):", 1).cast(pl.Int64).fill_null(0) * 60
            + expr.cast(pl.Utf8)
               .str.extract(r":(\d+):", 1).cast(pl.Int64).fill_null(0)
        )
    )

def _segments_present(df: pl.DataFrame, leg: int) -> List[int]:
    """
    掃欄名自動找出 legs{leg}_segments{idx}_* 存在的 index。
    只要 marketing/operating/duration 任一存在就算有該 segment。
    """
    pat = re.compile(rf"^legs{leg}_segments(\d+)_(marketingCarrier_code|operatingCarrier_code|duration)$")
    seen = set()
    for c in df.columns:
        m = pat.match(c)
        if m:
            seen.add(int(m.group(1)))
    # 若完全沒有也至少回傳 [0]，避免下游空清單（多數資料都有 seg0）
    return sorted(seen) if seen else [0]

def build_frequent_flyer_match_features(
    df: pl.DataFrame,
    output_dir: str = None,
    output_filename: str = "3_frequent_flyer_features.parquet"
) -> pl.DataFrame:
    """
    自動偵測實際存在的 segments，產生：
      - has_frequentFlyer, n_ff_programs, is_vip_freq
      - 各段 carrier 與 frequentFlyer 的 match flags
      - matched/unmatched duration，加總與排名（僅針對存在欄位）
    """
    # 0) FrequentFlyer 基礎欄
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

    # 1) cleaned frequent flyer list（一列一個 list）
    ff_list = (
        pl.col("frequentFlyer")
        .fill_null("")
        .str.replace_all("missing", "")
        .str.split("/")
    )
    # 2) 自動偵測 segments
    segs_leg0 = _segments_present(df, leg=0)
    segs_leg1 = _segments_present(df, leg=1)

    # 3) 產生 carrier match flags（僅對存在欄位）
    flag_exprs = []
    for leg, segs in [(0, segs_leg0), (1, segs_leg1)]:
        for i in segs:
            for carrier_type in ["marketingCarrier_code", "operatingCarrier_code"]:
                colname = f"legs{leg}_segments{i}_{carrier_type}"
                if colname in df.columns:
                    flag_exprs.append(
                        ff_list.list.contains(pl.col(colname).cast(pl.Utf8).fill_null(""))
                              .cast(pl.Int8)
                              .alias(f"{colname}_in_ff")
                    )
    if flag_exprs:
        df = df.with_columns(flag_exprs)

    # 4) 轉換 duration → 分鐘（僅轉存在欄位）
    duration_cols = []
    for leg in (0, 1):
        leg_dur = f"legs{leg}_duration"
        if leg_dur in df.columns:
            duration_cols.append(leg_dur)
    for leg, segs in [(0, segs_leg0), (1, segs_leg1)]:
        for i in segs:
            c = f"legs{leg}_segments{i}_duration"
            if c in df.columns:
                duration_cols.append(c)

    if duration_cols:
        df = df.with_columns([
            _parse_duration_to_min(pl.col(c)).alias(c) for c in duration_cols
        ])

    # 5) total_duration（兩段都存在才做）
    if "legs0_duration" in df.columns and "legs1_duration" in df.columns:
        df = df.with_columns([
            (pl.col("legs0_duration") + pl.col("legs1_duration")).alias("total_duration")
        ])

    # 6) matched duration（僅累計存在欄位）
    def _matched_sum_for_leg(leg: int, seg_indices: List[int]) -> pl.Expr:
        terms = []
        for i in seg_indices:
            dur_col = f"legs{leg}_segments{i}_duration"
            flag_m = f"legs{leg}_segments{i}_marketingCarrier_code_in_ff"
            flag_o = f"legs{leg}_segments{i}_operatingCarrier_code_in_ff"
            if dur_col in df.columns and (flag_m in df.columns or flag_o in df.columns):
                # 缺哪個就當 0
                fm = pl.col(flag_m) if flag_m in df.columns else pl.lit(0, dtype=pl.Int8)
                fo = pl.col(flag_o) if flag_o in df.columns else pl.lit(0, dtype=pl.Int8)
                terms.append(pl.col(dur_col) * ((fm | fo).cast(pl.Int8)))
        return pl.sum_horizontal(terms) if terms else pl.lit(0, dtype=pl.Int64)

    df = df.with_columns([
        _matched_sum_for_leg(0, segs_leg0).alias("legs0_matched_duration_sum"),
        _matched_sum_for_leg(1, segs_leg1).alias("legs1_matched_duration_sum"),
    ])
    df = df.with_columns([
        (pl.col("legs0_matched_duration_sum") + pl.col("legs1_matched_duration_sum")).alias("all_matched_duration_sum")
    ])

    # 7) unmatched（有 total_duration 才能算）
    if "total_duration" in df.columns:
        df = df.with_columns([
            (pl.col("total_duration") - pl.col("all_matched_duration_sum")).alias("unmatched_duration")
        ])

    # 8) 排名（僅對存在欄位）
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
        pl.col("all_matched_duration_sum")
          .rank(method="dense", descending=True)
          .over("ranker_id")
          .cast(pl.Int32)
          .alias("all_matched_duration_sum_rank"),


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
      
    # 9) 輸出
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        df.write_parquet(output_path)
        print(f"✅ 已儲存 Parquet: {output_path}")

    print("✅ 已完成 frequentFlyer / segment 自動偵測版本特徵生成")
    return df

# import polars as pl
# import os

# def build_frequent_flyer_match_features(
#     df: pl.DataFrame,
#     output_dir: str = None,
#     output_filename: str = "3_frequent_flyer_features.parquet"
# ) -> pl.DataFrame:
#     """
#     將 frequentFlyer 與各段航段 carrier_code 比對，建立下列特徵：
#     - has_frequentFlyer
#     - n_ff_programs
#     - carrier match flags
#     - matched/unmatched duration
#     - 各種排名
#     """
#     # 0. FrequentFlyer 衍生特徵
#     df = df.with_columns([
#         pl.col("frequentFlyer").cast(pl.Utf8).fill_null("missing").alias("frequentFlyer")
#     ])
    
#     df = df.with_columns([
#         (
#             (pl.col("frequentFlyer") != "") &
#             (pl.col("frequentFlyer") != "missing")
#         ).cast(pl.Int8).alias("has_frequentFlyer"),
#         (
#             pl.col("frequentFlyer").map_elements(
#                 lambda s: 0 if s in ("", "missing") else s.count("/") + 1,
#                 return_dtype=pl.Int32
#             ).alias("n_ff_programs")
#         )
#     ])
#     df = df.with_columns([
#         (
#             ((pl.col("isVip") == 1) | (pl.col("n_ff_programs") > 0))
#             .cast(pl.Int32)
#             .alias("is_vip_freq")
#         )
#     ])
#     # 1. clean frequentFlyer
#     cleaned_ff = (
#         pl.col("frequentFlyer")
#         .fill_null("")
#         .str.replace_all("missing", "")
#         .str.split("/")
#     )

#     # 2. segments
#     segments = [
#         "legs0_segments0",
#         "legs0_segments1",
#         "legs0_segments2",
#         "legs0_segments3",
#         "legs1_segments0",
#         "legs1_segments1",
#         "legs1_segments2",
#         "legs1_segments3"
#     ]

#     # 3. 是否 in_ff
#     exprs = []
#     for seg in segments:
#         for carrier_type in ["marketingCarrier_code", "operatingCarrier_code"]:
#             carrier_col = f"{seg}_{carrier_type}"
#             exprs.append(
#                 pl.col(carrier_col)
#                 .fill_null("")
#                 .is_in(cleaned_ff)
#                 .cast(pl.Int8)
#                 .alias(f"{carrier_col}_in_ff")
#             )

#     df = df.with_columns(exprs)

#     # 4. duration欄位轉分鐘
#     duration_cols = [
#         "legs0_duration",
#         "legs1_duration",
#         "legs0_segments0_duration",
#         "legs0_segments1_duration",
#         "legs0_segments2_duration",
#         "legs0_segments3_duration",
#         "legs1_segments0_duration",
#         "legs1_segments1_duration",
#         "legs1_segments2_duration",
#         "legs1_segments3_duration"
#     ]

#     duration_exprs = [
#         pl.when(pl.col(c).is_in([None, "missing"]))
#           .then(0)
#           .otherwise(
#               pl.col(c).str.extract(r"^(\d+):", 1).cast(pl.Int64) * 60 +
#               pl.col(c).str.extract(r":(\d+):", 1).cast(pl.Int64)
    #       )
    #       .alias(c)
    #     for c in duration_cols if c in df.columns
    # ]

    # df = df.with_columns(duration_exprs)

    # # 5. total_duration
    # if all(c in df.columns for c in ["legs0_duration", "legs1_duration"]):
    #     df = df.with_columns([
    #         (pl.col("legs0_duration") + pl.col("legs1_duration")).alias("total_duration")
    #     ])

    # # 6. 累積matched duration
    # legs0_matched_duration_sum = pl.sum_horizontal([
    #     pl.col(f"legs0_segments{i}_duration") *
    #     (
    #         pl.col(f"legs0_segments{i}_marketingCarrier_code_in_ff") |
    #         pl.col(f"legs0_segments{i}_operatingCarrier_code_in_ff")
    #     ).cast(pl.Int8)
    #     for i in range(4)
    # ]).alias("legs0_matched_duration_sum")

    # legs1_matched_duration_sum = pl.sum_horizontal([
    #     pl.col(f"legs1_segments{i}_duration") *
    #     (
    #         pl.col(f"legs1_segments{i}_marketingCarrier_code_in_ff") |
    #         pl.col(f"legs1_segments{i}_operatingCarrier_code_in_ff")
    #     ).cast(pl.Int8)
    #     for i in range(4)
    # ]).alias("legs1_matched_duration_sum")

    # df = df.with_columns([
    #     legs0_matched_duration_sum,
    #     legs1_matched_duration_sum,
    # ])
    # df = df.with_columns([

    #     (pl.col("legs0_matched_duration_sum") + pl.col("legs1_matched_duration_sum")).alias("all_matched_duration_sum"),
    # ])
    # # unmatched
    # df = df.with_columns([
    #     (pl.col("total_duration") - pl.col("all_matched_duration_sum")).alias("unmatched_duration")
    # ])

    # # 7. 排名
    # rank_exprs = [
    #     pl.col("legs0_matched_duration_sum")
    #       .rank(method="dense", descending=True)
    #       .over("ranker_id")
    #       .cast(pl.Int32)
    #       .alias("legs0_matched_duration_sum_rank"),

    #     pl.col("legs1_matched_duration_sum")
    #       .rank(method="dense", descending=True)
    #       .over("ranker_id")
    #       .cast(pl.Int32)
    #       .alias("legs1_matched_duration_sum_rank"),

    #     pl.col("unmatched_duration")
    #       .rank(method="dense", descending=False)
    #       .over("ranker_id")
    #       .cast(pl.Int32)
    #       .alias("unmatched_duration_rank")
    # ]

    # duration_rank_exprs = [
    #     pl.col(c)
    #       .rank(method="dense", descending=False)
    #       .over("ranker_id")
    #       .cast(pl.Int32)
    #       .alias(f"{c}_rank")
    #     for c in (duration_cols + ["total_duration"]) if c in df.columns
    # ]

    # df = df.with_columns(rank_exprs + duration_rank_exprs)

    # # 8. 輸出
    # if output_dir:
    #     os.makedirs(output_dir, exist_ok=True)
    #     output_path = os.path.join(output_dir, output_filename)
    #     df.write_parquet(output_path)
    #     print(f"✅ 已儲存 Parquet: {output_path}")

    # print("✅ 已完成 frequentFlyer 特徵 + match 特徵 + duration 特徵生成")
    # return df

import polars as pl
import os, re, math
from typing import List

def build_cabin_features(
    df: pl.DataFrame,
    output_dir: str = None,
    output_filename: str = "5_cabin_features.parquet"
) -> pl.DataFrame:
    # ---------- helpers ----------
    def _find_segment_indices(df: pl.DataFrame, leg: int) -> List[int]:
        pat = re.compile(rf"^legs{leg}_segments(\d+)_(?:cabinClass|duration)$")
        idx = set()
        for c in df.columns:
            m = pat.match(c)
            if m:
                idx.add(int(m.group(1)))
        return sorted(idx)

    def _dur_to_minutes_expr(col: str) -> pl.Expr:
        """'H:MM:SS' → 分鐘；缺失/無法解析 → NaN (Float64)。數值型直接轉 Float64。"""
        dtype = df.schema.get(col)
        if dtype and dtype.is_numeric():
            return pl.col(col).cast(pl.Float64, strict=False).alias(col)

        s    = pl.col(col).cast(pl.Utf8)
        hrs  = s.str.extract(r"^(\d+):", 1).cast(pl.Float64)
        mins = s.str.extract(r":(\d+):", 1).cast(pl.Float64)
        minutes = hrs * 60.0 + mins
        return (
            pl.when(s.is_null() | (s == "missing") | hrs.is_null() | mins.is_null())
              .then(pl.lit(float("nan")))
              .otherwise(minutes)
              .alias(col)
        )

    def _leg_stats(dur_cols: List[str], cab_cols: List[str], leg_name: str):
        """
        回傳該腿的：
          - ok：所有 segment 要嘛同時 NaN、要嘛同時非 NaN，且至少有一個 pair 同時非 NaN
          - wmean：只用 pair 同時有值且 cabin>0 的加權平均（否則 NaN）
          - mean ：只用 cabin>0 的平均（否則 NaN）
          - max_cabin：在 pair 同時有值中，挑最大 duration 的 cabin（否則 NaN）
        """
        # 若沒有欄位，全部回 NaN/False
        if not dur_cols or not cab_cols:
            return (
                pl.lit(False).alias(f"{leg_name}_ok"),
                pl.lit(float("nan")).alias(f"{leg_name}_weighted_mean_cabin"),
                pl.lit(float("nan")).alias(f"{leg_name}_mean_cabin"),
                pl.lit(float("nan")).alias(f"{leg_name}_max_duration_cabin"),
            )

        # pair 的 present / mismatch
        present_flags = []
        mismatch_flags = []

        # 加權平均用的項
        num_terms = []
        den_terms = []

        # 取最長 duration 的 cabin 用
        dur_for_max = []
        cab_for_max = []

        for d, c in zip(dur_cols, cab_cols):
            if d not in df.columns or c not in df.columns:
                # 缺任一側視為「失配」
                mismatch_flags.append(pl.lit(1))
                present_flags.append(pl.lit(0))
                # 填空白項，避免空清單
                num_terms.append(pl.lit(0.0))
                den_terms.append(pl.lit(0.0))
                dur_for_max.append(pl.lit(float("-inf")))
                cab_for_max.append(pl.lit(float("nan")))
                continue

            dval = pl.col(d).cast(pl.Float64, strict=False)
            cval = pl.col(c).cast(pl.Float64, strict=False)

            d_present = ~(dval.is_null() | dval.is_nan())
            c_present = ~(cval.is_null() | cval.is_nan())

            pair_present  = (d_present & c_present).cast(pl.Int8)
            pair_mismatch = (d_present ^ c_present).cast(pl.Int8)

            present_flags.append(pair_present)
            mismatch_flags.append(pair_mismatch)

            # 只在 pair_present 且 cabin > 0 的情況用於加權平均
            num_terms.append(pl.when((pair_present == 1) & (cval > 0)).then(dval * cval).otherwise(0.0))
            den_terms.append(pl.when((pair_present == 1) & (cval > 0)).then(dval).otherwise(0.0))

            # 取最大 duration 的 cabin：沒同時有值就不要參與比較（用 -inf 遮蔽）
            dur_for_max.append(pl.when(pair_present == 1).then(dval).otherwise(float("-inf")))
            cab_for_max.append(pl.when(pair_present == 1).then(cval).otherwise(float("nan")))

        n_present   = pl.sum_horizontal(present_flags)
        n_mismatch  = pl.sum_horizontal(mismatch_flags)
        ok          = ((n_present > 0) & (n_mismatch == 0)).alias(f"{leg_name}_ok")

        numerator   = pl.sum_horizontal(num_terms)
        denominator = pl.sum_horizontal(den_terms)
        wmean = (
            pl.when(ok & (denominator > 0))
              .then(numerator / denominator)
              .otherwise(pl.lit(float("nan")))
              .cast(pl.Float64)
              .alias(f"{leg_name}_weighted_mean_cabin")
        )

        # 平均艙等（只看 cabin>0 且 pair 同時有值的 cabin）
        mean = (
            pl.when(ok)
              .then(
                  pl.concat_list([
                      pl.when((pf == 1) & (pl.col(c).cast(pl.Float64, strict=False) > 0))
                        .then(pl.col(c).cast(pl.Float64, strict=False))
                        .otherwise(None)
                      for pf, c in zip(present_flags, cab_cols)
                  ])
                  .list.drop_nulls()
                  .list.mean()
              )
              .otherwise(pl.lit(float("nan")))
              .cast(pl.Float64)
              .alias(f"{leg_name}_mean_cabin")
        )

        # 最大 duration 的艙等
        max_dur = pl.max_horizontal(dur_for_max)
        # 從所有 pair 中挑 equals max_dur 的 cabin，coalesce 取第一個
        pick_cabs = [pl.when(max_dur == d).then(c) for d, c in zip(dur_for_max, cab_for_max)]
        max_cabin = (
            pl.when(ok & max_dur.is_finite())
              .then(pl.coalesce(pick_cabs))
              .otherwise(pl.lit(float("nan")))
              .cast(pl.Float64)
              .alias(f"{leg_name}_max_duration_cabin")
        )

        return ok, wmean, mean, max_cabin

    # ---------- discover segments ----------
    segs_leg0 = _find_segment_indices(df, 0)
    segs_leg1 = _find_segment_indices(df, 1)
    legs0_cabin_cols    = [f"legs0_segments{i}_cabinClass" for i in segs_leg0]
    legs1_cabin_cols    = [f"legs1_segments{i}_cabinClass" for i in segs_leg1]
    legs0_duration_cols = [f"legs0_segments{i}_duration"    for i in segs_leg0]
    legs1_duration_cols = [f"legs1_segments{i}_duration"    for i in segs_leg1]

    # ---------- durations → minutes (保留 NaN) ----------
    duration_cols = [c for c in (legs0_duration_cols + legs1_duration_cols) if c in df.columns]
    if duration_cols:
        df = df.with_columns([_dur_to_minutes_expr(c) for c in duration_cols])

    # ---------- cabin → Float64 (保留 NaN；不存在就留空不建新欄) ----------
    for c in (legs0_cabin_cols + legs1_cabin_cols):
        if c in df.columns:
            df = df.with_columns(pl.col(c).cast(pl.Float64, strict=False))

    # ---------- per-leg stats（嚴格對齊規則） ----------
    ok0, wmean0, mean0, max0 = _leg_stats(legs0_duration_cols, legs0_cabin_cols, "legs0")
    ok1, wmean1, mean1, max1 = _leg_stats(legs1_duration_cols, legs1_cabin_cols, "legs1")

    # 單/雙腿判斷
    route_str = pl.col("searchRoute").cast(pl.Utf8)
    is_round_trip = pl.when(route_str.is_null() | (route_str == "missing")).then(None)\
                      .otherwise(route_str.str.contains("/"))

    # total_weighted_mean：
    #   單趟 → = legs0_weighted_mean_cabin
    #   來回 → 兩腿 ok 時，對「兩腿同時有值的 pair」做一次總加權；否則 NaN
    # 先做兩腿的「同時有值 & cabin>0」的總和（row-wise）
    both_num_terms = []
    both_den_terms = []
    for d, c in zip(legs0_duration_cols + legs1_duration_cols,
                    legs0_cabin_cols    + legs1_cabin_cols):
        if d in df.columns and c in df.columns:
            dval = pl.col(d).cast(pl.Float64, strict=False)
            cval = pl.col(c).cast(pl.Float64, strict=False)
            d_present = ~(dval.is_null() | dval.is_nan())
            c_present = ~(cval.is_null() | cval.is_nan())
            pair_present = d_present & c_present
            both_num_terms.append(pl.when(pair_present & (cval > 0)).then(dval * cval).otherwise(0.0))
            both_den_terms.append(pl.when(pair_present & (cval > 0)).then(dval).otherwise(0.0))

    total_num = pl.sum_horizontal(both_num_terms) if both_num_terms else pl.lit(0.0)
    total_den = pl.sum_horizontal(both_den_terms) if both_den_terms else pl.lit(0.0)
    total_wmean_round = (
        pl.when(ok0 & ok1 & (total_den > 0))
          .then(total_num / total_den)
          .otherwise(pl.lit(float("nan")))
          .cast(pl.Float64)
    )

    total_wmean = (
        pl.when(is_round_trip == False).then(wmean0)         # 單趟：等於 legs0_weighted_mean_cabin
         .when(is_round_trip == True).then(total_wmean_round)
         .otherwise(pl.lit(float("nan")))
         .alias("total_weighted_mean_cabin")
    )

    # is_max_duration_cabin_same：
    same_max = (
        pl.when(is_round_trip == False).then(pl.lit(1))  # 單趟直接視為相同
         .when(is_round_trip == True).then(
             (max0 == max1).cast(pl.Int8)
         )
         .otherwise(pl.lit(None, dtype=pl.Int8))
         .alias("is_max_duration_cabin_same")
    )

    # ---------- assemble ----------
    df = df.with_columns([wmean0, wmean1, mean0, mean1, max0, max1, total_wmean, same_max])

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        df.write_parquet(output_path)
        print(f"✅ 已儲存 Parquet: {output_path}")

    return df



import os
import polars as pl

# def build_cabin_features(
#     df: pl.DataFrame,
#     output_dir: str = None,
#     output_filename: str = "5_cabin_features.parquet"
# ) -> pl.DataFrame:
#     # 欄位
#     legs0_cabin_cols = [
#         "legs0_segments0_cabinClass",
#         "legs0_segments1_cabinClass",
#         "legs0_segments2_cabinClass",
#         "legs0_segments3_cabinClass"
#     ]
#     legs1_cabin_cols = [
#         "legs1_segments0_cabinClass",
#         "legs1_segments1_cabinClass",
#         "legs1_segments2_cabinClass",
#         "legs1_segments3_cabinClass"
#     ]
#     legs0_duration_cols = [
#         "legs0_segments0_duration",
#         "legs0_segments1_duration",
#         "legs0_segments2_duration",
#         "legs0_segments3_duration"
#     ]
#     legs1_duration_cols = [
#         "legs1_segments0_duration",
#         "legs1_segments1_duration",
#         "legs1_segments2_duration",
#         "legs1_segments3_duration"
#     ]

#     duration_cols = legs0_duration_cols + legs1_duration_cols

#     # Step1: duration欄位轉分鐘 (字串 "H:MM:SS")
#     duration_exprs = [
#         pl.when(pl.col(c).is_in([None, "missing"]))
#           .then(0)
#           .otherwise(
#               pl.col(c).str.extract(r"^(\d+):", 1).cast(pl.Int64) * 60 +
#               pl.col(c).str.extract(r":(\d+):", 1).cast(pl.Int64)
#           )
#           .alias(c)
#         for c in duration_cols if c in df.columns
#     ]
#     if duration_exprs:
#         df = df.with_columns(duration_exprs)

#     # Step2: cabin欄位轉 Int64（缺欄位則先補 0 再轉型，避免 ColumnNotFound）
#     for c in (legs0_cabin_cols + legs1_cabin_cols):
#         if c not in df.columns:
#             df = df.with_columns(pl.lit(0).alias(c))
#         df = df.with_columns(pl.col(c).cast(pl.Int64))

#     # 平均艙等（只計算 >0 值）
#     legs0_mean = (
#         pl.concat_list([pl.col(c) for c in legs0_cabin_cols])
#         .list.eval(pl.element().filter(pl.element() > 0))
#         .list.mean()
#         .fill_null(0)
#         .alias("legs0_mean_cabin")
#     )
#     legs1_mean = (
#         pl.concat_list([pl.col(c) for c in legs1_cabin_cols])
#         .list.eval(pl.element().filter(pl.element() > 0))
#         .list.mean()
#         .fill_null(0)
#         .alias("legs1_mean_cabin")
#     )

#     # legs0/legs1 艙等集合是否相同（去重後）
#     is_same_cabin = (
#         (
#             pl.concat_list([pl.col(c) for c in legs0_cabin_cols]).list.unique().list.sort()
#             ==
#             pl.concat_list([pl.col(c) for c in legs1_cabin_cols]).list.unique().list.sort()
#         ).cast(pl.Int8)
#         .alias("is_legs0_legs1_cabin_same")
#     )

#     # 最長 segment 的艙等（保留你的 UDF 作法即可；這段通常不會觸發你看到的錯誤）
#     def longest_segment_idx(durations):
#         if all(d is None for d in durations):
#             return None
#         idx = max(
#             ((i, int(d) if d is not None else -1) for i, d in enumerate(durations)),
#             key=lambda x: x[1]
#         )[0]
#         return idx

#     def max_duration_cabin(row, dur_cols, cabin_cols):
#         durations = [row[c] for c in dur_cols]
#         cabins = [row[c] for c in cabin_cols]
#         idx = longest_segment_idx(durations)
#         if idx is None:
#             return 0
#         return cabins[idx] if cabins[idx] is not None else 0

#     df = df.with_columns([
#         pl.struct(legs0_duration_cols + legs0_cabin_cols)
#           .map_elements(lambda row: max_duration_cabin(row, legs0_duration_cols, legs0_cabin_cols))
#           .alias("legs0_max_duration_cabin"),
#         pl.struct(legs1_duration_cols + legs1_cabin_cols)
#           .map_elements(lambda row: max_duration_cabin(row, legs1_duration_cols, legs1_cabin_cols))
#           .alias("legs1_max_duration_cabin"),
#     ])

#     df = df.with_columns(
#         (pl.col("legs0_max_duration_cabin") == pl.col("legs1_max_duration_cabin"))
#         .cast(pl.Int8)
#         .alias("is_max_duration_cabin_same")
#     )

#     # === 這裡是重點：改用純表達式計算「加權平均艙等」 ===
#     def weighted_mean_expr(dur_cols, cabin_cols, out_name: str):
#         # 分子：sum(duration_j * cabin_j  where cabin_j > 0)
#         num_terms = [
#             (pl.col(dur) * pl.when(pl.col(cab) > 0).then(pl.col(cab)).otherwise(0))
#             for dur, cab in zip(dur_cols, cabin_cols)
#         ]
#         numerator = pl.sum_horizontal(num_terms)

#         # 分母：sum(duration_j where cabin_j > 0)
#         den_terms = [
#             pl.when(pl.col(cab) > 0).then(pl.col(dur)).otherwise(0)
#             for dur, cab in zip(dur_cols, cabin_cols)
#         ]
#         denominator = pl.sum_horizontal(den_terms)

#         return (
#             pl.when(denominator > 0)
#               .then(numerator / denominator)
#               .otherwise(0.0)
#               .cast(pl.Float64)
#               .alias(out_name)
#         )

#     df = df.with_columns([
#         weighted_mean_expr(legs0_duration_cols, legs0_cabin_cols, "legs0_weighted_mean_cabin"),
#         weighted_mean_expr(legs1_duration_cols, legs1_cabin_cols, "legs1_weighted_mean_cabin"),
#         weighted_mean_expr(legs0_duration_cols + legs1_duration_cols,
#                            legs0_cabin_cols + legs1_cabin_cols,
#                            "total_weighted_mean_cabin"),
#         legs0_mean,
#         legs1_mean,
#         is_same_cabin,
#     ])

#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)
#         output_path = os.path.join(output_dir, output_filename)
#         df.write_parquet(output_path)
#         print(f"✅ 已儲存 Parquet: {output_path}")

#     return df

import polars as pl
import os
import os
import polars as pl

def build_baggage_fee_features(
    df: pl.DataFrame,
    output_dir: str = None,
    output_filename: str = "4_baggage_fee_features.parquet"
) -> pl.DataFrame:
    NAN = float("nan")

    # 小工具：把欄位轉 Float64，null -> NaN；欄位不存在就整欄 NaN
    def fcol(name: str) -> pl.Expr:
        return (
            pl.col(name).cast(pl.Float64, strict=False).fill_null(NAN)
            if name in df.columns else pl.lit(NAN)
        )

    # === 原子欄位（浮點＋NaN）===
    b0 = fcol("legs0_segments0_baggageAllowance_quantity")
    b1 = fcol("legs1_segments0_baggageAllowance_quantity")
    a0 = fcol("miniRules0_monetaryAmount")
    a1 = fcol("miniRules1_monetaryAmount")
    tp = fcol("totalPrice")

    s0 = (
        pl.col("miniRules0_statusInfos").cast(pl.Int64, strict=False)
        if "miniRules0_statusInfos" in df.columns else pl.lit(None, dtype=pl.Int64)
    )
    s1 = (
        pl.col("miniRules1_statusInfos").cast(pl.Int64, strict=False)
        if "miniRules1_statusInfos" in df.columns else pl.lit(None, dtype=pl.Int64)
    )

    # === 派生 ===
    baggage_total = (b0 + b1).alias("baggage_total")
    total_fees    = (a0 + a1).alias("total_fees")

    # has_baggage: NaN -> NaN；>0 -> 1.0；else 0.0
    has_baggage = (
        pl.when(pl.col("baggage_total").is_nan()).then(pl.lit(NAN))
         .when(pl.col("baggage_total") > 0).then(1.0)
         .otherwise(0.0)
         .alias("has_baggage")
    )

    # has_fees: 同理
    has_fees = (
        pl.when(pl.col("total_fees").is_nan()).then(pl.lit(NAN))
         .when(pl.col("total_fees") > 0).then(1.0)
         .otherwise(0.0)
         .alias("has_fees")
    )

    # 免費取消/改票：若金額或狀態缺失 -> NaN；否則 1.0/0.0
    free_cancel = (
        pl.when(a0.is_nan() | s0.is_null()).then(pl.lit(NAN))
         .otherwise(((a0 == 0.0) & (s0 == 1)).cast(pl.Float64))
         .alias("free_cancel")
    )
    free_exchange = (
        pl.when(a1.is_nan() | s1.is_null()).then(pl.lit(NAN))
         .otherwise(((a1 == 0.0) & (s1 == 1)).cast(pl.Float64))
         .alias("free_exchange")
    )

    # 價格衍生：NaN 自然傳播
    price_per_fee   = (tp / (pl.col("total_fees") + 1)).alias("price_per_fee")
    price_minus_fee = (tp - pl.col("total_fees")).alias("price_minus_fee")

    # 寫入主要欄位
    df = df.with_columns([baggage_total, total_fees, free_cancel, free_exchange])
    df = df.with_columns([has_baggage, has_fees, price_per_fee, price_minus_fee])

    # === 排名（NaN -> rank 也 NaN）===
    def rank_dense_desc_nan(src: str, out: str) -> pl.Expr:
        v = pl.col(src)
        r = v.rank(method="dense", descending=True).over("ranker_id")
        return pl.when(v.is_nan()).then(pl.lit(NAN)).otherwise(r.cast(pl.Float64)).alias(out)

    def rank_dense_asc_nan(src: str, out: str) -> pl.Expr:
        v = pl.col(src)
        r = v.rank(method="dense", descending=False).over("ranker_id")
        return pl.when(v.is_nan()).then(pl.lit(NAN)).otherwise(r.cast(pl.Float64)).alias(out)

    rank_exprs = []
    if "baggage_total"   in df.columns: rank_exprs.append(rank_dense_desc_nan("baggage_total",   "baggage_total_rank"))
    if "price_per_fee"   in df.columns: rank_exprs.append(rank_dense_desc_nan("price_per_fee",   "price_per_fee_rank"))
    if "price_minus_fee" in df.columns: rank_exprs.append(rank_dense_desc_nan("price_minus_fee", "price_minus_fee_rank"))
    if "total_fees"      in df.columns: rank_exprs.append(rank_dense_asc_nan ("total_fees",      "total_fees_rank"))

    if rank_exprs:
        df = df.with_columns(rank_exprs)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        df.write_parquet(output_path, compression="zstd")
        print(f"✅ 已儲存 Parquet: {output_path}")

    return df

def build_time_features(
    df: pl.DataFrame,
    output_dir: str = None,
    output_filename: str = "6_time_features.parquet"
) -> pl.DataFrame:
    def parse_dt(colname: str) -> pl.Expr:
        s = pl.col(colname).cast(pl.Utf8)
        return pl.when(s == "missing").then(None).otherwise(s).str.to_datetime(strict=False)

    nan = float("nan")

    # === legs0/legs1 衍生時間欄（缺值→NaN，型別 Float64）===
    time_cols = ["legs0_departureAt", "legs0_arrivalAt", "legs1_departureAt", "legs1_arrivalAt"]
    time_exprs = []
    for col in time_cols:
        if col in df.columns:
            dt = parse_dt(col)
            h  = dt.dt.hour()

            period = (
                pl.when(h.is_between(0, 5)).then(0)
                 .when(h.is_between(6, 11)).then(1)
                 .when(h.is_between(12, 17)).then(2)
                 .when(h.is_between(18, 23)).then(3)
            )

            time_exprs.extend([
                h.cast(pl.Float64).fill_null(nan).alias(f"{col}_hour"),
                dt.dt.weekday().cast(pl.Float64).fill_null(nan).alias(f"{col}_weekday"),
                (((h >= 6) & (h <= 9)) | ((h >= 17) & (h <= 20))).cast(pl.Float64).fill_null(nan).alias(f"{col}_business_time"),
                period.cast(pl.Float64).fill_null(nan).alias(f"{col}_day_period"),
                ( (dt.dt.weekday() >= 5).cast(pl.Float64).fill_null(nan) ).alias(f"{col}_is_weekend"),
            ])

    # === requestDate 衍生欄（缺值→NaN，Float64）===
    if "requestDate" in df.columns:
        req_dt = parse_dt("requestDate")
        req_h  = req_dt.dt.hour()
        req_period = (
            pl.when(req_h.is_between(0, 5)).then(0)
             .when(req_h.is_between(6, 11)).then(1)
             .when(req_h.is_between(12, 17)).then(2)
             .when(req_h.is_between(18, 23)).then(3)
        )

        time_exprs.extend([
            req_h.cast(pl.Float64).fill_null(nan).alias("requestDate_hour"),
            req_dt.dt.weekday().cast(pl.Float64).fill_null(nan).alias("requestDate_weekday"),
            (((req_h >= 6) & (req_h <= 9)) | ((req_h >= 17) & (req_h <= 20))).cast(pl.Float64).fill_null(nan).alias("requestDate_business_time"),
            req_period.cast(pl.Float64).fill_null(nan).alias("requestDate_day_period"),
            ( (req_dt.dt.weekday() >= 5).cast(pl.Float64).fill_null(nan) ).alias("requestDate_is_weekend"),
        ])
    else:
        req_dt = pl.lit(None, dtype=pl.Datetime)

    # === 先把 is_round_trip 寫進 df（用 searchRoute 判斷）===
    if "searchRoute" in df.columns:
        is_round_trip_expr = (
            pl.col("searchRoute").cast(pl.Utf8)
            .str.contains("/", literal=True)
            .fill_null(False)
            .cast(pl.Int8)
            .alias("is_round_trip")
        )
    else:
        # 後備：看 legs1 是否有值（仍先寫進 df）
        is_round_trip_expr = (
            (
                (pl.col("legs1_departureAt").cast(pl.Utf8).is_not_null() & (pl.col("legs1_departureAt").cast(pl.Utf8) != "missing"))
                | (pl.col("legs1_arrivalAt").cast(pl.Utf8).is_not_null() & (pl.col("legs1_arrivalAt").cast(pl.Utf8) != "missing"))
            )
            .cast(pl.Int8)
            .alias("is_round_trip")
        )

    df = df.with_columns(is_round_trip_expr)

    # === 天數計算 ===
    depart_dt = parse_dt("legs0_departureAt") if "legs0_departureAt" in df.columns else pl.lit(None, dtype=pl.Datetime)
    arrive_dt = parse_dt("legs1_arrivalAt")   if "legs1_arrivalAt"   in df.columns else pl.lit(None, dtype=pl.Datetime)

    day_ms = 1000 * 60 * 60 * 24
    days_between_calc = (
        ((arrive_dt - depart_dt).dt.total_milliseconds() / day_ms)
        .floor()
        .cast(pl.Float64)
    )
    # round trip: 缺值→NaN；單趟：0.0
    days_between = (
        pl.when(pl.col("is_round_trip") == 1)
          .then(pl.when(days_between_calc.is_null()).then(pl.lit(nan)).otherwise(days_between_calc))
          .otherwise(pl.lit(0.0))
          .alias("days_between_departure_arrival")
    )

    # 預訂 ~ 出發：缺值→NaN
    if "requestDate" in df.columns:
        days_before_calc = (
            ((depart_dt - req_dt).dt.total_milliseconds() / day_ms)
            .floor()
            .cast(pl.Float64)
        )
        days_before_departure = (
            pl.when(days_before_calc.is_null()).then(pl.lit(nan)).otherwise(days_before_calc)
            .alias("days_before_departure")
        )
    else:
        days_before_departure = pl.lit(nan).alias("days_before_departure")

    # === 寫入其他衍生欄 ===
    df = df.with_columns(time_exprs + [days_between, days_before_departure])

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        df.write_parquet(output_path, compression="zstd")
        print(f"✅ 已儲存 Parquet: {output_path}")

    return df

import polars as pl
import os

def build_corporate_access_route_features(
    df: pl.DataFrame,
    output_dir: str = None,
    output_filename: str = "7_corporate_access_route_features.parquet"
) -> pl.DataFrame:
    """
    產生三個欄位（Float64；缺失→NaN）：
      - has_corporate_tariff: corporateTariffCode 有值→1.0；缺失→NaN
      - has_access_tp       : pricingInfo_isAccessTP==1 →1.0；==0/!=1 →0.0；缺失→NaN
      - is_popular_route    : searchRoute 在白名單 →1.0；不在→0.0；缺失/“MISSING”/空字串→NaN
    """

    # --- corporate tariff ---
    corp = pl.col("corporateTariffCode").cast(pl.Float64, strict=False)
    has_corporate_tariff = (
        pl.when(corp.is_null() | corp.is_nan())
          .then(pl.lit(float("nan")))
          .otherwise(pl.lit(1.0))
          .alias("has_corporate_tariff")
    )

    # --- access TP ---
    acc = pl.col("pricingInfo_isAccessTP").cast(pl.Float64, strict=False)
    has_access_tp = (
        pl.when(acc.is_null() | acc.is_nan())
          .then(pl.lit(float("nan")))
          .otherwise((acc == 1.0).cast(pl.Float64))
          .alias("has_access_tp")
    )

    # --- popular route（大小寫不敏感；"MISSING"/空字串/Null 視為缺失→NaN）---
    popular_routes = {"MOWLED/LEDMOW", "LEDMOW/MOWLED", "MOWLED", "LEDMOW"}
    route = pl.col("searchRoute").cast(pl.Utf8).str.strip_chars().str.to_uppercase()
    route_missing = route.is_null() | (route == "") | (route == "MISSING")

    is_popular_route = (
        pl.when(route_missing)
          .then(pl.lit(float("nan")))
          .otherwise(route.is_in(list(popular_routes)).cast(pl.Float64))
          .alias("is_popular_route")
    )

    df = df.with_columns([has_corporate_tariff, has_access_tp, is_popular_route])

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, output_filename)
        df.write_parquet(out_path, compression="zstd")
        print(f"✅ 已儲存 Parquet: {out_path}")

    print("✅ 已完成 corporate/access/route 特徵生成（缺失值保留為 NaN）")
    return df

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


import polars as pl
from typing import Optional

def build_carrier_consistency_features(
    df: pl.DataFrame,
    output_dir: str = None,
    output_filename: str = "9_carrier_consistency_features.parquet",
    transform_config: Optional[dict] = None
) -> pl.DataFrame:
    """
    建立 legs0/legs1 主 Carrier 一致性特徵 (自動先計算轉機次數)。
    - 單趟（searchRoute 不含 '/'）→ both_legs_carrier_all_same = 1
    - 使用 transform_config encoding 時，unseen/缺值保持 null，不再補 -1
    """

    # 判斷是否單趟 / 來回（★ 單趟/來回旗標）
    route_str = pl.col("searchRoute").cast(pl.Utf8)
    is_single_trip = (
        pl.when(route_str.is_null() | (route_str == "missing"))
         .then(None)  # 無法判定 → None
         .otherwise(~route_str.str.contains("/"))  # True=單趟, False=來回
         .alias("_is_single_trip")
    )
    df = df.with_columns(is_single_trip)

    # legs0/legs1 轉機次數
    legs0_segment_cols = [
        "legs0_segments1_departureFrom_airport_iata",
        "legs0_segments2_departureFrom_airport_iata",
        "legs0_segments3_departureFrom_airport_iata",
    ]
    legs1_segment_cols = [
        "legs1_segments1_departureFrom_airport_iata",
        "legs1_segments2_departureFrom_airport_iata",
        "legs1_segments3_departureFrom_airport_iata",
    ]

    legs0_num_transfers = (
        pl.sum_horizontal([
            ((pl.col(c).is_not_null()) & (pl.col(c) != "missing")).cast(pl.Int8)
            for c in legs0_segment_cols if c in df.columns
        ]).alias("legs0_num_transfers")
    )
    legs1_num_transfers = (
        pl.sum_horizontal([
            ((pl.col(c).is_not_null()) & (pl.col(c) != "missing")).cast(pl.Int8)
            for c in legs1_segment_cols if c in df.columns
        ]).alias("legs1_num_transfers")
    )
    df = df.with_columns([legs0_num_transfers, legs1_num_transfers])

    # 方便用的「是否 SU/S7」旗標（你原本就有）
    df = df.with_columns(
        pl.col("legs0_segments0_marketingCarrier_code").is_in(["SU", "S7"])
          .cast(pl.Int32).alias("is_major_carrier")
    )

    # 主 carrier（第一個非空的 marketingCarrier_code）
    legs0_marketing_cols = [
        f"legs0_segments{s}_marketingCarrier_code"
        for s in range(4) if f"legs0_segments{s}_marketingCarrier_code" in df.columns
    ]
    legs1_marketing_cols = [
        f"legs1_segments{s}_marketingCarrier_code"
        for s in range(4) if f"legs1_segments{s}_marketingCarrier_code" in df.columns
    ]
    df = df.with_columns([
        pl.coalesce([pl.col(c) for c in legs0_marketing_cols]).alias("legs0_main_carrier"),
        pl.coalesce([pl.col(c) for c in legs1_marketing_cols]).alias("legs1_main_carrier"),
    ])

    # 各腿是否全相同 carrier
    legs0_all_same = (
        pl.when(pl.col("legs0_num_transfers") == 0)
         .then(1)
         .otherwise(
             pl.all_horizontal([
                 (pl.col(c) == pl.col("legs0_main_carrier")) & pl.col(c).is_not_null()
                 for c in legs0_marketing_cols
             ]).cast(pl.Int8)
         ).alias("legs0_all_segments_carrier_same")
    )
    legs1_all_same = (
        pl.when(pl.col("legs1_num_transfers") == 0)
         .then(1)
         .otherwise(
             pl.all_horizontal([
                 (pl.col(c) == pl.col("legs1_main_carrier")) & pl.col(c).is_not_null()
                 for c in legs1_marketing_cols
             ]).cast(pl.Int8)
         ).alias("legs1_all_segments_carrier_same")
    )
    df = df.with_columns([legs0_all_same, legs1_all_same])

    # 兩腿一致性（★ 單趟直接 1；來回才用你原本條件；未知保持原條件結果）
    both_legs_all_same = (
        pl.when(pl.col("_is_single_trip") == True)
         .then(pl.lit(1, dtype=pl.Int8))
         .otherwise(
             (
                 (pl.col("legs0_all_segments_carrier_same") == 1) &
                 (pl.col("legs1_all_segments_carrier_same") == 1) &
                 (pl.col("legs0_main_carrier") == pl.col("legs1_main_carrier")) &
                 pl.col("legs0_main_carrier").is_not_null() &
                 pl.col("legs1_main_carrier").is_not_null()
             ).cast(pl.Int8)
         )
         .alias("both_legs_carrier_all_same")
    )
    df = df.with_columns(both_legs_all_same)

    # 可選：共用 encoding（★ 不再 fill_null(-1)，維持 null）
    if transform_config:
        # carrier 共用 encoder
        carrier_enc = transform_config["label_encoders"]["carrier_cols"]
        mapping_df = pl.DataFrame({
            "value": carrier_enc["values"],
            "rank_id": carrier_enc["codes"],
        })
        cols_to_encode = ["legs0_main_carrier", "legs1_main_carrier"]
        # 若也想同時編碼出發機場可加上（存在才會處理）
        for dep_col in ["legs0_segments0_departureFrom_airport_iata",
                        "legs1_segments0_departureFrom_airport_iata"]:
            if dep_col in df.columns:
                cols_to_encode.append(dep_col)

        for col in cols_to_encode:
            # 左連接到 mapping；未出現過的值/缺值 → rank_id = null（★關鍵）
            mapped = (
                df.select([col])
                  .with_columns(pl.col(col).cast(pl.Utf8))
                  .join(mapping_df.rename({"value": col}), on=col, how="left")
                  .with_columns(
                      pl.col("rank_id").cast(pl.Int32).alias(f"{col}_encoded")
                  )
                  .drop([col, "rank_id"])
                  .rename({f"{col}_encoded": col})
            )
            # 以 row 對齊加到 df；保留 null
            df = df.with_columns(mapped)

    # 清理臨時欄位
    df = df.drop([    "ranker_id","searchRoute",
    # legs0 轉機判斷
    "legs0_segments1_departureFrom_airport_iata",
    "legs0_segments2_departureFrom_airport_iata",
    "legs0_segments3_departureFrom_airport_iata",
    # legs1 轉機判斷
    "legs1_segments1_departureFrom_airport_iata",
    "legs1_segments2_departureFrom_airport_iata",
    "legs1_segments3_departureFrom_airport_iata",
    # legs0 Marketing
    "legs0_segments0_marketingCarrier_code",
    "legs0_segments1_marketingCarrier_code",
    # legs1 Marketing
    "legs1_segments0_marketingCarrier_code",
    "legs1_segments1_marketingCarrier_code",])

    # 輸出（可選）
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, output_filename)
        df.write_parquet(out_path)
        print(f"✅ 已儲存 Parquet: {out_path}")

    print("✅ 主Carrier一致性/轉機次數特徵完成（單趟直接一致；encoding 未補 -1）")
    return df

import os
import pickle
from typing import Optional, Dict, Tuple
import polars as pl

def build_label_encoding_features(
    df: pl.DataFrame,
    output_dir: Optional[str] = None,
    transform_config: Optional[Dict] = None,
    id_col: str = "Id",
    selected_col: str = "selected"
) -> Tuple[pl.DataFrame, dict]:
    """
    Label Encoding + carrier_pop (carrier0_pop/carrier1_pop)。
    - 缺失策略：null、空字串、"missing" 一律視為缺失，保持為 NaN，不編碼為 -1。
    - train：產生映射與 popularity 字典；test：套用 transform_config。
    """
    if id_col not in df.columns:
        raise ValueError(f"'{id_col}' 不存在於 df.columns")

    all_cols = df.columns
    aircraft_cols = [c for c in all_cols if c.endswith("_aircraft_code")]
    flightnum_cols = [c for c in all_cols if c.endswith("_flightNumber")]
    airport_cols = [c for c in all_cols if "_arrivalTo_airport_" in c or "_departureFrom_airport_" in c]
    carrier_cols = [c for c in all_cols if c.endswith("_marketingCarrier_code") or c.endswith("_operatingCarrier_code")]
    label_enc_cols = aircraft_cols + flightnum_cols + airport_cols + carrier_cols
    if "searchRoute" in all_cols:
        label_enc_cols.append("searchRoute")

    # -------- 小工具：正規化分類欄（""/"missing" -> null）--------
    def norm_expr(col: str) -> pl.Expr:
        s = pl.col(col).cast(pl.Utf8)
        return pl.when(s.str.strip_chars().str.to_lowercase().is_in(["", "missing"])).then(None).otherwise(s).alias(col)

    # ================== Test 模式：套用既有映射 ==================
    if transform_config:
        encoders = transform_config["label_encoders"]

        # carrier_pop 用 NaN 當預設
        df_result = df.select([id_col, "legs0_segments0_marketingCarrier_code", "legs1_segments0_marketingCarrier_code"])
        df_result = df_result.with_columns([
            pl.col("legs0_segments0_marketingCarrier_code").cast(pl.Utf8)
              .replace(transform_config["carrier0_pop_lookup"], default=float("nan"))
              .cast(pl.Float32).alias("carrier0_pop"),
            pl.col("legs1_segments0_marketingCarrier_code").cast(pl.Utf8)
              .replace(transform_config["carrier1_pop_lookup"], default=float("nan"))
              .cast(pl.Float32).alias("carrier1_pop"),
        ])
        df_result = df_result.with_columns([
            (pl.col("carrier0_pop") * pl.col("carrier1_pop")).alias("carrier_pop_product"),
        ])

        # label encoding：保持缺失為 NaN（不 fill_null）
        for key, enc in encoders.items():
            cols = enc["columns"]
            mapping_df = pl.DataFrame({"value": enc["values"], "rank_id": enc["codes"]})
            for col in cols:
                df_col = (
                    df.select([id_col, col])
                      .with_columns([norm_expr(col)])  # 先正規化
                      .join(mapping_df.rename({"value": col}), on=col, how="left")
                      .with_columns(pl.col("rank_id").cast(pl.Float32).alias(col))  # 保持 NaN
                      .drop("rank_id")
                )
                if col in df_result.columns:
                    df_result = df_result.drop(col)
                df_result = df_result.join(df_col, on=id_col, how="left")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, "10_df_restored_features.parquet")
            df_result.write_parquet(path)
        return df_result, transform_config

    # ================== Train 模式：計算 popularity & 映射 ==================
    # carrier_pop（以缺失為 NaN）
    carrier0_pop_df = (
        df.filter(pl.col("legs0_segments0_marketingCarrier_code").is_not_null())
          .group_by('legs0_segments0_marketingCarrier_code')
          .agg(pl.mean(selected_col).alias('carrier0_pop'))
    )
    carrier1_pop_df = (
        df.filter(pl.col("legs1_segments0_marketingCarrier_code").is_not_null())
          .group_by('legs1_segments0_marketingCarrier_code')
          .agg(pl.mean(selected_col).alias('carrier1_pop'))
    )
    carrier0_pop_dict = dict(zip(
        carrier0_pop_df['legs0_segments0_marketingCarrier_code'].to_list(),
        carrier0_pop_df['carrier0_pop'].to_list()
    ))
    carrier1_pop_dict = dict(zip(
        carrier1_pop_df['legs1_segments0_marketingCarrier_code'].to_list(),
        carrier1_pop_df['carrier1_pop'].to_list()
    ))

    df = df.with_columns([
        pl.col("legs0_segments0_marketingCarrier_code").cast(pl.Utf8)
          .replace(carrier0_pop_dict, default=float("nan"))
          .cast(pl.Float32).alias("carrier0_pop"),
        pl.col("legs1_segments0_marketingCarrier_code").cast(pl.Utf8)
          .replace(carrier1_pop_dict, default=float("nan"))
          .cast(pl.Float32).alias("carrier1_pop"),
    ])
    df = df.with_columns([
        (pl.col("carrier0_pop") * pl.col("carrier1_pop")).alias("carrier_pop_product"),
    ])

    # 編碼容器
    label_encoders: Dict[str, Dict] = {}
    feature_cols = [id_col, "carrier0_pop", "carrier1_pop", "carrier_pop_product"]
    df_encoded = df.select(feature_cols)

    # ---- 共用欄位的編碼（melt 後去重、丟掉 null，保持 NaN）----
    def encode_shared(cols: list, encoder_name: str):
        df_norm = df.select([id_col] + cols).with_columns([norm_expr(c) for c in cols])
        mapping_df = (
            df_norm.select(cols).melt()
                  .select(pl.col("value"))
                  .drop_nulls()
                  .unique()
                  .sort("value")
                  .with_columns(pl.arange(0, pl.len()).cast(pl.Int32).alias("rank_id"))
        )
        label_encoders[encoder_name] = {
            "columns": cols,
            "values": mapping_df["value"].to_list(),
            "codes":  mapping_df["rank_id"].to_list()
        }
        nonlocal df_encoded
        for col in cols:
            encoded = (
                df_norm.select([id_col, col])
                       .join(mapping_df.rename({"value": col}), on=col, how="left")
                       .with_columns(pl.col("rank_id").cast(pl.Float32).alias(col))  # NaN preserved
                       .drop("rank_id")
            )
            df_encoded = df_encoded.join(encoded, on=id_col, how="left")

    # ---- 單欄位的編碼（同樣丟掉 null）----
    def encode_individual(col: str):
        df_norm = df.select([id_col, col]).with_columns([norm_expr(col)])
        mapping_df = (
            df_norm.select(pl.col(col))
                   .drop_nulls()
                   .unique()
                   .sort(col)
                   .with_columns(pl.arange(0, pl.len()).cast(pl.Int32).alias("rank_id"))
        )
        label_encoders[col] = {
            "columns": [col],
            "values": mapping_df[col].to_list(),
            "codes":  mapping_df["rank_id"].to_list()
        }
        encoded = (
            df_norm.join(mapping_df, on=col, how="left")
                   .with_columns(pl.col("rank_id").cast(pl.Float32).alias(col))  # NaN preserved
                   .drop("rank_id")
        )
        nonlocal df_encoded
        df_encoded = df_encoded.join(encoded, on=id_col, how="left")

    # 依類別群組編碼
    if airport_cols:  encode_shared(airport_cols, "airport_cols")
    if carrier_cols:  encode_shared(carrier_cols, "carrier_cols")
    if aircraft_cols: encode_shared(aircraft_cols, "aircraft_cols")
    if flightnum_cols: encode_shared(flightnum_cols, "flightnum_cols")
    if "searchRoute" in all_cols: encode_individual("searchRoute")

    # 組合 config
    config = {
        "label_encoders": label_encoders,
        "carrier0_pop_lookup": carrier0_pop_dict,
        "carrier1_pop_lookup": carrier1_pop_dict
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if transform_config is None:
            config_path = os.path.join(output_dir, "transform_config_rank.pkl")
            with open(config_path, "wb") as f:
                pickle.dump(config, f)
            print(f"✅ 已儲存 transform_config: {config_path}")
        encoded_path = os.path.join(output_dir, "10_df_encoded_features.parquet")
        df_encoded.write_parquet(encoded_path)
        print(f"✅ 已儲存編碼後特徵: {encoded_path}")

    print("✅ Label Encoding + Carrier Popularity 完成（缺失維持 NaN）")
    return df_encoded, config





import polars as pl

def clean_fill_and_cast_columns(
    df: pl.DataFrame,
    test: bool = False
) -> pl.DataFrame:
    """
    清理資料：
    - 字串欄：空字串 -> null -> "missing"
    - 浮點欄：null -> NaN
    - 整數欄：若該欄有 null，轉成 Float64 並將 null -> NaN；否則保持整數不動
    - 布林欄：轉 0/1 (Int8)，保留原本的 null（會是 Int8 的 null）
    - 若 test=True：把列舉的 duration 欄位轉成字串並填 "missing"
    """
    # 類型歸類
    str_cols   = [c for c in df.columns if df[c].dtype in (pl.Utf8, pl.String)]
    float_cols = [c for c in df.columns if df[c].dtype.is_float()]
    int_cols   = [c for c in df.columns if df[c].dtype.is_integer()]
    bool_cols  = [c for c in df.columns if df[c].dtype == pl.Boolean]

    # 1) 字串：空字串 -> null -> "missing"
    if str_cols:
        df = df.with_columns([
            pl.when(pl.col(c).str.strip_chars() == "")
              .then(None)
              .otherwise(pl.col(c))
              .alias(c)
            for c in str_cols
        ])
        df = df.with_columns([pl.col(c).fill_null("missing") for c in str_cols])

    # 2) 數值：null -> NaN
    #    先找出有缺值的整數欄，只轉這些欄位為 Float64；沒缺值的整數欄保持整數不動
    if float_cols or int_cols:
        # 每欄位的 null 計數
        nc_df = df.null_count()
        null_counts = {c: nc_df[c][0] for c in df.columns}

        ints_with_nulls = [c for c in int_cols if null_counts.get(c, 0) > 0]

        num_exprs = []
        # 浮點欄：直接填 NaN（保留原 dtype）
        for c in float_cols:
            num_exprs.append(pl.col(c).fill_null(float("nan")).alias(c))
        # 有缺值的整數欄：轉成 Float64 並填 NaN
        for c in ints_with_nulls:
            num_exprs.append(pl.col(c).cast(pl.Float64).fill_null(float("nan")).alias(c))

        if num_exprs:
            df = df.with_columns(num_exprs)

    # 3) 布林：轉 0/1（null 照舊）
    if bool_cols:
        df = df.with_columns([pl.col(c).cast(pl.Int8).alias(c) for c in bool_cols])

    # 4) test=True 時，把 duration 欄位轉字串並填 "missing"
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
            "legs1_segments3_duration",
        ]
        exist = [c for c in duration_cols if c in df.columns]
        if exist:
            df = df.with_columns([
                pl.col(c).cast(pl.Utf8).fill_null("missing").alias(c) for c in exist
            ])
            print(f"✅ test=True: 已將 {len(exist)} 個 duration 欄位轉為 str 並填 'missing'")
    # 找字串欄
    str_cols = [c for c in df.columns if df[c].dtype in (pl.Utf8, pl.String)]
    # 找數值欄
    numeric_cols = [c for c in df.columns if df[c].dtype.is_numeric()]
    # 找布林欄
    bool_cols = [c for c in df.columns if df[c].dtype == pl.Boolean]

    print(f"✅ 共找到 {len(str_cols)} 個字串欄位")
    print(f"✅ 共找到 {len(numeric_cols)} 個數值欄位")
    print(f"✅ 共找到 {len(bool_cols)} 個布林欄位")
    known_cols = set(str_cols + numeric_cols + bool_cols)
    other_cols = [c for c in df.columns if c not in known_cols]

    print(f"🔍 尚未分類的欄位共有 {len(other_cols)} 個：")
    print(other_cols)

    print("✅ 完成：字串清理、數值 null→NaN、布林 0/1（保留布林 null）")
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
import polars as pl
import re, os, pickle
from typing import Optional, Tuple, Dict, List

def enrich_flight_view_features(
    df: pl.DataFrame,
    output_dir: Optional[str] = None,
    output_filename: str = "11_flight_view_features.parquet",
    transform_config: Optional[dict] = None
) -> Tuple[pl.DataFrame, Optional[dict]]:
    # ---------- 工具：找出每條腿實際存在的 segment 索引 ----------
    def _find_segment_indices(df: pl.DataFrame, leg: int) -> List[int]:
        # 只要有 departure/arrival 任一種就算這個 seg 存在
        pat = re.compile(rf"^legs{leg}_segments(\d+)_(?:departureFrom|arrivalTo)_airport_iata$")
        idx = set()
        for c in df.columns:
            m = pat.match(c)
            if m:
                idx.add(int(m.group(1)))
        return sorted(idx)

    segs0 = _find_segment_indices(df, 0)
    segs1 = _find_segment_indices(df, 1)

    # 若真的沒有任一段，避免後面空 list 爆掉
    if not segs0 and not segs1:
        # 什麼都沒有就直接回傳原 df
        return df, None

    # ---------- 為每個實際存在的 segment 生成 key 欄位 ----------
    def make_leg_segment_key_exprs(leg_prefix: str, seg_indices: List[int]) -> List[pl.Expr]:
        exprs = []
        for i in seg_indices:
            key_name = f"{leg_prefix}_segments{i}_key"
            dep = pl.col(f"{leg_prefix}_segments{i}_departureFrom_airport_iata").cast(pl.Utf8).fill_null("missing")
            arr = pl.col(f"{leg_prefix}_segments{i}_arrivalTo_airport_iata").cast(pl.Utf8).fill_null("missing")
            exprs.append((dep + "-" + arr).alias(key_name))
        return exprs

    df = df.with_columns(
        make_leg_segment_key_exprs("legs0", segs0) +
        make_leg_segment_key_exprs("legs1", segs1)
    )

    seg_key_cols_0 = [f"legs0_segments{i}_key" for i in segs0]
    seg_key_cols_1 = [f"legs1_segments{i}_key" for i in segs1]
    all_seg_key_cols = seg_key_cols_0 + seg_key_cols_1

    # ---------- segment pair 熱度（出現次數） ----------
    if transform_config is None:
        if all_seg_key_cols:
            segment_counts = (
                df.melt(id_vars=[], value_vars=all_seg_key_cols)
                  .filter(pl.col("value") != "missing-missing")
                  .group_by("value")
                  .agg(pl.count().alias("segment_view_count"))
            )
            segment_counts_dict = segment_counts.to_dict(as_series=False)
        else:
            segment_counts = pl.DataFrame({"value": [], "segment_view_count": []})
            segment_counts_dict = segment_counts.to_dict(as_series=False)
    else:
        segment_counts = pl.DataFrame(transform_config["segment_counts"])

    # 把各 segment 的 view_count 合併回 df
    for seg_col in all_seg_key_cols:
        df = (
            df.join(segment_counts, left_on=seg_col, right_on="value", how="left")
              .with_columns(pl.col("segment_view_count").fill_null(0).alias(f"{seg_col}_view_count"))
              .drop("segment_view_count")
        )

    # ---------- leg 層級的 key（把該腿實際存在的 seg key 用 '|' 串起來） ----------
    def make_leg_full_key_expr(leg_prefix: str, seg_indices: List[int], out_col: str) -> pl.Expr:
        cols = [f"{leg_prefix}_segments{i}_key" for i in seg_indices]
        if cols:
            return pl.concat_str([pl.col(c) for c in cols], separator="|").alias(out_col)
        else:
            # 沒有 segment 的話就給空字串，避免 concat_str 空清單報錯
            return pl.lit("", dtype=pl.Utf8).alias(out_col)

    # 先建立 legs0_key / legs1_key
    df = df.with_columns([
        make_leg_full_key_expr("legs0", segs0, "legs0_key"),
        make_leg_full_key_expr("legs1", segs1, "legs1_key"),
    ])

    # 用已存在的欄位直接組 all_key（不需要任何 *_tmp 欄位）
    df = df.with_columns([
        (pl.col("legs0_key") + pl.lit("||") + pl.col("legs1_key")).alias("all_key")
    ])


    # ---------- 計算 key 出現次數（train）或從 config 載入（test） ----------
    if transform_config is None:
        leg0_counts = df.group_by("legs0_key").agg(pl.count().alias("leg0_flight_view_count"))
        leg1_counts = df.group_by("legs1_key").agg(pl.count().alias("leg1_flight_view_count"))
        all_counts  = df.group_by("all_key").agg(pl.count().alias("all_flight_view_count"))
        leg0_counts_dict = leg0_counts.to_dict(as_series=False)
        leg1_counts_dict = leg1_counts.to_dict(as_series=False)
        all_counts_dict  = all_counts.to_dict(as_series=False)
    else:
        leg0_counts = pl.DataFrame(transform_config["leg0_counts"])
        leg1_counts = pl.DataFrame(transform_config["leg1_counts"])
        all_counts  = pl.DataFrame(transform_config["all_counts"])

    df = df.join(leg0_counts, on="legs0_key", how="left")
    df = df.join(leg1_counts, on="legs1_key", how="left")
    df = df.join(all_counts,  on="all_key",  how="left")

    # ---------- ranker 內的 max/mean 與正規化 ----------
    ranker_stats_max = df.group_by("ranker_id").agg([
        pl.max("leg0_flight_view_count").alias("leg0_view_max"),
        pl.max("leg1_flight_view_count").alias("leg1_view_max"),
        pl.max("all_flight_view_count").alias("all_view_max"),
    ])
    df = df.join(ranker_stats_max, on="ranker_id", how="left")

    df = df.with_columns([
        (pl.col("leg0_flight_view_count") / (pl.col("leg0_view_max") + 1e-5)).alias("leg0_view_norm"),
        (pl.col("leg1_flight_view_count") / (pl.col("leg1_view_max") + 1e-5)).alias("leg1_view_norm"),
        (pl.col("all_flight_view_count")  / (pl.col("all_view_max")  + 1e-5)).alias("all_view_norm"),
        pl.len().over("ranker_id").alias("group_size"),
    ])
    df = df.with_columns(pl.col("group_size").log1p().alias("group_size_log"))

    ranker_stats_mean = df.group_by("ranker_id").agg([
        pl.mean("leg0_flight_view_count").alias("leg0_view_mean"),
        pl.mean("leg1_flight_view_count").alias("leg1_view_mean"),
        pl.mean("all_flight_view_count").alias("all_view_mean"),
    ])
    df = df.join(ranker_stats_mean, on="ranker_id", how="left")

    df = df.with_columns([
        (pl.col("leg0_flight_view_count") - pl.col("leg0_view_mean")).alias("leg0_view_diff_mean"),
        (pl.col("leg1_flight_view_count") - pl.col("leg1_view_mean")).alias("leg1_view_diff_mean"),
        (pl.col("all_flight_view_count")  - pl.col("all_view_mean")).alias("all_view_diff_mean"),
    ])

    # ---------- rank（動態收集所有要排名的欄位） ----------
    rank_features = [
        "leg0_flight_view_count",
        "leg1_flight_view_count",
        "all_flight_view_count",
    ]
    # 加上各 segment 的 view_count 欄
    rank_features += [f"{c}_view_count" for c in all_seg_key_cols]

    rank_exprs = [
        pl.col(c).rank(method="dense").over("ranker_id").alias(f"{c}_rank")
        for c in rank_features if c in df.columns
    ]
    df = df.with_columns(rank_exprs)

    # ---------- 輸出用的 transform_config（train 時才有） ----------
    output_config = None
    if transform_config is None:
        output_config = {
            "segment_counts": segment_counts_dict,
            "leg0_counts": leg0_counts_dict,
            "leg1_counts": leg1_counts_dict,
            "all_counts": all_counts_dict
        }

    # ---------- 要 drop 的欄位（動態） ----------
    airport_cols_used = [
        f"{leg}_segments{i}_{side}_airport_iata"
        for leg, segs in (("legs0", segs0), ("legs1", segs1))
        for i in segs
        for side in ("departureFrom", "arrivalTo")
    ]
    seg_key_cols_all = seg_key_cols_0 + seg_key_cols_1
    columns_to_drop = (
        ["leg0_view_max", "leg1_view_max", "all_view_max",
         "leg0_view_mean", "leg1_view_mean", "all_view_mean",
         "legs0_key", "legs1_key", "all_key", "searchRoute", "ranker_id"]
        + seg_key_cols_all
        + airport_cols_used
    )
    df_save = df.drop([c for c in columns_to_drop if c in df.columns])

    # 只為了回傳時把 Id 放在最前；保留所有欄位
    keep_cols = ["Id"] + [c for c in df.columns if c != "Id"]
    df = df.select(keep_cols)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df_save.write_parquet(os.path.join(output_dir, output_filename))
        print(f"✅ 已儲存 flight view 特徵: {os.path.join(output_dir, output_filename)}")
        if output_config is not None:
            config_path = os.path.join(output_dir, "transform_flight_view_key_config.pkl")
            with open(config_path, "wb") as f:
                pickle.dump(output_config, f)
            print(f"✅ 已儲存 transform_config: {config_path}")

    return df, output_config


def make_companyID_into_features(
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
        df = df.with_columns([
            pl.col(target_col).cast(pl.Int8)
        ])
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
        df_path = os.path.join(output_dir, "12_companyID_into_features.parquet")
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



import polars as pl
import re
from typing import List, Optional

def replace_group_minmax_for_rankish_features(
    df: pl.DataFrame,
    group_col: str = "ranker_id",
    id_col: str = "Id",
    *,
    keyword: str = "rank",                 # 只要欄名包含此字串（不分大小寫）
    include_cols: Optional[List[str]] = None,  # 指定白名單（優先於 keyword）
    exclude_cols: Optional[List[str]] = None,  # 額外排除
    constant_fill: float = 0.5,            # 當組內 max==min（常數組）時的回填值
) -> pl.DataFrame:
    # 基本檢查
    for c in (group_col, id_col):
        if c not in df.columns:
            raise ValueError(f"缺少必要欄位: {c}")

    exclude = set(exclude_cols or [])
    exclude.update({group_col, id_col})

    # 1) 找出候選欄位：包含 'rank'（不分大小寫）或白名單
    if include_cols:
        candidates = [c for c in include_cols if c in df.columns and c not in exclude]
    else:
        pat = re.compile(re.escape(keyword), re.IGNORECASE)
        candidates = [c for c in df.columns if c not in exclude and pat.search(c)]

    if not candidates:
        print("ℹ️ 找不到需要正規化的 rank 欄位；直接回傳原 df。")
        return df

    # 2) 僅保留數值欄
    num_dtypes = {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64
    }
    rank_cols = [c for c in candidates if df.schema.get(c) in num_dtypes]
    if not rank_cols:
        print("ℹ️ 找到含 'rank' 欄位但皆非數值型；直接回傳原 df。")
        return df

    # 3) 用 select + window 算 group-wise Min/Max，再 join 回原 df 覆寫
    need_cols = [id_col, group_col] + rank_cols
    base = df.select([c for c in need_cols if c in df.columns])

    exprs = []
    for c in rank_cols:
        x = pl.col(c).cast(pl.Float32)
        gmin = x.min().over(group_col)
        gmax = x.max().over(group_col)
        den  = gmax - gmin
        # 規則：
        # - 若組內皆為 null → 保留 null（den 會是 null）
        # - 若 max==min → 回填 constant_fill
        # - 其他 → (x - min)/(max - min)
        mm = (
            pl.when(den.is_null())                      # 整組皆空
              .then(None)
            .when(den == 0)                             # 整組常數
              .then(constant_fill)
            .otherwise((x - gmin) / den)                # 一般情況
        )
        exprs.append(mm.alias(c))                       # 直接用原欄名，等會覆寫

    mm_df = base.with_columns(exprs).select([id_col] + rank_cols)

    # 丟掉原 rank 欄位，併回正規化後的同名欄位 → 完成覆寫
    out = df.drop(rank_cols).join(mm_df, on=id_col, how="left")
    return out

