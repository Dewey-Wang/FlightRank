import polars as pl

CONSTANT_COLS = [
    'bySelf','companyID','currencyCode','hasAssistant','isGlobal',
    'isSegDiscountAsExactValue','isSegDiscountVariant','isVip','nationality',
    'option_legs_count','pricingInfo_count','pricingInfo_passengerCount',
    'pricingInfo_passengerType','pricingInfo_selfPaid','pricing_present',
    'profileId','requestDate','requestDepartureDate','request_time',
    'root_$id','searchRoute','searchType','sex','yearOfBirth','frequentFlyer'
]

def enforce_ranker_constants(
    df: pl.DataFrame,
    group_col: str = "ranker_id",
    cols: list[str] = None,
    missing_markers: tuple[str, ...] = ("missing", "")
) -> pl.DataFrame:
    """
    將指定欄位在每個 ranker_id 群內統一成「第一個有效值」。
    有效值定義：
      - 數值: 非 null 且(若為浮點) 非 NaN
      - 布林: 非 null
      - 其他(字串/日期等): 非 null 且不等於 missing_markers（如 "missing"、空字串）
    若群內全無有效值，保留原值不動。
    """
    if cols is None:
        cols = CONSTANT_COLS

    if group_col not in df.columns:
        raise ValueError(f"'{group_col}' 不在資料中")

    # 先為每個欄位建一個「有效值候選」欄位，之後 groupby 取第一個
    cand_exprs = []
    for c in cols:
        if c not in df.columns:
            continue
        dt = df.schema.get(c)
        if dt is None:
            continue

        if dt.is_float():
            valid = pl.col(c).is_not_null() & (~pl.col(c).is_nan())
            cand = pl.when(valid).then(pl.col(c)).otherwise(None).alias(f"__cand__{c}")
        elif dt.is_numeric() or dt == pl.Boolean:
            valid = pl.col(c).is_not_null()
            cand = pl.when(valid).then(pl.col(c)).otherwise(None).alias(f"__cand__{c}")
        else:
            s = pl.col(c).cast(pl.Utf8)
            valid = s.is_not_null() & (~s.is_in(list(missing_markers)))
            cand = pl.when(valid).then(s).otherwise(None).alias(f"__cand__{c}")

        cand_exprs.append(cand)

    if not cand_exprs:
        return df

    df2 = df.with_columns(cand_exprs)

    # 每個群取第一個有效值作為 canonical
    agg_exprs = [pl.col(f"__cand__{c}").first().alias(f"__canon__{c}")
                 for c in cols if f"__cand__{c}" in df2.columns]
    canon = (
        df2.lazy()
           .group_by(group_col)
           .agg(agg_exprs)
           .collect()
    )

    # 併回去並覆寫欄位：若 canonical 有值，一律用 canonical；否則保留原值
    out = df2.join(canon, on=group_col, how="left")
    set_exprs = []
    for c in cols:
        can = f"__canon__{c}"
        if can not in out.columns or c not in out.columns:
            continue
        # 盡量保持原型別
        target_dtype = df.schema.get(c, pl.Utf8)
        set_exprs.append(
            pl.when(pl.col(can).is_not_null())
              .then(pl.col(can).cast(target_dtype, strict=False))
              .otherwise(pl.col(c))
              .alias(c)
        )

    out = out.with_columns(set_exprs)

    # 移除暫存欄
    drop_cols = [f"__cand__{c}" for c in cols if f"__cand__{c}" in out.columns] + \
                [f"__canon__{c}" for c in cols if f"__canon__{c}" in out.columns]
    out = out.drop([c for c in drop_cols if c in out.columns])

    return out


import polars as pl

def global_constant_features(
    df: pl.DataFrame,
    group_col: str = "ranker_id",
    *,
    ignore_missing_markers: tuple[str, ...] = ("missing", ""),
    exclude_cols: tuple[str, ...] = ("Id", "selected"),
) -> tuple[list[str], pl.DataFrame]:
    """
    找出「每個 ranker_id 內皆為恆定」的共同欄位（交集）。
    恆定判定：同一群內『有效值』唯一且至少一筆。
      - 數值: 忽略 null/NaN
      - 字串: 忽略 null / "" / "missing"
      - 布林: 忽略 null
    回傳：
      - common_feats: list[str]（共同恆定欄位）
      - coverage_df: 每個 feature 覆蓋了幾個群、總群數為多少（方便檢視）
    """
    if group_col not in df.columns:
        raise ValueError(f"'{group_col}' 不在資料中")

    cols = [c for c in df.columns if c != group_col and c not in exclude_cols]
    n_groups = df.select(pl.col(group_col).n_unique()).item()

    per_col_frames = []

    for c in cols:
        dt = df.schema.get(c)
        if dt is None:
            continue

        # 建「有效值」欄 __val__：無效→None
        if dt.is_numeric():
            valid_num = pl.col(c).is_not_null()
            if dt in (pl.Float32, pl.Float64):
                valid_num = valid_num & (~pl.col(c).is_nan())
            val = pl.when(valid_num).then(pl.col(c)).otherwise(None).alias("__val__")
        elif dt == pl.Boolean:
            val = pl.when(pl.col(c).is_not_null()).then(pl.col(c)).otherwise(None).alias("__val__")
        else:
            s = pl.col(c).cast(pl.Utf8)
            val = (
                pl.when(s.is_not_null() & (~s.is_in(list(ignore_missing_markers))))
                  .then(s).otherwise(None).alias("__val__")
            )

        # 這個欄位在哪些群是「恆定」：n_unique(__val__) == 1 且有至少一筆有效值
        lf = (
            df.lazy()
              .select([pl.col(group_col), val])
              .group_by(group_col)
              .agg([
                  pl.col("__val__").n_unique().alias("__nuniq__"),
                  pl.col("__val__").count().alias("__nvalid__"),  # count 會忽略 None
              ])
              .filter((pl.col("__nvalid__") > 0) & (pl.col("__nuniq__") == 1))
              .select([pl.col(group_col), pl.lit(c).alias("feature")])
        )
        per_col_frames.append(lf)

    if not per_col_frames:
        return [], pl.DataFrame({"feature": [], "n_groups_const": [], "total_groups": []})

    const_hits = pl.concat([lf.collect() for lf in per_col_frames], how="vertical")

    coverage_df = (
        const_hits
        .group_by("feature")
        .agg(pl.col(group_col).n_unique().alias("n_groups_const"))
        .with_columns(pl.lit(n_groups).alias("total_groups"))
        .sort(["n_groups_const", "feature"], descending=[True, False])
    )

    # 共同恆定欄位 = 在所有群都恆定
    common_feats = (
        coverage_df
        .filter(pl.col("n_groups_const") == pl.col("total_groups"))
        .select("feature")
        .to_series()
        .to_list()
    )

    return common_feats, coverage_df
