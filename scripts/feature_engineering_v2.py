#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Feature Engineering (v2) for Credit Risk
-------------------------------------------------
读取原始 train/test，产出 data/processed_v2 下的处理后缓存（CSV + meta.json），
与现有训练脚本 try_load_processed_cache() 完全兼容。

用法示例：
  python feature_engineering_v2.py \
    --train_path data/train.csv \
    --test_path data/testA.csv \
    --target isDefault \
    --id_col id \
    --cache_dir data/processed_v2 \
    --te_folds 5 --te_m 50 --woe_bins 10 --random_state 2025 \
    --high_card_threshold 200 --fast_te 1
"""

# ----------------------------
# CPU 线程控制（32核）
# ----------------------------
import os
os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
os.environ.setdefault("MKL_NUM_THREADS", "32")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "32")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "32")

# ----------------------------
# imports & warnings
# ----------------------------
import re
import json
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# 静默 pandas groupby 的未来警告 & 其它常见噪音
warnings.filterwarnings("ignore", category=FutureWarning, module=r"pandas\.core\.groupby")
warnings.filterwarnings("ignore", category=UserWarning, message=r"Boolean Series.*index.*")

try:
    from tqdm.auto import tqdm
    TQDM_KW = dict(leave=False, dynamic_ncols=True)
except Exception:
    def tqdm(x, **k): return x
    TQDM_KW = {}

# ----------------------------
# 小工具
# ----------------------------
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def reduce_mem_usage(df: pd.DataFrame, verbose=True, desc="Downcast"):
    start = df.memory_usage().sum() / 1024**2
    for col in tqdm(list(df.columns), desc=desc, **TQDM_KW):
        col_type = df[col].dtype
        # 整数列：若有缺失值则统一转 float32
        if str(col_type).startswith(("int", "uint")):
            if df[col].isna().any():
                df[col] = df[col].astype("float32")
                continue
            c_min, c_max = df[col].min(), df[col].max()
            if c_min >= 0:
                if c_max <= np.iinfo(np.uint8).max:    df[col] = df[col].astype(np.uint8)
                elif c_max <= np.iinfo(np.uint16).max: df[col] = df[col].astype(np.uint16)
                elif c_max <= np.iinfo(np.uint32).max: df[col] = df[col].astype(np.uint32)
                else:                                  df[col] = df[col].astype(np.uint64)
            else:
                if np.iinfo(np.int8).min  <= c_min <= np.iinfo(np.int8).max:    df[col] = df[col].astype(np.int8)
                elif np.iinfo(np.int16).min <= c_min <= np.iinfo(np.int16).max: df[col] = df[col].astype(np.int16)
                elif np.iinfo(np.int32).min <= c_min <= np.iinfo(np.int32).max: df[col] = df[col].astype(np.int32)
                else:                                                           df[col] = df[col].astype(np.int64)
        elif str(col_type).startswith("float"):
            df[col] = df[col].astype(np.float32)
    end = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"[reduce_mem_usage] {start:.2f} -> {end:.2f} MB")
    return df

def parse_employment_length(x):
    if pd.isna(x): return np.nan
    s = str(x).lower().strip()
    if "10" in s: return 10.0
    if "<" in s:  return 0.0
    m = re.findall(r"\d+", s)
    return float(m[0]) if m else np.nan

def normalize_employment_length_token(x):
    if pd.isna(x): return "na"
    s = str(x).lower().strip()
    if "10" in s: return "10+"
    if "<" in s:  return "<1"
    m = re.findall(r"\d+", s)
    return m[0] if m else "na"

def parse_term_general(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float)): return float(x)*12 if x <= 10 else float(x)
    s = str(x).lower()
    m = re.findall(r"\d+", s)
    if not m: return np.nan
    n = float(m[0])
    return n*12 if "year" in s else n

def parse_month(s):
    if pd.isna(s): return pd.NaT
    s = str(s).strip()
    for fmt in ("%Y/%m/%d","%Y-%m-%d","%Y-%m","%Y/%m","%b-%y","%b-%Y"):
        try: return pd.to_datetime(s, format=fmt)
        except: continue
    return pd.to_datetime(s, errors="coerce")

def safe_cut(series, bins, labels=None):
    try:
        return pd.cut(series, bins=bins, labels=labels, include_lowest=True)
    except Exception:
        return pd.Series(pd.Categorical([np.nan]*len(series)))

def freq_encode(series: pd.Series):
    vc = series.astype(str).value_counts(dropna=False)
    mapping = vc / vc.sum()
    return series.astype(str).map(mapping).astype("float32")

def to_3digit_postcode_str(series: pd.Series):
    s = series.astype(str).str.split(".").str[0]
    s = s.where(~s.isin(["nan","None","none","NaN"]), other="")
    return s.str.zfill(3)

def winsorize_series(s: pd.Series, lower=0.005, upper=0.995):
    try:
        lo, hi = s.quantile(lower), s.quantile(upper)
        return s.clip(lo, hi)
    except Exception:
        return s

def add_missing_indicator(df: pd.DataFrame, cols: list):
    for c in cols:
        if c in df.columns and df[c].isna().any():
            df[f"{c}_isna"] = df[c].isna().astype("int8")
    return df

def add_n_stats(df, n_cols):
    n_df = df[n_cols].apply(pd.to_numeric, errors="coerce")
    df["n_sum"]      = n_df.sum(axis=1).astype("float32")
    df["n_mean"]     = n_df.mean(axis=1).astype("float32")
    df["n_max"]      = n_df.max(axis=1).astype("float32")
    df["n_std"]      = n_df.std(axis=1).astype("float32")
    df["n_nonzero"]  = (n_df > 0).sum(axis=1).astype("float32")
    return df

# ----------------------------
# KFold Target Encoding：安全 & 快速版
# ----------------------------
from sklearn.model_selection import KFold

def kfold_target_encoding_fast(full, col, target, n_splits=5, m=50.0, seed=2025):
    """
    快速 & 稳定的 KFold Target Encoding：
    - full[col] 先 astype('category')
    - 使用 .cat.codes + np.bincount 统计
    - 避免 Pandas Categorical 写入问题
    """
    assert "__is_train__" in full.columns and target in full.columns
    is_tr = (full["__is_train__"] == 1).values
    idx_tr = np.where(is_tr)[0]
    idx_te = np.where(~is_tr)[0]

    y = full.loc[idx_tr, target].astype(float).values
    prior = float(np.nanmean(y)) if len(y) else 0.5

    te_name = f"te_{col}"
    full[te_name] = np.nan
    full[te_name] = full[te_name].astype("float32")

    full[col] = full[col].astype("category")
    codes = full[col].cat.codes.values  # -1 表示缺失
    codes_tr = codes[idx_tr]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, va_idx in kf.split(idx_tr):
        train_ids = idx_tr[tr_idx]
        valid_ids = idx_tr[va_idx]

        c_tr = codes[train_ids]
        y_tr = full.loc[train_ids, target].astype(float).values

        mask = (c_tr >= 0)
        c_tr = c_tr[mask]
        y_tr = y_tr[mask]

        K = int(full[col].cat.categories.size)
        sum_y = np.bincount(c_tr, weights=y_tr, minlength=K)
        cnt   = np.bincount(c_tr, minlength=K)

        enc_vec = (sum_y + prior*m) / (cnt + m)

        c_va = codes[valid_ids]
        te_va = np.full_like(c_va, fill_value=np.nan, dtype=float)
        ok = (c_va >= 0)
        te_va[ok] = enc_vec[c_va[ok]]
        te_va[~ok] = prior
        full.loc[valid_ids, te_name] = te_va.astype("float32")

    # 测试用全量训练映射
    c_all = codes[idx_tr]
    y_all = full.loc[idx_tr, target].astype(float).values
    mask = (c_all >= 0)
    c_all = c_all[mask]; y_all = y_all[mask]
    K = int(full[col].cat.categories.size)
    sum_y = np.bincount(c_all, weights=y_all, minlength=K)
    cnt   = np.bincount(c_all, minlength=K)
    enc_vec = (sum_y + prior*m) / (cnt + m)

    c_te = codes[idx_te]
    te_te = np.full_like(c_te, fill_value=np.nan, dtype=float)
    ok = (c_te >= 0)
    te_te[ok] = enc_vec[c_te[ok]]
    te_te[~ok] = prior
    full.loc[idx_te, te_name] = te_te.astype("float32")

    return full, te_name

# ----------------------------
# WOE（等频分箱，安全映射）
# ----------------------------
def woe_encode_from_quantiles_on_full(full: pd.DataFrame,
                                      col: str,
                                      target: str,
                                      q: int = 10,
                                      eps: float = 0.5):
    """
    仅用训练部分计算分箱与 WOE，然后映射到全量。
    关键：
      - groupby(..., observed=True)
      - binned.astype('object').map(dict) → float Series，避免 Categorical 写入问题
    """
    assert "__is_train__" in full.columns and target in full.columns
    tr_mask = (full["__is_train__"] == 1)

    if col not in full.columns:
        full[f"woe_{col}"] = 0.0
        return full, f"woe_{col}"

    try:
        tr_vals = full.loc[tr_mask, col]
        if tr_vals.nunique(dropna=True) < 2:
            full[f"woe_{col}"] = 0.0
            return full, f"woe_{col}"

        bins_tr = pd.qcut(tr_vals, q=q, duplicates="drop")
        edges = sorted({b.left for b in bins_tr.cat.categories} | {b.right for b in bins_tr.cat.categories})
        if len(edges) < 3:
            full[f"woe_{col}"] = 0.0
            return full, f"woe_{col}"
    except Exception:
        full[f"woe_{col}"] = 0.0
        return full, f"woe_{col}"

    binned = pd.cut(full[col], bins=edges, include_lowest=True)

    grp = full.loc[tr_mask].groupby(binned[tr_mask], observed=True)[target].agg(["sum", "count"])
    good = grp["count"] - grp["sum"]
    bad  = grp["sum"]
    good_total = good.sum()
    bad_total  = bad.sum()

    woe = np.log(((good + eps) / (good_total + eps)) / ((bad + eps) / (bad_total + eps)))
    woe = woe.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # 安全映射（float）
    wmap = woe.to_dict()
    vals = binned.astype("object").map(wmap)
    full[f"woe_{col}"] = vals.fillna(0.0).astype("float32")
    return full, f"woe_{col}"

# ----------------------------
# 特征工程 v2
# ----------------------------
def build_features_v2(train: pd.DataFrame, test: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    train = train.copy()
    test  = test.copy()
    for df in (train, test):
        if "__is_train__" in df.columns:
            df.drop(columns=["__is_train__"], inplace=True, errors="ignore")

    train["__is_train__"] = 1
    test["__is_train__"]  = 0
    full = pd.concat([train, test], axis=0, ignore_index=True)
    full = reduce_mem_usage(full, desc="Downcast (before FE)")

    # 1) 原有解析/枚举
    if "term" in full.columns:
        full["term_mon"] = full["term"].apply(parse_term_general).astype("float32")
        full["term_bin"] = safe_cut(full["term_mon"], [-1,36,60,np.inf], ["<=36","37-60",">60"]).astype("category")

    if "employmentLength" in full.columns:
        full["emp_len_year"] = full["employmentLength"].apply(parse_employment_length).astype("float32")
        full["employmentLength_cat"] = full["employmentLength"].apply(normalize_employment_length_token).astype("category")

    if "issueDate" in full.columns:
        full["issueDate_cat"] = full["issueDate"].astype(str).astype("category")
        full["issueDate_dt"]  = full["issueDate"].apply(parse_month)
        full["issue_year"]    = full["issueDate_dt"].dt.year.astype("float32")
        full["issue_month"]   = full["issueDate_dt"].dt.month.astype("float32")
    if "earliesCreditLine" in full.columns:
        full["earliesCreditLine_cat"] = full["earliesCreditLine"].astype(str).astype("category")
        full["earliest_dt"] = full["earliesCreditLine"].apply(parse_month)
    if {"issueDate_dt","earliest_dt"}.issubset(full.columns):
        diff_days = (full["issueDate_dt"] - full["earliest_dt"]).dt.days
        full["credit_hist_mon"] = (diff_days/30.0).astype("float32")

    if {"ficoRangeLow","ficoRangeHigh"}.issubset(full.columns):
        full["fico_mean"]  = ((full["ficoRangeLow"] + full["ficoRangeHigh"]) / 2.0).astype("float32")
        full["fico_range"] = (full["ficoRangeHigh"] - full["ficoRangeLow"]).astype("float32")

    if {"loanAmnt","annualIncome"}.issubset(full.columns):
        full["loan_income_ratio"] = (full["loanAmnt"] / (full["annualIncome"] + 1)).astype("float32")
        if "installment" in full.columns:
            full["install_to_income"] = (full["installment"] / (full["annualIncome"] + 1)).astype("float32")

    for col in ["employmentTitle","title"]:
        if col in full.columns:
            as_str = full[col].astype(str)
            full[f"{col}_cat"]  = as_str.astype("category")
            full[f"{col}_freq"] = freq_encode(as_str)

    if "postCode" in full.columns:
        pc3 = to_3digit_postcode_str(full["postCode"])
        full["postCode_cat"]  = pc3.astype("category")
        full["postCode_freq"] = freq_encode(pc3)

    low_card_int_as_cat = ["grade","subGrade","homeOwnership","verificationStatus",
                           "purpose","regionCode","initialListStatus","applicationType","policyCode"]
    for c in [x for x in low_card_int_as_cat if x in full.columns]:
        if str(full[c].dtype) != "category":
            full[c] = full[c].astype("category")

    # 2) 比率/交互
    if {"interestRate","term_mon"}.issubset(full.columns):
        full["ir_x_term"] = (full["interestRate"] * full["term_mon"]).astype("float32")
    if {"installment","loanAmnt"}.issubset(full.columns):
        full["inst_to_loan"] = (full["installment"] / (full["loanAmnt"] + 1)).astype("float32")
    if {"revolBal","totalAcc"}.issubset(full.columns):
        full["revol_per_acc"] = (full["revolBal"] / (full["totalAcc"] + 1)).astype("float32")

    # 3) 长尾稳健化
    for c in ["annualIncome","revolBal","loanAmnt","installment"]:
        if c in full.columns:
            full[f"log1p_{c}"] = np.log1p(full[c].clip(lower=0)).astype("float32")
            full[f"win_{c}"]   = winsorize_series(full[c]).astype("float32")

    # 4) 缺失指示
    miss_cols = [c for c in ["revolUtil","dti","emp_len_year","fico_mean","credit_hist_mon"] if c in full.columns]
    full = add_missing_indicator(full, miss_cols)

    # 5) n* 统计
    n_cols = [f"n{i}" for i in range(15) if f"n{i}" in full.columns]
    if n_cols:
        full = add_n_stats(full, n_cols)

    # 6) KFold Target Encoding（泄露安全）
    tgt = cfg.get("target", "isDefault")
    te_targets = []
    if tgt in full.columns:
        te_cands = [c for c in [
            "employmentTitle_cat","title_cat","postCode_cat",
            "subGrade","issueDate_cat","earliesCreditLine_cat"
        ] if c in full.columns]
        for c in tqdm(te_cands, desc="TE", **TQDM_KW):
            full, te_name = kfold_target_encoding_fast(
                full, c, tgt,
                n_splits=int(cfg.get("te_folds", 5)),
                m=float(cfg.get("te_m", 50.0)),
                seed=int(cfg.get("random_state", 2025))
            )
            te_targets.append(te_name)

    # 7) 数值分箱 + WOE
    woe_targets = []
    woe_cands = [c for c in ["fico_mean","credit_hist_mon","interestRate","dti","revolUtil"] if c in full.columns]
    for c in tqdm(woe_cands, desc="WOE", **TQDM_KW):
        full, wname = woe_encode_from_quantiles_on_full(
            full, col=c, target=tgt, q=int(cfg.get("woe_bins", 10)), eps=0.5
        )
        woe_targets.append(wname)

    # 8) 清理/填充
    num_cols_full = full.select_dtypes(include=["number"]).columns
    full[num_cols_full] = full[num_cols_full].replace([np.inf,-np.inf], np.nan)
    full[num_cols_full] = full[num_cols_full].fillna(full[num_cols_full].median())

    for col in ["issueDate_dt","earliest_dt"]:
        if col in full.columns:
            full.drop(columns=[col], inplace=True)

    full = reduce_mem_usage(full, desc="Downcast (after FE)")

    # 可选记录
    full.attrs["te_features"]  = te_targets
    full.attrs["woe_features"] = woe_targets
    return full

# ----------------------------
# 高基数类别降级（保留 *_freq，原列转 str）
# ----------------------------
def downgrade_high_card_categories(full: pd.DataFrame, threshold: int):
    cat_cols_now = [c for c in full.columns if str(full[c].dtype) == "category"]
    card = {c: full[c].nunique(dropna=False) for c in cat_cols_now}
    high_card_cols = [c for c, v in card.items() if v > threshold]
    print(f"[speed] high-card categories: {high_card_cols}")
    for c in high_card_cols:
        freq_col = f"{c}_freq"
        if freq_col not in full.columns:
            full[freq_col] = freq_encode(full[c].astype(str))
        full[c] = full[c].astype(str)  # 避免训练时被当作 category
    return full

# ----------------------------
# 存缓存（CSV + meta.json）—与现有训练脚本兼容
# ----------------------------
def save_processed_cache_csv(cache_dir: str,
                             train_fe: pd.DataFrame,
                             test_fe: pd.DataFrame,
                             features: list,
                             cat_features: list,
                             id_col: str,
                             target: str):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_csv = cache_dir / "train_fe.csv"
    test_csv  = cache_dir / "test_fe.csv"
    meta_js   = cache_dir / "meta.json"

    train_fe.to_csv(train_csv, index=False)
    test_fe.to_csv(test_csv, index=False)

    try:
        import lightgbm as lgb
        lgb_ver = lgb.__version__
    except Exception:
        lgb_ver = "N/A"

    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "id_col": id_col,
        "target": target,
        "features": features,
        "cat_features": cat_features,
        "n_train": int(len(train_fe)),
        "n_test": int(len(test_fe)),
        "pandas": pd.__version__,
        "lightgbm": lgb_ver,
        "format": "csv"
    }
    with open(meta_js, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[cache] saved (csv) -> {train_csv}, {test_csv}, {meta_js}")

# ----------------------------
# 主流程
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", type=str, required=True)
    ap.add_argument("--test_path",  type=str, required=True)
    ap.add_argument("--target",     type=str, default="isDefault")
    ap.add_argument("--id_col",     type=str, default="id")
    ap.add_argument("--cache_dir",  type=str, default="data/processed_v2")

    ap.add_argument("--te_folds",   type=int, default=5)
    ap.add_argument("--te_m",       type=float, default=50.0)
    ap.add_argument("--woe_bins",   type=int, default=10)
    ap.add_argument("--random_state", type=int, default=2025)
    ap.add_argument("--high_card_threshold", type=int, default=200)
    ap.add_argument("--fast_te",    type=int, default=1)   # 已默认开启极速 TE

    args = ap.parse_args()
    cfg = vars(args)

    # 读取原始数据
    train = pd.read_csv(args.train_path, low_memory=False)
    test  = pd.read_csv(args.test_path,  low_memory=False)
    print(f"[load] train={train.shape}, test={test.shape}")

    # 构建增强特征
    full = build_features_v2(train, test, cfg)

    # 降级超高基数类别
    full = downgrade_high_card_categories(full, args.high_card_threshold)

    # 切回 train/test
    train_fe = full[full["__is_train__"]==1].drop(columns=["__is_train__"])
    test_fe  = full[full["__is_train__"]==0].drop(columns=["__is_train__", args.target], errors="ignore")

    # 选择 features / cat_features（与训练脚本一致）
    from pandas.api.types import is_numeric_dtype, is_bool_dtype
    all_cols = train_fe.columns.tolist()
    features = [c for c in all_cols if c not in [args.id_col, args.target]]
    features = [c for c in features if (is_numeric_dtype(train_fe[c]) or is_bool_dtype(train_fe[c]) or str(train_fe[c].dtype)=="category")]
    cat_features = [c for c in features if str(train_fe[c].dtype)=="category"]

    print(f"[select] #features={len(features)}, #cat={len(cat_features)}")

    # 存缓存（让训练脚本直接 use_cache 读取）
    save_processed_cache_csv(
        cache_dir=args.cache_dir,
        train_fe=train_fe,
        test_fe=test_fe,
        features=features,
        cat_features=cat_features,
        id_col=args.id_col,
        target=args.target
    )

if __name__ == "__main__":
    main()