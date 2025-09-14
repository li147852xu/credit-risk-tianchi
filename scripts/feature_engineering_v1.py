#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature Engineering v1 for Credit Risk (Basic Version)
======================================================
- 输入: 原始 train.csv / testA.csv
- 输出: data/processed_v1/{train_fe.csv, test_fe.csv, meta.json}
- 与现有训练脚本完全兼容: 训练脚本只需把 --cache_dir 指到 data/processed_v1

包含基础特征工程:
  * 基础解析/枚举: term/emp_len/issueDate/earliestCreditLine/fico/比率/岗位/标题/postCode 等
  * 基础比率特征: loan_income_ratio, installment_to_income 等
  * 基础统计: n系列统计特征
  * 类别特征: 低基数整数转category，高基数类别降级
  * 基础清洗: 缺失值填充，异常值处理

用法:
  python feature_engineering_v1.py \
    --train_path data/train.csv \
    --test_path data/testA.csv \
    --target isDefault \
    --id_col id \
    --cache_dir data/processed_v1
"""

import os
import re
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# --------------------
# 参数
# --------------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", type=str, default="data/train.csv")
    ap.add_argument("--test_path",  type=str, default="data/testA.csv")
    ap.add_argument("--target",     type=str, default="isDefault")
    ap.add_argument("--id_col",     type=str, default="id")
    ap.add_argument("--cache_dir",  type=str, default="data/processed_v1")
    
    # 高基数类别阈值
    ap.add_argument("--high_card_threshold", type=int, default=200)
    return ap.parse_args()

# --------------------
# 工具函数
# --------------------
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def reduce_mem_usage(df: pd.DataFrame, verbose=True, desc="Downcast"):
    start = df.memory_usage().sum() / 1024**2
    for col in tqdm(df.columns, desc=desc, leave=False, dynamic_ncols=True):
        c = df[col]
        if pd.api.types.is_integer_dtype(c):
            cmin, cmax = c.min(), c.max()
            if cmin >= 0:
                if cmax < 255: df[col] = c.astype(np.uint8)
                elif cmax < 65535: df[col] = c.astype(np.uint16)
                elif cmax < 4294967295: df[col] = c.astype(np.uint32)
                else: df[col] = c.astype(np.uint64)
            else:
                if np.iinfo(np.int8).min <= cmin and cmax <= np.iinfo(np.int8).max: df[col] = c.astype(np.int8)
                elif np.iinfo(np.int16).min <= cmin and cmax <= np.iinfo(np.int16).max: df[col] = c.astype(np.int16)
                elif np.iinfo(np.int32).min <= cmin and cmax <= np.iinfo(np.int32).max: df[col] = c.astype(np.int32)
                else: df[col] = c.astype(np.int64)
        elif pd.api.types.is_float_dtype(c):
            df[col] = c.astype(np.float32)
    end = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"[reduce_mem_usage] {start:.2f} -> {end:.2f} MB")
    return df

def parse_month(s):
    if pd.isna(s): return pd.NaT
    s = str(s).strip()
    for fmt in ("%Y/%m/%d","%Y-%m-%d","%Y-%m","%Y/%m","%b-%y","%b-%Y"):
        try: return pd.to_datetime(s, format=fmt)
        except: pass
    return pd.to_datetime(s, errors="coerce")

def parse_term_general(x):
    if pd.isna(x): return np.nan
    if isinstance(x,(int,float)): return float(x)*12 if x<=10 else float(x)
    s=str(x).lower()
    m=re.findall(r"\d+", s)
    if not m: return np.nan
    n=float(m[0])
    return n*12 if "year" in s else n

def parse_employment_length(x):
    if pd.isna(x): return np.nan
    s=str(x).lower().strip()
    if "10" in s: return 10.0
    if "<" in s:  return 0.0
    m=re.findall(r"\d+", s)
    return float(m[0]) if m else np.nan

def normalize_employment_length_token(x):
    if pd.isna(x): return "na"
    s=str(x).lower().strip()
    if "10" in s: return "10+"
    if "<" in s: return "<1"
    m=re.findall(r"\d+", s)
    return m[0] if m else "na"

def to_3digit_postcode_str(series: pd.Series):
    s = series.astype(str).str.split(".").str[0]
    s = s.where(~s.isin(["nan","None","none","NaN"]), other="")
    return s.str.zfill(3)

def freq_encode(series: pd.Series):
    vc = series.astype(str).value_counts(dropna=False)
    mapping = (vc / vc.sum()).astype("float32")
    return series.astype(str).map(mapping).astype("float32")

def safe_cut(series, bins, labels=None):
    try:
        return pd.cut(series, bins=bins, labels=labels, include_lowest=True)
    except Exception:
        return pd.Series(pd.Categorical([np.nan]*len(series)))

def add_n_stats(df: pd.DataFrame, n_cols):
    n_df = df[n_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df["n_sum"]      = n_df.sum(axis=1).astype("float32")
    df["n_mean"]     = n_df.mean(axis=1).astype("float32")
    df["n_max"]      = n_df.max(axis=1).astype("float32")
    df["n_std"]      = n_df.std(axis=1).astype("float32")
    df["n_nonzero"]  = (n_df > 0).sum(axis=1).astype("float32")
    return df

# --------------------
# 主特征工程
# --------------------
def build_features_v1(train: pd.DataFrame, test: pd.DataFrame, args) -> pd.DataFrame:
    train["__is_train__"] = 1
    test["__is_train__"]  = 0
    full = pd.concat([train, test], axis=0, ignore_index=True)
    full = reduce_mem_usage(full, desc="Downcast (before FE)")

    # === 基础解析/派生 ===
    # term
    if "term" in full.columns:
        full["term_mon"] = full["term"].apply(parse_term_general).astype("float32")
        full["term_bin"] = safe_cut(full["term_mon"], [-1,36,60,np.inf], ["<=36","37-60",">60"]).astype("category")

    # employmentLength
    if "employmentLength" in full.columns:
        full["emp_len_year"] = full["employmentLength"].apply(parse_employment_length).astype("float32")
        full["employmentLength_cat"] = full["employmentLength"].apply(normalize_employment_length_token).astype("category")

    # issueDate & earliestCreditLine
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

    # fico mean
    if {"ficoRangeLow","ficoRangeHigh"}.issubset(full.columns):
        full["fico_mean"] = ((full["ficoRangeLow"] + full["ficoRangeHigh"]) / 2.0).astype("float32")

    # 基础比率特征
    if {"loanAmnt","annualIncome"}.issubset(full.columns):
        full["loan_income_ratio"] = (full["loanAmnt"] / (full["annualIncome"] + 1)).astype("float32")
        if "installment" in full.columns:
            full["install_to_income"] = (full["installment"] / (full["annualIncome"] + 1)).astype("float32")

    # 岗位/标题 & postCode
    for col in ["employmentTitle","title"]:
        if col in full.columns:
            s = full[col].astype(str)
            full[f"{col}_cat"]  = s.astype("category")
            full[f"{col}_freq"] = freq_encode(s)

    if "postCode" in full.columns:
        pc3 = to_3digit_postcode_str(full["postCode"])
        full["postCode_cat"]  = pc3.astype("category")
        full["postCode_freq"] = freq_encode(pc3)

    # 离散/低基数整数作为 category
    low_card_int_as_cat = ["grade","subGrade","homeOwnership","verificationStatus","purpose","regionCode","initialListStatus","applicationType","policyCode"]
    for c in [x for x in low_card_int_as_cat if x in full.columns]:
        if str(full[c].dtype) != "category":
            full[c] = full[c].astype("category")

    # n 系列统计
    n_cols = [f"n{i}" for i in range(15) if f"n{i}" in full.columns]
    if n_cols:
        full = add_n_stats(full, n_cols)

    # === 清理/填充 ===
    # remove intermediate date cols
    for col in ["issueDate_dt","earliest_dt"]:
        if col in full.columns: full.drop(columns=[col], inplace=True)
    
    # 数值: inf/nan → 中位数
    num_cols_full = full.select_dtypes(include=["number"]).columns
    full[num_cols_full] = full[num_cols_full].replace([np.inf,-np.inf], np.nan)
    full[num_cols_full] = full[num_cols_full].fillna(full[num_cols_full].median())

    # 超高基数类别降级
    cat_cols_now = [c for c in full.columns if str(full[c].dtype)=="category"]
    card = {c: full[c].nunique(dropna=False) for c in cat_cols_now}
    high_card = [c for c,v in card.items() if v > args.high_card_threshold]
    if high_card:
        print(f"[high-card] downgrade: {high_card}")
        for c in high_card:
            fcf = f"{c}_freq"
            if fcf not in full.columns:
                full[fcf] = freq_encode(full[c].astype(str))
            full[c] = full[c].astype(str)  # 降级为 string，避免后续 cat_features 捕获
    
    # 再次降内存
    full = reduce_mem_usage(full, desc="Downcast (after FE)")
    return full

# --------------------
# 保存缓存 (CSV 协议)
# --------------------
def save_processed_cache_csv(out_dir: str,
                             train_fe: pd.DataFrame,
                             test_fe: pd.DataFrame,
                             features: list,
                             cat_features: list,
                             meta_extra: dict = None):
    out = Path(out_dir)
    ensure_dir(out)
    train_csv = out / "train_fe.csv"
    test_csv  = out / "test_fe.csv"
    meta_js   = out / "meta.json"

    train_fe.to_csv(train_csv, index=False)
    test_fe.to_csv(test_csv, index=False)

    meta = {
        "format": "csv",
        "features": features,
        "cat_features": cat_features,
        "n_train": int(len(train_fe)),
        "n_test": int(len(test_fe))
    }
    if meta_extra: meta.update(meta_extra)
    with open(meta_js, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[cache] saved -> {train_csv}\n                 {test_csv}\n                 {meta_js}")

# --------------------
# 主流程
# --------------------
def main():
    args = get_args()
    ensure_dir(args.cache_dir)

    # 读原始
    train = pd.read_csv(args.train_path, low_memory=False)
    test  = pd.read_csv(args.test_path,  low_memory=False)
    print(f"[load] train={train.shape}, test={test.shape}")

    # 构造 v1 特征
    full = build_features_v1(train.copy(), test.copy(), args)

    # 切回 train/test
    train_fe = full[full["__is_train__"]==1].drop(columns=["__is_train__"])
    test_fe  = full[full["__is_train__"]==0].drop(columns=["__is_train__", args.target], errors="ignore")

    # 选择可训练的特征:
    #   - 数值 或 bool 或 category
    num_bool = list(train_fe.select_dtypes(include=["number","bool"]).columns)
    cat_cols = [c for c in train_fe.columns if str(train_fe[c].dtype)=="category"]
    # 去掉 id/target
    features = [c for c in (num_bool + cat_cols) if c not in [args.id_col, args.target]]
    cat_features = [c for c in cat_cols if c not in [args.id_col, args.target]]

    # 元信息（便于训练脚本复原 dtype）
    meta_extra = {
        "id_col": args.id_col,
        "target": args.target,
        "high_card_threshold": args.high_card_threshold
    }
    save_processed_cache_csv(args.cache_dir, train_fe, test_fe, features, cat_features, meta_extra)

    print(f"[features] total={len(features)} (num/bool={len(num_bool)}, cat={len(cat_features)})")
    print("[done] You can now run training with:  --cache_dir data/processed_v1")

if __name__ == "__main__":
    main()
