#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature Engineering v3 for Credit Risk (time-aware, leakage-safe)
================================================================
- 输入: 原始 train.csv / testA.csv
- 输出: data/processed_v3/{train_fe.csv, test_fe.csv, meta.json}
- 与现有训练脚本完全兼容: 训练脚本只需把 --cache_dir 指到 data/processed_v3

包含:
  * v1/v2 基础特征 (term/emp_len/issueDate/earliestCreditLine/fico/比率/岗位/标题/postCode 等)
  * v3 增强:
      - 强交叉: grade×term_bin, fico_bin×dti_bin, purpose×homeOwnership, issue_year×regionCode
      - 比率 & 稳定变换: 支付比例、余额/收入、开放额度比例 + log1p + winsor
      - 时间结构: 月份 sin/cos, vintage_rank, 信用史分桶 WOE(仅训练内, 折内), 历史滚动违约率(仅用过去)
      - 匿名 n0-n14 概括: n_pos_rate / n_top3 / n_entropy
  * 时间感知目标均值编码 (折内 + 仅过去窗口, 避免泄漏)
  * 超高基数类别降级 (频率编码兜底)

用法:
  python feature_engineering_v3.py \
    --train_path data/train.csv \
    --test_path data/testA.csv \
    --target isDefault \
    --id_col id \
    --out_cache_dir data/processed_v3
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
    ap.add_argument("--out_cache_dir", type=str, default="data/processed_v3")

    # 时间感知编码/滚动配置
    ap.add_argument("--time_col",   type=str, default="issueDate")
    ap.add_argument("--time_folds", type=int, default=5, help="时间分层折数 (用于TE)")
    ap.add_argument("--min_train_months", type=int, default=1, help="每个验证块前至少要有多少个月作为过去窗口")

    # 高基数类别阈值
    ap.add_argument("--high_card_threshold", type=int, default=300)
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

def winsor(s: pd.Series, p=0.005):
    try:
        lo, hi = s.quantile(p), s.quantile(1-p)
        return s.clip(lo, hi)
    except Exception:
        return s

def safe_cut(series, bins, labels=None):
    try:
        return pd.cut(series, bins=bins, labels=labels, include_lowest=True)
    except Exception:
        return pd.Series(pd.Categorical([np.nan]*len(series)))

def add_n_stats(df: pd.DataFrame, n_cols):
    n_df = df[n_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    n_mat = n_df.values
    df["n_sum"]      = n_df.sum(axis=1).astype("float32")
    df["n_mean"]     = n_df.mean(axis=1).astype("float32")
    df["n_max"]      = n_df.max(axis=1).astype("float32")
    df["n_std"]      = n_df.std(axis=1).astype("float32")
    df["n_nonzero"]  = (n_df > 0).sum(axis=1).astype("float32")
    # v3 扩展
    df["n_pos_rate"] = (n_mat > 0).mean(axis=1).astype("float32")
    top3 = np.partition(n_mat, -3, axis=1)[:, -3:]
    df["n_top3"]     = top3.sum(axis=1).astype("float32")
    # 简易熵 (逐列 minmax → Bernoulli 熵均值)
    nmin, nmax = n_mat.min(axis=0), n_mat.max(axis=0)
    nz = ((n_mat - nmin)/(nmax - nmin + 1e-6)).clip(1e-6, 1-1e-6)
    df["n_entropy"]  = (-(nz*np.log(nz) + (1-nz)*np.log(1-nz))).mean(axis=1).astype("float32")
    return df

# --------------------
# 时间相关辅助 (时间感知编码/滚动统计)
# --------------------
def add_time_keys(full: pd.DataFrame, time_col: str):
    full[f"{time_col}_dt"] = full[time_col].apply(parse_month)
    # month key: YYYYMM int
    key = full[f"{time_col}_dt"].dt.year.fillna(0).astype(int)*100 + full[f"{time_col}_dt"].dt.month.fillna(0).astype(int)
    full["time_key"] = key.astype(int)
    # 连续排名 (vintage)
    rank = full[f"{time_col}_dt"].rank(method="average")
    denom = max(int(full[f"{time_col}_dt"].notna().sum()), 1)
    full["vintage_rank"] = (rank/denom).fillna(0).astype("float32")
    # month sin/cos
    m = full[f"{time_col}_dt"].dt.month.fillna(1).astype(int)
    full["issue_m_sin"] = np.sin(2*np.pi*m/12).astype("float32")
    full["issue_m_cos"] = np.cos(2*np.pi*m/12).astype("float32")
    return full

def make_time_blocks(train_idx, time_keys, n_blocks=5, min_train_gap=1):
    """
    将时间key按升序分成 n_blocks 个连续块；对第 i 块做验证，训练集 = 该块开始之前的所有时间（至少 min_train_gap 个月）。
    返回: 列表 [(tr_mask, va_mask), ...] 仅对训练样本
    """
    tk = time_keys[train_idx]
    uniq = np.array(sorted(np.unique(tk[tk>0])))
    if len(uniq) < n_blocks:  # 时间月数较少时退化
        n_blocks = max(2, len(uniq))
    splits=[]
    # 块边界
    bins = np.array_split(uniq, n_blocks)
    for b in bins:
        if len(b)==0: continue
        start = b[0]
        # 至少留出 min_train_gap 个月的过去窗口
        past = uniq[uniq < start]
        if len(past) < min_train_gap:
            continue
        tr_mask = np.isin(tk, past)
        va_mask = np.isin(tk, b)
        splits.append((tr_mask, va_mask))
    # 若失败则退回普通 KFold 按时间排序
    if not splits:
        order = np.argsort(tk)
        folds = np.array_split(order, n_blocks)
        for i in range(n_blocks):
            va = folds[i]
            tr = np.concatenate([folds[j] for j in range(i)])
            m_tr = np.zeros_like(tk, dtype=bool); m_tr[tr]=True
            m_va = np.zeros_like(tk, dtype=bool); m_va[va]=True
            splits.append((m_tr, m_va))
    return splits

def time_aware_mean_encoding(full: pd.DataFrame, col: str, target: str,
                             time_key: str, n_blocks=5, min_train_gap=1):
    """
    时间感知目标均值编码 (折内仅用过去窗口)，返回修改后的 full 和新列名。
    关键点：
      - te 列预分配 float32
      - 被映射列按 object 视图处理，避免 map 返回 Categorical
      - 用 iloc + 明确的 numpy 数组赋值，避免 pandas lossy setitem
    """
    te_name = f"te_{col}"
    # 预分配为 float32，彻底避免 dtype 冲突
    full[te_name] = np.full(len(full), np.nan, dtype="float32")

    is_tr = full["__is_train__"].values == 1
    idx_tr = np.flatnonzero(is_tr)
    tk     = full.loc[idx_tr, time_key].to_numpy()
    splits = make_time_blocks(idx_tr, tk, n_blocks=n_blocks, min_train_gap=min_train_gap)
    prior  = float(full.loc[idx_tr, target].mean())

    # 以 object 视图避免 Categorical 赋值问题
    base_obj = full[col].astype("object")

    te_col_idx = full.columns.get_loc(te_name)

    for tr_mask, va_mask in splits:
        tr_idx = idx_tr[tr_mask]
        va_idx = idx_tr[va_mask]

        # 折内历史均值
        enc = pd.Series(full.loc[tr_idx, target].values)\
                .groupby(base_obj.iloc[tr_idx]).mean()

        mapped = base_obj.iloc[va_idx].map(enc)
        arr = mapped.to_numpy(dtype="float64")  # 先 float64，后转 float32 更稳
        arr = np.where(np.isnan(arr), prior, arr).astype("float32")

        # 用 iloc + 明确的列位置赋值
        full.iloc[va_idx, te_col_idx] = arr

    # 测试集：用全训练的均值
    enc_all = pd.Series(full.loc[idx_tr, target].values)\
                .groupby(base_obj.iloc[idx_tr]).mean()
    te_mask = ~is_tr
    mapped_te = base_obj.iloc[te_mask].map(enc_all)
    arr_te = mapped_te.to_numpy(dtype="float64")
    arr_te = np.where(np.isnan(arr_te), prior, arr_te).astype("float32")
    full.iloc[te_mask, te_col_idx] = arr_te

    return full, te_name

def time_aware_past_rate(full: pd.DataFrame, key_col: str, target: str, time_key: str):
    """
    历史滚动违约率 (仅用过去)，避免 dtype 冲突：
      - 结果列预分配 float32
      - 统一用 iloc + 明确列位置赋值
    """
    name = f"hist_rate_{key_col}"
    full[name] = np.full(len(full), np.nan, dtype="float32")
    name_col_idx = full.columns.get_loc(name)

    is_tr = full["__is_train__"].values == 1
    df = full.loc[is_tr, [key_col, target, time_key]].copy()
    df = df.sort_values(time_key)

    grp = df.groupby([key_col, time_key])[target].agg(["sum","count"]).reset_index()
    grp = grp.sort_values([key_col, time_key])
    grp["csum"] = grp.groupby(key_col)["sum"].cumsum().shift(1)
    grp["ccnt"] = grp.groupby(key_col)["count"].cumsum().shift(1)

    global_rate = float(grp["sum"].sum() / max(grp["count"].sum(), 1))
    grp["rate"] = (grp["csum"] / grp["ccnt"]).fillna(global_rate)

    # 训练/验证：映射 (key,time)
    m = grp.set_index([key_col, time_key])["rate"]
    idx = pd.MultiIndex.from_arrays([full[key_col].to_numpy(), full[time_key].to_numpy()])
    mapped_all = m.reindex(idx)

    arr_tr = mapped_all.to_numpy(dtype="float64")[is_tr]
    arr_tr = np.where(np.isnan(arr_tr), global_rate, arr_tr).astype("float32")
    full.iloc[is_tr, name_col_idx] = arr_tr

    # 测试：用每个 key 的最后一个 rate
    last_rate = grp.groupby(key_col, sort=False)["rate"].last()
    te_mask = ~is_tr
    arr_te = full.loc[te_mask, key_col].map(last_rate).to_numpy(dtype="float64")
    arr_te = np.where(np.isnan(arr_te), global_rate, arr_te).astype("float32")
    full.iloc[te_mask, name_col_idx] = arr_te

    return full, name

def woe_encode_from_bins(full: pd.DataFrame, col: str, target: str, bins=10):
    name = f"woe_{col}"
    is_tr = full["__is_train__"].values == 1
    s = pd.to_numeric(full[col], errors="coerce")

    # 仅用训练端做分箱
    binned_tr = pd.qcut(s[is_tr], q=min(bins, int(is_tr.sum())), duplicates="drop")
    # 用训练的箱边界切全体
    full[f"{col}_bin"] = pd.cut(s, bins=binned_tr.cat.categories, include_lowest=True)

    grp = full.loc[is_tr].groupby(f"{col}_bin")[target].agg(["sum","count"])
    pos = grp["sum"].sum(); neg = grp["count"].sum() - pos
    grp["woe"] = np.log(((grp["sum"]+0.5)/(pos+1.0)) / (((grp["count"]-grp["sum"])+0.5)/(neg+1.0)))

    # 结果列：直接生成 float32，避免中间 categorical
    arr = full[f"{col}_bin"].map(grp["woe"]).to_numpy(dtype="float64")
    full[name] = np.nan_to_num(arr, nan=0.0).astype("float32")

    full.drop(columns=[f"{col}_bin"], inplace=True)
    return full, name

# --------------------
# 主特征工程
# --------------------
def build_features_v3(train: pd.DataFrame, test: pd.DataFrame, args) -> pd.DataFrame:
    train["__is_train__"] = 1
    test["__is_train__"]  = 0
    full = pd.concat([train, test], axis=0, ignore_index=True)
    full = reduce_mem_usage(full, desc="Downcast (before FE)")

    # === 基础清洗/派生 ===
    # term
    if "term" in full.columns:
        full["term_mon"] = full["term"].apply(parse_term_general).astype("float32")
        full["term_bin"] = safe_cut(full["term_mon"], [-1,36,60,np.inf], ["<=36","37-60",">60"]).astype("category")

    # employmentLength
    if "employmentLength" in full.columns:
        full["emp_len_year"] = full["employmentLength"].apply(parse_employment_length).astype("float32")
        full["employmentLength_cat"] = full["employmentLength"].apply(normalize_employment_length_token).astype("category")

    # issueDate & earliestCreditLine
    full = add_time_keys(full, args.time_col)
    if "earliesCreditLine" in full.columns:
        full["earliest_dt"] = full["earliesCreditLine"].apply(parse_month)
        diff_days = (full[f"{args.time_col}_dt"] - full["earliest_dt"]).dt.days
        full["credit_hist_mon"] = (diff_days/30.0).astype("float32")

    # fico mean
    if {"ficoRangeLow","ficoRangeHigh"}.issubset(full.columns):
        full["fico_mean"] = ((full["ficoRangeLow"] + full["ficoRangeHigh"]) / 2.0).astype("float32")

    # 比率
    if {"installment","loanAmnt"}.issubset(full.columns):
        full["pay_ratio"] = (full["installment"]/(full["loanAmnt"]+1)).astype("float32")
    if {"revolBal","annualIncome"}.issubset(full.columns):
        full["revol_income"] = (full["revolBal"]/(full["annualIncome"]+1)).astype("float32")
    if {"openAcc","totalAcc"}.issubset(full.columns):
        full["open_ratio"] = (full["openAcc"]/(full["totalAcc"]+1)).astype("float32")

    # log1p + winsor
    for c in ["loanAmnt","annualIncome","revolBal","revolUtil","dti"]:
        if c in full.columns:
            full[f"{c}_log1p"] = np.log1p(pd.to_numeric(full[c], errors="coerce").clip(lower=0)).astype("float32")
            full[f"{c}_w"]     = winsor(pd.to_numeric(full[c], errors="coerce")).astype("float32")

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

    # n 系列统计增强
    n_cols = [f"n{i}" for i in range(15) if f"n{i}" in full.columns]
    if n_cols:
        full = add_n_stats(full, n_cols)

    # === v3: 强交叉 ===
    # 1) grade × term_bin
    if {"grade","term_bin"}.issubset(full.columns):
        full["grade_term"] = full["grade"].astype(str) + "_" + full["term_bin"].astype(str)
        full["grade_term_freq"] = full["grade_term"].map(full.loc[full["__is_train__"]==1,"grade_term"].value_counts()).fillna(0).astype("float32").values
    # 2) fico_bin × dti_bin
    if {"fico_mean","dti"}.issubset(full.columns):
        full["fico_bin"] = pd.qcut(full["fico_mean"], q=20, duplicates="drop")
        full["dti_bin"]  = pd.qcut(pd.to_numeric(full["dti"], errors="coerce"), q=20, duplicates="drop")
        full["fico_dti"] = full["fico_bin"].astype(str) + "_" + full["dti_bin"].astype(str)
        full["fico_dti_freq"] = full["fico_dti"].map(full.loc[full["__is_train__"]==1, "fico_dti"].value_counts()).fillna(0).astype("float32").values
    # 3) purpose × homeOwnership
    if {"purpose","homeOwnership"}.issubset(full.columns):
        full["purpose_home"] = full["purpose"].astype(str) + "_" + full["homeOwnership"].astype(str)
        full["purpose_home_freq"] = full["purpose_home"].map(full.loc[full["__is_train__"]==1,"purpose_home"].value_counts()).fillna(0).astype("float32").values
    # 4) issue_year × regionCode
    if {"regionCode", f"{args.time_col}_dt"}.issubset(full.columns):
        iy = full[f"{args.time_col}_dt"].dt.year.fillna(0).astype(int).astype(str)
        full["iy_region"] = iy + "_" + full["regionCode"].astype(str)
        full["iy_region_freq"] = full["iy_region"].map(full.loc[full["__is_train__"]==1,"iy_region"].value_counts()).fillna(0).astype("float32").values

    # === 时间感知目标均值编码 (泄漏安全) ===
    target = args.target
    time_key = "time_key"
    te_cols = []
    # 对交叉列/高价值类别做 TE
    for c in ["grade_term","fico_dti","purpose_home","iy_region","subGrade","employmentLength_cat","postCode_cat"]:
        if c in full.columns:
            full, te_name = time_aware_mean_encoding(full, c, target, time_key, n_blocks=args.time_folds, min_train_gap=args.min_train_months)
            te_cols.append(te_name)

    # === 历史滚动违约率 (仅过去) ===
    for key_col in ["regionCode","grade","purpose","postCode_cat"]:
        if key_col in full.columns:
            full, _ = time_aware_past_rate(full, key_col, target, time_key)

    # === WOE on 数值分箱 (非时间感知, 描述单调结构) ===
    for nc in ["credit_hist_mon","revolUtil","dti"]:
        if nc in full.columns:
            full, _ = woe_encode_from_bins(full, nc, target, bins=10)

    # === 清理/填充 ===
    # remove intermediate date cols
    for col in [f"{args.time_col}_dt","earliest_dt"]:
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
    ensure_dir(args.out_cache_dir)

    # 读原始
    train = pd.read_csv(args.train_path, low_memory=False)
    test  = pd.read_csv(args.test_path,  low_memory=False)
    print(f"[load] train={train.shape}, test={test.shape}")

    # 构造 v3 特征
    full = build_features_v3(train.copy(), test.copy(), args)

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
        "time_col": args.time_col,
        "time_folds": args.time_folds,
        "high_card_threshold": args.high_card_threshold
    }
    save_processed_cache_csv(args.out_cache_dir, train_fe, test_fe, features, cat_features, meta_extra)

    print(f"[features] total={len(features)} (num/bool={len(num_bool)}, cat={len(cat_features)})")
    print("[done] You can now run training with:  --cache_dir data/processed_v3")

if __name__ == "__main__":
    main()