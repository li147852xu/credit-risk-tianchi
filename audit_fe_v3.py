#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Audit v3 Processed Features for Credit Risk
===========================================

用途：
- 对 data/processed_v3 的缓存做体检，验证时间感知编码是否无泄漏、历史滚动率是否正确、
  以及常规缺失/无穷/类型/基数问题。
- 需要原始 train.csv（用于取目标列 & 再核验）。

用法示例：
  python audit_fe_v3.py \
    --cache_dir data/processed_v3 \
    --raw_train data/train.csv \
    --target isDefault \
    --id_col id \
    --time_key time_key \
    --report_dir output/audit_v3 \
    --sample_rows 400000 \
    --max_months 36
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score

# ---------------------------
# 参数
# ---------------------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, default="data/processed_v3")
    ap.add_argument("--raw_train", type=str, default="data/train.csv")
    ap.add_argument("--target", type=str, default="isDefault")
    ap.add_argument("--id_col", type=str, default="id")
    ap.add_argument("--time_key", type=str, default="time_key")
    ap.add_argument("--report_dir", type=str, default="output/audit_v3")
    ap.add_argument("--sample_rows", type=int, default=0, help=">0 时对训练数据下采样加速")
    ap.add_argument("--max_months", type=int, default=0, help=">0 时仅检查最近 N 个 time_key 月份")
    ap.add_argument("--seed", type=int, default=2025)
    return ap.parse_args()

# ---------------------------
# 小工具
# ---------------------------
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def safe_auc(y_true, y_score):
    # 单类/NaN 保护
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if np.unique(y_true).size < 2:
        return np.nan
    mask = np.isfinite(y_score)
    if mask.sum() < 2:
        return np.nan
    return roc_auc_score(y_true[mask], y_score[mask])

def load_cache(cache_dir: str, target: str):
    cache_dir = Path(cache_dir)
    train_csv = cache_dir / "train_fe.csv"
    test_csv  = cache_dir / "test_fe.csv"
    meta_js   = cache_dir / "meta.json"
    if not (train_csv.exists() and test_csv.exists() and meta_js.exists()):
        raise FileNotFoundError(f"cache files not complete in {cache_dir}")
    with open(meta_js, "r", encoding="utf-8") as f:
        meta = json.load(f)

    features = meta.get("features", [])
    cat_features = meta.get("cat_features", [])
    # 按 v3 缓存协议读取：先全 str，再按 meta 复原
    train_fe = pd.read_csv(train_csv, dtype=str, low_memory=False)
    test_fe  = pd.read_csv(test_csv,  dtype=str, low_memory=False)

    # 数值列：features 去掉 cat_features 即 non-cat
    non_cat = [c for c in features if c not in cat_features]
    for c in non_cat:
        if c in train_fe.columns:
            train_fe[c] = pd.to_numeric(train_fe[c], errors="coerce")
        if c in test_fe.columns:
            test_fe[c] = pd.to_numeric(test_fe[c], errors="coerce")
    # 类别列
    for c in cat_features:
        if c in train_fe.columns:
            train_fe[c] = train_fe[c].astype("category")
        if c in test_fe.columns:
            test_fe[c] = test_fe[c].astype("category")
    # 目标列（可能在 train_fe 已保留）
    if target in train_fe.columns:
        train_fe[target] = pd.to_numeric(train_fe[target], errors="coerce")

    return train_fe, test_fe, meta

def basic_profile(train_fe: pd.DataFrame, target: str):
    info = {}
    info["shape"] = tuple(train_fe.shape)
    info["n_cols"] = train_fe.shape[1]
    info["n_num"]  = int(train_fe.select_dtypes(include=["number"]).shape[1])
    info["n_cat"]  = int((train_fe.dtypes == "category").sum())
    info["n_obj"]  = int((train_fe.dtypes == "object").sum())
    info["target_pos_rate"] = float(train_fe[target].mean()) if target in train_fe.columns else None
    return info

def list_by_prefix(cols, p):
    return [c for c in cols if c.startswith(p)]

def list_like(cols, substr):
    return [c for c in cols if substr in c]

# ---------------------------
# 审计 1：缺失/无穷/类型/基数
# ---------------------------
def audit_missing_inf_types(train_fe: pd.DataFrame, cat_features: list, report_dir: Path):
    miss = []
    infs = []
    for c in train_fe.columns:
        s = train_fe[c]
        if s.dtype.kind in "biufc":  # 数值
            m = s.isna().mean()
            n_inf = np.isinf(s.astype("float64")).sum()
            miss.append((c, float(m)))
            infs.append((c, int(n_inf)))
    miss_df = pd.DataFrame(miss, columns=["col","nan_ratio"]).sort_values("nan_ratio", ascending=False)
    infs_df = pd.DataFrame(infs, columns=["col","n_inf"]).sort_values("n_inf", ascending=False)
    miss_df.to_csv(report_dir/"missing_ratio.csv", index=False)
    infs_df.to_csv(report_dir/"inf_counts.csv", index=False)

    # 类别基数
    cards = []
    for c in cat_features:
        if c in train_fe.columns:
            cards.append((c, int(train_fe[c].nunique(dropna=False))))
    card_df = pd.DataFrame(cards, columns=["cat_col","nunique"]).sort_values("nunique", ascending=False)
    card_df.to_csv(report_dir/"category_cardinality.csv", index=False)
    return miss_df, infs_df, card_df

# ---------------------------
# 审计 2：WOE 列健康
# ---------------------------
def audit_woe(train_fe: pd.DataFrame, target: str, report_dir: Path):
    woe_cols = list_by_prefix(train_fe.columns, "woe_")
    rows = []
    for c in woe_cols:
        s = pd.to_numeric(train_fe[c], errors="coerce")
        nanr = float(s.isna().mean())
        ninf = int(np.isinf(s.astype("float64")).sum())
        auc  = safe_auc(train_fe[target].values, s.values) if target in train_fe.columns else np.nan
        rows.append((c, nanr, ninf, float(np.nanmean(s)), float(np.nanstd(s)), auc))
    df = pd.DataFrame(rows, columns=["col","nan_ratio","n_inf","mean","std","auc_vs_target"]).sort_values("auc_vs_target", ascending=False)
    df.to_csv(report_dir/"woe_summary.csv", index=False)
    return df

# ---------------------------
# 审计 3：时间感知 TE 无泄漏核验
# （逐月重算“过去均值”并对比 te_* 列）
# ---------------------------
def audit_time_aware_te(train_fe: pd.DataFrame, target: str, time_key: str, report_dir: Path, max_months: int = 0):
    """
    审计时间感知 TE 是否“只用过去”：
      - 对每个月 m：用 <m 的历史样本计算 base→mean 映射，
        与缓存中的 te_<base> 在该月的取值做 MAE/RMSE 对比。
    修复点：
      - 基列用 object 视图，避免 map 返回 Categorical
      - 映射与比较全转 numpy 流程，避免 pandas lossy setitem 错
      - groupby(observed=True) 消除 FutureWarning
    """
    te_cols = [c for c in train_fe.columns if c.startswith("te_")]
    if not te_cols or time_key not in train_fe.columns:
        print("[TE audit] te_* 或 time_key 缺失，跳过。")
        return pd.DataFrame(columns=["te_col","mae","rmse","check_months","rows_compared"])

    df = train_fe[[time_key, target] + te_cols].copy()
    df = df.sort_values(time_key)
    uniq = np.array(sorted(df[time_key].dropna().unique()))
    if max_months and len(uniq) > max_months:
        uniq = uniq[-max_months:]

    results = []
    for te in tqdm(te_cols, desc="TE audit", leave=False):
        base = te[len("te_"):]
        # 有些 te_* 可能对应的是交叉列（如 grade_term），必须存在才可校验
        if base not in train_fe.columns:
            results.append((te, np.nan, np.nan, int(len(uniq)), 0))
            continue

        base_obj = train_fe[base].astype("object")  # 避免 Categorical 链式问题
        mae_list, rmse_list, n_total = [], [], 0

        for i, m in enumerate(uniq):
            past_idx = df.index[df[time_key] <  m]
            cur_idx  = df.index[df[time_key] == m]
            if len(past_idx) == 0 or len(cur_idx) == 0:
                continue

            # 历史均值 (只用过去) —— 注意 observed=True
            past_enc = pd.Series(train_fe.loc[past_idx, target].values) \
                         .groupby(base_obj.loc[past_idx], observed=True).mean()

            mapped = base_obj.loc[cur_idx].map(past_enc)  # Series，可能含 NaN
            prior  = float(train_fe.loc[past_idx, target].mean())

            mapped_arr = mapped.to_numpy(dtype="float64")
            mapped_arr = np.where(np.isnan(mapped_arr), prior, mapped_arr)

            real_arr = pd.to_numeric(train_fe.loc[cur_idx, te], errors="coerce") \
                         .to_numpy(dtype="float64")

            mask = np.isfinite(mapped_arr) & np.isfinite(real_arr)
            if mask.sum() == 0:
                continue

            diff = mapped_arr[mask] - real_arr[mask]
            mae_list.append(np.mean(np.abs(diff)))
            rmse_list.append(np.sqrt(np.mean(diff**2)))
            n_total += int(mask.sum())

        mae  = float(np.mean(mae_list)) if mae_list else np.nan
        rmse = float(np.mean(rmse_list)) if rmse_list else np.nan
        results.append((te, mae, rmse, int(len(uniq)), int(n_total)))

    out = pd.DataFrame(results, columns=["te_col","mae","rmse","check_months","rows_compared"]) \
            .sort_values(["mae","rmse"], na_position="last")
    out.to_csv(report_dir/"te_time_aware_check.csv", index=False)
    return out

# ---------------------------
# 审计 4：历史滚动违约率核验
# （重算 csum/ccnt 的滞后率并与 hist_rate_* 对比）
# ---------------------------
def audit_hist_rate(train_fe: pd.DataFrame, target: str, time_key: str, report_dir: Path, max_months: int = 0):
    cand = list_by_prefix(train_fe.columns, "hist_rate_")
    results = []
    if time_key not in train_fe.columns or not cand:
        print("[hist audit] time_key or hist_rate_* not found. skip.")
        return pd.DataFrame(columns=["hist_col","key_col","mae","rmse","rows_compared"])
    # key 名： hist_rate_<key>
    for h in tqdm(cand, desc="HIST audit", leave=False):
        key_col = h.replace("hist_rate_","")
        if key_col not in train_fe.columns:
            continue
        df = train_fe[[time_key, target, key_col, h]].copy()
        df = df.sort_values(time_key)
        uniq = np.array(sorted(df[time_key].dropna().unique()))
        if max_months and len(uniq) > max_months:
            uniq = uniq[-max_months:]
            df = df[df[time_key].isin(uniq)]

        grp = df.groupby([key_col, time_key], observed=True)[target].agg(["sum","count"]).reset_index()
        grp = grp.sort_values([key_col, time_key])
        grp["csum"] = grp.groupby(key_col, observed=True)["sum"].cumsum().shift(1)
        grp["ccnt"] = grp.groupby(key_col, observed=True)["count"].cumsum().shift(1)
        global_rate = float(grp["sum"].sum() / max(grp["count"].sum(),1))
        grp["rate"] = (grp["csum"]/grp["ccnt"]).fillna(global_rate)

        m = grp.set_index([key_col, time_key])["rate"]
        idx = pd.MultiIndex.from_arrays([df[key_col].values, df[time_key].values])
        mapped = m.reindex(idx).to_numpy(dtype="float64")
        real   = df[h].to_numpy(dtype="float64")
        mask = np.isfinite(mapped) & np.isfinite(real)
        mae = float(np.mean(np.abs(mapped[mask]-real[mask]))) if mask.sum() else np.nan
        rmse= float(np.sqrt(np.mean((mapped[mask]-real[mask])**2))) if mask.sum() else np.nan
        results.append((h, key_col, mae, rmse, int(mask.sum())))
    out = pd.DataFrame(results, columns=["hist_col","key_col","mae","rmse","rows_compared"]).sort_values("mae")
    out.to_csv(report_dir/"hist_rate_check.csv", index=False)
    return out

# ---------------------------
# 审计 5：按月 AUC（时序稳健性）
# ---------------------------
def audit_monthly_auc(train_fe: pd.DataFrame, target: str, time_key: str, score_cols: list, report_dir: Path, max_months: int = 0):
    if time_key not in train_fe.columns:
        print("[monthly auc] time_key missing; skip.")
        return pd.DataFrame()
    uniq = np.array(sorted(train_fe[time_key].dropna().unique()))
    if max_months and len(uniq) > max_months:
        uniq = uniq[-max_months:]

    rows = []
    for m in uniq:
        mask = train_fe[time_key] == m
        y = train_fe.loc[mask, target].values
        for c in score_cols:
            s = pd.to_numeric(train_fe.loc[mask, c], errors="coerce").values
            auc = safe_auc(y, s)
            rows.append((int(m), c, auc, int(mask.sum())))
    out = pd.DataFrame(rows, columns=["time_key","col","auc","n"]).sort_values(["col","time_key"])
    out.to_csv(report_dir/"monthly_auc.csv", index=False)
    return out

# ---------------------------
# 主流程
# ---------------------------
def main():
    args = get_args()
    np.random.seed(args.seed)

    ensure_dir(args.report_dir)
    train_fe, test_fe, meta = load_cache(args.cache_dir, target=args.target)
    print(f"[load] train_fe={train_fe.shape}, test_fe={test_fe.shape}")

    # 需要目标列；如果缓存里没带，读原始 train.csv merge
    if args.target not in train_fe.columns:
        raw = pd.read_csv(args.raw_train, low_memory=False, usecols=[args.id_col, args.target])
        train_fe = train_fe.merge(raw, on=args.id_col, how="left")
        print("[merge] target merged from raw train.csv")

    # 采样（可选）
    if args.sample_rows and args.sample_rows > 0 and len(train_fe) > args.sample_rows:
        train_fe = train_fe.sample(n=args.sample_rows, random_state=args.seed)
        print(f"[sample] downsampled train_fe -> {train_fe.shape}")

    # 基本信息
    prof = basic_profile(train_fe, args.target)
    with open(Path(args.report_dir)/"basic_profile.json", "w", encoding="utf-8") as f:
        json.dump(prof, f, indent=2, ensure_ascii=False)
    print("[profile] saved basic_profile.json")

    # 1) 缺失/无穷/类别基数
    miss_df, infs_df, card_df = audit_missing_inf_types(train_fe, meta.get("cat_features", []), Path(args.report_dir))

    # 2) WOE 健康
    _woe = audit_woe(train_fe, args.target, Path(args.report_dir))

    # 3) TE 时间感知无泄漏核验
    _te = audit_time_aware_te(train_fe, args.target, args.time_key, Path(args.report_dir), max_months=args.max_months)

    # 4) 历史滚动违约率核验
    _hist = audit_hist_rate(train_fe, args.target, args.time_key, Path(args.report_dir), max_months=args.max_months)

    # 5) 按月 AUC（挑若干代表列：te_*/hist_rate_*）
    score_cols = []
    score_cols += list_by_prefix(train_fe.columns, "te_")[:10]  # 取前10个以控制体量
    score_cols += list_by_prefix(train_fe.columns, "hist_rate_")[:10]
    score_cols += list_by_prefix(train_fe.columns, "woe_")[:10]
    score_cols = list(dict.fromkeys(score_cols))  # 去重保持顺序
    _mauc = audit_monthly_auc(train_fe, args.target, args.time_key, score_cols, Path(args.report_dir), max_months=args.max_months)

    # 汇总 Markdown 报告
    md = []
    md.append("# FE v3 审计报告")
    md.append("")
    md.append("## 基本信息")
    md.append(f"- train_fe: **{prof['shape']}** | 数值列: {prof['n_num']} | 类别列: {prof['n_cat']} | 目标正例率: {prof['target_pos_rate']:.6f}")
    md.append("")
    md.append("## 缺失/无穷/类别基数")
    if not miss_df.empty:
        top_miss = miss_df.head(10).to_string(index=False)
        md.append("**缺失率 Top10（数值列）**\n\n```\n"+top_miss+"\n```")
    if not infs_df.empty:
        top_inf = infs_df.head(10).to_string(index=False)
        md.append("\n**无穷值 Top10（数值列）**\n\n```\n"+top_inf+"\n```")
    if not card_df.empty:
        top_card = card_df.head(10).to_string(index=False)
        md.append("\n**类别基数 Top10**\n\n```\n"+top_card+"\n```")
    md.append("")
    md.append("## 时间感知目标编码（te_*）无泄漏检测")
    if _te.empty:
        md.append("- 未找到 te_* 或 time_key，跳过。")
    else:
        bad = _te[(_te["mae"]>1e-6) & (_te["rows_compared"]>0)].sort_values("mae", ascending=False)
        md.append(_te.head(10).to_string(index=False))
        md.append("\n判定：若 `mae/rmse` 接近 0（1e-6 量级），说明实现与“仅用过去”一致；若明显偏大，可能存在实现偏差。")
        if not bad.empty:
            md.append("\n**疑似异常列（MAE>1e-6）Top5**\n\n```\n"+bad.head(5).to_string(index=False)+"\n```")

    md.append("\n## 历史滚动违约率（hist_rate_*）核验")
    if _hist.empty:
        md.append("- 未找到 hist_rate_* 或 time_key，跳过。")
    else:
        badh = _hist[(_hist["mae"]>1e-6) & (_hist["rows_compared"]>0)].sort_values("mae", ascending=False)
        md.append(_hist.head(10).to_string(index=False))
        if not badh.empty:
            md.append("\n**疑似异常列（MAE>1e-6）Top5**\n\n```\n"+badh.head(5).to_string(index=False)+"\n```")

    md.append("\n## 按月 AUC（抽样列）")
    if not _mauc.empty:
        md.append(_mauc.head(20).to_string(index=False))
        md.append("\n提示：关注某些月份 AUC 明显异常（过高/过低），可能存在时序分布差异或泄漏风险。")

    report_path = Path(args.report_dir)/"audit_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[done] report written to: {report_path}")
    print(f"[tips] 更多细节见 CSV：missing_ratio.csv / inf_counts.csv / category_cardinality.csv / te_time_aware_check.csv / hist_rate_check.csv / woe_summary.csv / monthly_auc.csv")

if __name__ == "__main__":
    main()