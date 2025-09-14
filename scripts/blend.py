#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Blending Script
====================
模型融合脚本，支持多种融合策略
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
from scipy.stats import spearmanr
from scipy.optimize import minimize


def get_args():
    """获取命令行参数"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", type=str, default="outputs", help="模型输出目录")
    ap.add_argument("--oof_pattern", type=str, default="oof_*.csv", help="OOF文件模式")
    ap.add_argument("--sub_pattern", type=str, default="submission*.csv", help="提交文件模式")
    ap.add_argument("--id_col", type=str, default="id", help="ID列名")
    ap.add_argument("--target", type=str, default="isDefault", help="目标列名")

    # 相关性过滤参数
    ap.add_argument("--corr_threshold", type=float, default=0.985, help="相关性阈值")
    ap.add_argument("--min_models", type=int, default=3, help="最少模型数")
    
    # 融合参数
    ap.add_argument("--seed", type=int, default=2025, help="随机种子")
    ap.add_argument("--folds", type=int, default=5, help="交叉验证折数")
    ap.add_argument("--logreg_C", type=float, default=1.0, help="逻辑回归C参数")
    ap.add_argument("--ridge_alpha", type=float, default=2.0, help="Ridge回归alpha参数")

    # 并行参数
    ap.add_argument("--n_jobs", type=int, default=32, help="并行核数")

    # 权重优化参数
    ap.add_argument("--weight_opt_iters", type=int, default=6000, help="权重优化迭代数")
    ap.add_argument("--weight_opt_restarts", type=int, default=8, help="多起点重启数")

    # 贪心选择参数
    ap.add_argument("--greedy_topk", type=int, default=6, help="贪心选择topk")
    ap.add_argument("--greedy_max_models", type=int, default=8, help="贪心最大模型数")
    ap.add_argument("--greedy_iters_each", type=int, default=400, help="贪心每步迭代数")
    ap.add_argument("--greedy_patience", type=int, default=2, help="贪心耐心参数")
    ap.add_argument("--greedy_min_gain", type=float, default=5e-5, help="贪心最小增益")
    
    return ap.parse_args()


def timestamp():
    """生成时间戳"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)


def scan_runs(root_dir, oof_pattern, sub_pattern):
    """扫描模型运行结果"""
    runs = []
    root = Path(root_dir)
    if not root.exists():
        return runs
        
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
            
        oofs = sorted(d.glob(oof_pattern))
        subs = sorted(d.glob(sub_pattern))
        if not subs:
            subs = sorted(d.rglob(sub_pattern))
            
        if not oofs or not subs:
            continue
            
        runs.append({
            "name": d.name,
            "dir": d,
            "oof_path": oofs[0],
            "sub_path": subs[0]
        })
    
    return runs


def load_and_align(runs, id_col, target):
    """加载和对齐预测结果"""
    base_ids = None
    base_y = None
    oof_merged = None
    sub_merged = None
    cols = []
    
    for r in runs:
        oof = pd.read_csv(r["oof_path"])
        sub = pd.read_csv(r["sub_path"])

        # 处理OOF文件
        if "oof_pred" not in oof.columns:
            cand = [c for c in oof.columns if c not in (id_col, target)]
            assert len(cand) >= 1, f"bad oof file: {r['oof_path']}"
            oof = oof[[id_col, target, cand[-1]]].rename(columns={cand[-1]: "oof_pred"})
        
        # 处理提交文件
        if "isDefault" not in sub.columns and target in sub.columns:
            sub = sub.rename(columns={target: "isDefault"})
        assert "isDefault" in sub.columns, f"bad sub file: {r['sub_path']}"

        if base_ids is None:
            base_ids = oof[id_col].values
            base_y = oof[target].values
            oof_merged = pd.DataFrame({id_col: base_ids, target: base_y})
            sub_merged = pd.DataFrame({id_col: sub[id_col].values})

        oof_merged = oof_merged.merge(oof[[id_col, "oof_pred"]], on=id_col, how="inner")
        sub_merged = sub_merged.merge(sub[[id_col, "isDefault"]], on=id_col, how="inner")

        new_col = r["name"]
        oof_merged.rename(columns={"oof_pred": new_col}, inplace=True)
        sub_merged.rename(columns={"isDefault": new_col}, inplace=True)
        cols.append(new_col)
    
    return oof_merged, sub_merged, cols


def individual_auc(oof_df, cols, target, n_jobs=1):
    """计算单个模型AUC"""
    y = oof_df[target].values
    
    def _auc(c):
        try:
            return roc_auc_score(y, oof_df[c].values)
        except Exception:
            return np.nan
    
    aucs = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_auc)(c) for c in cols)
    return pd.Series(aucs, index=cols).sort_values(ascending=False)


def corr_spearman_fast(oof_df, cols):
    """快速计算Spearman相关性"""
    R = oof_df[cols].rank(method="average").values
    C = np.corrcoef(R, rowvar=False)
    return pd.DataFrame(C, index=cols, columns=cols)


def drop_high_corr(oof_df, cols, aucs, thr=0.985, min_models=3):
    """去除高相关性模型"""
    kept, dropped, used = [], [], set()
    corr = corr_spearman_fast(oof_df, cols)
    
    for a in cols:
        if a in used:
            continue
            
        group = [a]
        for b in cols:
            if b == a or b in used:
                continue
            if corr.loc[a, b] >= thr:
                group.append(b)
        
        best = max(group, key=lambda x: aucs.get(x, 0.0))
        kept.append(best)
        used.update(group)
        
        for g in group:
            if g != best:
                dropped.append(g)
    
    if len(kept) < min_models:
        return cols, []
    
    return kept, dropped


def minmax_normalize(x):
    """MinMax标准化"""
    x = x.astype(float)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def blend_mean(oof_df, sub_df, cols, y):
    """简单平均融合"""
    X = oof_df[cols].values
    T = sub_df[cols].values
    p = X.mean(axis=1)
    t = T.mean(axis=1)
    auc = roc_auc_score(y, p)
    return auc, p, t, {"weights": {c: 1/len(cols) for c in cols}}


def blend_logit_mean(oof_df, sub_df, cols, y):
    """Logit平均融合"""
    X = np.clip(oof_df[cols].values, 1e-6, 1-1e-6)
    T = np.clip(sub_df[cols].values, 1e-6, 1-1e-6)
    logitX = np.log(X/(1-X))
    logitT = np.log(T/(1-T))
    p = 1/(1+np.exp(-logitX.mean(axis=1)))
    t = 1/(1+np.exp(-logitT.mean(axis=1)))
    auc = roc_auc_score(y, p)
    return auc, p, t, {"weights": {c: 1/len(cols) for c in cols}}


def blend_rank_mean(oof_df, sub_df, cols, y):
    """排名平均融合"""
    RX = oof_df[cols].rank(method="average").values
    RT = sub_df[cols].rank(method="average").values
    p = minmax_normalize(RX.mean(axis=1))
    t = minmax_normalize(RT.mean(axis=1))
    auc = roc_auc_score(y, p)
    return auc, p, t, {"weights": {c: 1/len(cols) for c in cols}}


def stack_lr(oof_df, sub_df, cols, y, folds=5, seed=2025, C=1.0, n_jobs=32):
    """逻辑回归Stacking"""
    X = oof_df[cols].values
    T = sub_df[cols].values
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    test = np.zeros(len(T))
    
    for tr, va in skf.split(X, y):
        clf = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=500, n_jobs=n_jobs)
        clf.fit(X[tr], y[tr])
        oof[va] = clf.predict_proba(X[va])[:,1]
        test += clf.predict_proba(T)[:,1]/folds
    
    auc = roc_auc_score(y, oof)
    return auc, oof, test, {"coef_norm": float(np.linalg.norm(clf.coef_))}


def stack_ridge(oof_df, sub_df, cols, y, folds=5, seed=2025, alpha=2.0):
    """Ridge回归Stacking"""
    X = oof_df[cols].values
    T = sub_df[cols].values
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    test = np.zeros(len(T))
    
    for tr, va in skf.split(X, y):
        reg = Ridge(alpha=alpha, random_state=seed)
        reg.fit(X[tr], y[tr])
        oof[va] = reg.predict(X[va])
        test += reg.predict(T)/folds
    
    auc = roc_auc_score(y, oof)
    return auc, oof, test, {"alpha": alpha}


def optimize_weights(X, y, iters=6000, restarts=8, seed=2025, n_jobs=32):
    """权重优化"""
    def objective(w):
        w = w / w.sum()  # 归一化
        pred = X @ w
        return -roc_auc_score(y, pred)
    
    def optimize_single_start():
        rng = np.random.default_rng(seed)
        M = X.shape[1]
        w_init = rng.dirichlet(np.ones(M))
        result = minimize(objective, w_init, method='L-BFGS-B', 
                         bounds=[(0, 1)] * M, options={'maxiter': iters//restarts})
        return result.x, -result.fun
    
    # 并行多起点优化
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(optimize_single_start)() for _ in range(restarts)
    )
    
    # 选择最佳权重
    best_w, best_score = max(results, key=lambda x: x[1])
    best_w = best_w / best_w.sum()
    
    return best_w, best_score


def weight_opt(oof_df, sub_df, cols, y, iters=6000, restarts=8, seed=2025, n_jobs=32):
    """权重优化融合"""
    X = oof_df[cols].values
    T = sub_df[cols].values
    
    w, best_auc = optimize_weights(X, y, iters, restarts, seed, n_jobs)
    p = X @ w
    t = T @ w
    
    return best_auc, p, t, {"weights": {c: float(w[i]) for i, c in enumerate(cols)}}


def save_blend_result(out_dir, name, oof_df, sub_df, y, oof_pred, sub_pred, meta, auc):
    """保存融合结果"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存OOF预测
    oof_df_result = pd.DataFrame({
        oof_df.columns[0]: oof_df.iloc[:, 0].values,
        "oof_pred": oof_pred,
        "isDefault": y
    })
    oof_df_result.to_csv(out_dir / f"oof_{name}.csv", index=False)
    
    # 保存测试集预测
    sub_df_result = pd.DataFrame({
        sub_df.columns[0]: sub_df.iloc[:, 0].values,
        "isDefault": np.clip(sub_pred, 0, 1)
    })
    sub_df_result.to_csv(out_dir / f"submission_{name}.csv", index=False)
    
    # 保存权重
    with open(out_dir / f"weights_{name}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    print(f"[{name}] OOF AUC = {auc:.6f}")


def main():
    args = get_args()
    
    # 设置线程数
    os.environ.setdefault("OMP_NUM_THREADS", str(args.n_jobs))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.n_jobs))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.n_jobs))
    
    # 扫描模型
    runs = scan_runs(args.root_dir, args.oof_pattern, args.sub_pattern)
    if not runs:
        print("[scan] no runs found")
        return
    
    print("[scan] runs:")
    for r in runs:
        print(f"   - {r['name']}")
    
    # 加载数据
    oof_df, sub_df, cols = load_and_align(runs, args.id_col, args.target)
    y = oof_df[args.target].values
    print(f"[merge] OOF rows = {len(oof_df)}, models = {len(cols)}")
    print(f"[merge] SUB rows = {len(sub_df)}")
    
    # 计算单模型AUC
    aucs_series = individual_auc(oof_df, cols, args.target, n_jobs=args.n_jobs)
    print("\n== individual OOF AUC ==")
    for k, v in aucs_series.items():
        print(f"{k:20s} {v:.6f}")
    aucs = aucs_series.to_dict()
    
    # 计算相关性
    corr = corr_spearman_fast(oof_df, cols)
    print("\n== pairwise Spearman corr ==")
    print(corr.round(6))
    
    # 去重
    kept, dropped = drop_high_corr(oof_df, cols, aucs, thr=args.corr_threshold, min_models=args.min_models)
    if kept != cols:
        print("\n[dedup] kept models:")
        for c in kept:
            print("  -", c)
        if dropped:
            print("[dedup] dropped (high corr):")
            for c in dropped:
                print("  -", c)
    else:
        print("\n[dedup] keep all models (no heavy redundancy)")
    
    # 创建输出目录
    out_dir = Path(args.root_dir) / f"blend_{timestamp()}"
    ensure_dir(out_dir)
    
    # 保存元数据
    aucs_series.to_csv(out_dir / "aucs.csv", header=["auc"])
    corr.to_csv(out_dir / "corr_spearman.csv")
    with open(out_dir / "kept_models.txt", "w") as f:
        f.write("\n".join(kept))
    with open(out_dir / "dropped_models.txt", "w") as f:
        f.write("\n".join(dropped))
    
    results = []
    
    # 各种融合策略
    strategies = [
        ("mean", lambda: blend_mean(oof_df, sub_df, kept, y)),
        ("logit_mean", lambda: blend_logit_mean(oof_df, sub_df, kept, y)),
        ("rank_mean", lambda: blend_rank_mean(oof_df, sub_df, kept, y)),
        ("stack_lr", lambda: stack_lr(oof_df, sub_df, kept, y, folds=args.folds, 
                                    seed=args.seed, C=args.logreg_C, n_jobs=args.n_jobs)),
        ("stack_ridge", lambda: stack_ridge(oof_df, sub_df, kept, y, folds=args.folds, 
                                          seed=args.seed, alpha=args.ridge_alpha)),
        ("weight_opt", lambda: weight_opt(oof_df, sub_df, kept, y, 
                                        iters=args.weight_opt_iters, restarts=args.weight_opt_restarts,
                                        seed=args.seed, n_jobs=args.n_jobs))
    ]
    
    for name, strategy_func in strategies:
        try:
            auc, p, t, meta = strategy_func()
            save_blend_result(out_dir, name, oof_df, sub_df, y, p, t, meta, auc)
            results.append((name, auc))
        except Exception as e:
            print(f"[{name}] Failed: {e}")
            results.append((name, 0.0))
    
    # 保存结果摘要
    res_df = pd.DataFrame(results, columns=["strategy", "oof_auc"]).sort_values("oof_auc", ascending=False)
    res_df.to_csv(out_dir / "blend_summary.csv", index=False)
    print(f"\n[done] outputs in: {out_dir}")


if __name__ == "__main__":
    main()
