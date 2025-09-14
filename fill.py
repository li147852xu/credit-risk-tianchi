#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
补齐 CatBoost 训练产物：特征重要性 + 提交文件

示例：
  python fill.py \
    --run_dir output/C0_0.738666 \
    --cache_dir data/processed_v1 \
    --id_col id \
    --target isDefault
"""

import os
import argparse
import numpy as np
import pandas as pd
from glob import glob

try:
    from tqdm.auto import tqdm
except Exception:
    from tqdm import tqdm


# ---------- 小工具 ----------
def read_list(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def ensure_exists(path: str, what: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERR] {what} 不存在：{path}")

def coerce_catboost_cats(df: pd.DataFrame, cat_cols: list):
    """
    将类别列转换为 CatBoost 可接受的底层类型：
      - 若为整数 dtype：保持 int64
      - 其它情况（float/object/category/...）：统一转 string，并把缺失值填为 'nan'
    """
    for c in cat_cols:
        if c not in df.columns:
            continue
        s = df[c]
        if pd.api.types.is_integer_dtype(s):
            df[c] = s.astype("int64")
        else:
            df[c] = s.astype("string").fillna("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True,
                    help="这次 run 的输出目录（含 features.txt / cat_features.txt / models/）")
    ap.add_argument("--models_dir", type=str, default=None,
                    help="模型目录，默认 <run_dir>/models")
    ap.add_argument("--cache_dir", type=str, default="data/processed_v1",
                    help="处理后数据缓存目录（需有 train_fe.csv / test_fe.csv）")
    ap.add_argument("--train_csv", type=str, default=None,
                    help="覆盖默认的训练特征 CSV 路径")
    ap.add_argument("--test_csv", type=str, default=None,
                    help="覆盖默认的测试特征 CSV 路径")
    ap.add_argument("--id_col", type=str, default="id")
    ap.add_argument("--target", type=str, default="isDefault")
    ap.add_argument("--imp_out", type=str, default=None,
                    help="特征重要性输出路径（默认 <run_dir>/feature_importance.csv）")
    ap.add_argument("--sub_out", type=str, default=None,
                    help="提交文件输出路径（默认 <run_dir>/submission.csv）")
    args = ap.parse_args()

    run_dir = args.run_dir.rstrip("/")
    feat_path = os.path.join(run_dir, "features.txt")
    catf_path = os.path.join(run_dir, "cat_features.txt")
    ensure_exists(feat_path, "features.txt")
    ensure_exists(catf_path, "cat_features.txt")

    FEATURES = read_list(feat_path)
    CAT_FEATURES_ALL = read_list(catf_path)

    # 数据
    cache_dir = args.cache_dir.rstrip("/")
    train_csv = args.train_csv or os.path.join(cache_dir, "train_fe.csv")
    test_csv  = args.test_csv  or os.path.join(cache_dir, "test_fe.csv")
    ensure_exists(train_csv, "train_fe.csv")
    ensure_exists(test_csv,  "test_fe.csv")

    print(f"[load] train_fe: {train_csv}")
    print(f"[load] test_fe : {test_csv}")
    train_fe = pd.read_csv(train_csv)
    test_fe  = pd.read_csv(test_csv)

    # 仅使用与 FEATURES 交集的类别列，并确保存在于数据中
    CAT_FEATURES = [c for c in CAT_FEATURES_ALL if (c in FEATURES and c in train_fe.columns)]
    # 转换类别列为 CatBoost 要求的类型
    coerce_catboost_cats(train_fe, CAT_FEATURES)
    coerce_catboost_cats(test_fe,  CAT_FEATURES)

    # （可选）再把这些列标记为 pandas.category（对 CatBoost 非必需）
    for c in CAT_FEATURES:
        train_fe[c] = train_fe[c].astype("category")
        if c in test_fe.columns:
            test_fe[c] = test_fe[c].astype("category")

    # 检查特征完整性
    miss_train = [c for c in FEATURES if c not in train_fe.columns]
    miss_test  = [c for c in FEATURES if c not in test_fe.columns]
    if miss_train:
        raise KeyError(f"[ERR] train_fe 缺少训练时的特征列：{miss_train[:10]} ...")
    if miss_test:
        raise KeyError(f"[ERR] test_fe 缺少训练时的特征列：{miss_test[:10]} ...")

    # 模型清单
    models_dir = args.models_dir or os.path.join(run_dir, "models")
    ensure_exists(models_dir, "models 目录")
    cbm_list  = sorted(glob(os.path.join(models_dir, "cat_fold*.cbm")))
    json_list = sorted(glob(os.path.join(models_dir, "cat_fold*.json")))
    model_paths = cbm_list if cbm_list else json_list
    if not model_paths:
        raise FileNotFoundError(f"[ERR] 在 {models_dir} 未找到 cat_foldXX.cbm/.json")

    print(f"[info] found {len(model_paths)} fold models in {models_dir}")

    # 输出路径
    imp_out = args.imp_out or os.path.join(run_dir, "feature_importance.csv")
    sub_out = args.sub_out or os.path.join(run_dir, "submission.csv")

    # CatBoost 预测与重要性
    from catboost import CatBoostClassifier, Pool

    # 用 FEATURES 的顺序计算 cat 索引
    cat_idx = [i for i, c in enumerate(FEATURES) if c in CAT_FEATURES]

    y = train_fe[args.target].values if args.target in train_fe.columns else None
    full_pool = Pool(train_fe[FEATURES], label=y, cat_features=cat_idx)
    test_pool = Pool(test_fe[FEATURES],  cat_features=cat_idx)

    test_pred = np.zeros(len(test_fe), dtype=np.float64)
    imp_rows = []

    for k, mpath in enumerate(tqdm(model_paths, desc="Folds", dynamic_ncols=False), 1):
        model = CatBoostClassifier()
        try:
            if mpath.endswith(".cbm"):
                model.load_model(mpath)
            else:
                model.load_model(mpath, format="json")
        except Exception as e:
            raise RuntimeError(f"[ERR] 加载模型失败：{mpath}\n{e}")

        # 特征重要性：PredictionValuesChange（若失败退回 LossFunctionChange）
        try:
            scores = model.get_feature_importance(full_pool, type="PredictionValuesChange")
        except Exception:
            scores = model.get_feature_importance(full_pool, type="LossFunctionChange")

        imp_rows.append(pd.DataFrame({
            "feature": FEATURES,
            "score": scores,
            "fold": k
        }))

        # 测试集预测（各折平均）
        test_pred += model.predict_proba(test_pool)[:, 1] / len(model_paths)

    # 保存特征重要性
    fi_all = pd.concat(imp_rows, ignore_index=True)
    fi_agg = (fi_all.groupby("feature", as_index=False)["score"]
              .mean()
              .sort_values("score", ascending=False))
    fi_agg.to_csv(imp_out, index=False)
    print(f"[save] feature importance -> {imp_out}")

    # 保存提交文件
    sub_df = pd.DataFrame({
        args.id_col: test_fe[args.id_col].values,
        "isDefault": np.clip(test_pred, 0, 1)
    })
    sub_df.to_csv(sub_out, index=False)
    print(f"[save] submission -> {sub_out}")

    print("[done] completed.")


if __name__ == "__main__":
    main()