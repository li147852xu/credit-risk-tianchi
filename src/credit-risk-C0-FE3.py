#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CreditRisk Baseline (LightGBM CPU, tqdm) — single-file script (CSV cache)

Usage:
  python creditrisk_baseline.py \
    --train_path data/train.csv \
    --test_path data/testA.csv \
    --target isDefault \
    --id_col id \
    --out_dir outputs
"""

# =========================
# ===== 输入区（CONFIG）=====
# =========================
import os
CONFIG = {
    # 数据与字段
    "train_path": "data/train.csv",
    "test_path": "data/testA.csv",
    "target": "isDefault",
    "id_col": "id",

    # 训练与CV
    "n_folds": 7,                 # 想更快可改 3
    "random_state": 2025,
    "num_boost_round": 8000,
    "early_stopping_rounds": 400,
    "update_every": 10,
    "sample_frac": 1.0,           # <1.0 时做分层抽样（仅在未命中缓存且走特征工程时生效）
    
    # LightGBM 参数
    "lgbm_params": {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.01,
        "num_leaves": 255,
        "max_depth": -1,
        "max_bin": 511,
        "min_data_in_leaf": 50,
        "min_sum_hessian_in_leaf": 1.0,
        "lambda_l1": 1.0,
        "lambda_l2": 5.0,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "max_cat_threshold": 1024,   # 若版本不支持，可删
        "cat_smooth": 10.0,          # 若版本不支持，可删
        "cat_l2": 10.0,              # 若版本不支持，可删
        "force_col_wise": True,
        "bin_construct_sample_cnt": 500000,
        "verbosity": -1,
        "num_threads": os.cpu_count() or 4,
        "deterministic": True,
    },

    # —— XGBoost（CPU）参数 —— #
    "xgb_params":  {
        "booster": "gbtree",
        "objective": "binary:logistic",
        "eval_metric": "auc",
        
        "tree_method": "hist",
        "predictor": "cpu_predictor",
        "enable_categorical": True,
        
        "learning_rate": 0.06,          # 高一些的lr，缩短有效轮数
        "max_depth": 8,                 # 改用 depthwise 浅树
        "grow_policy": "depthwise",     # 和你之前的 lossguide 完全不同
        "max_bin": 256,                 # 较快；兼顾精度
        "min_child_weight": 5.0,        # 强正则，抑制碎切分
        "gamma": 0.1,                   # 分裂阈，进一步稳泛化
        "lambda": 20.0,                 # L2 强一些
        "alpha": 2.0,                   # L1 略开
        
        "subsample": 0.80,              # 行采样更激进
        "colsample_bytree": 0.70,       # 列采样更激进
        "colsample_bylevel": 0.70,
        "colsample_bynode": 0.70,
        
        "max_cat_to_onehot": 128,       # 控制高基数展开成本
        "categorical_smoothing": 10,
        
        "nthread": 128,
        "verbosity": 0
    },

    # 高基数类别阈值（>阈值则不用作category，改用频次数值列）
    "high_card_threshold": 200,

    # 输出
    "out_dir": "outputs-1",
    "save_oof": True,
    "save_importance": True,
    "submission_name": "submission.csv",

    # 日志与模型
    "log_to_file": True,
    "log_filename": "train.log",
    "save_models": True,
    "model_dir": "models",
    "save_meta": True,

    # 处理后数据缓存（CSV）
    "use_cache": True,                 # True 时优先尝试加载缓存
    "cache_dir": "data/processed_v3",  # 缓存目录
}

CONFIG.update({
    "cat_params": {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        # —— 容量与速率 —— #
        "depth": 6,                 # 树深
        "learning_rate": 0.03,
        "l2_leaf_reg": 8.0,
        "iterations": 8000,         # 作为上限，仍用早停
        "early_stopping_rounds": 400,
        # —— 采样与稳定化 —— #
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.5,
        "random_strength": 1.0,
        "leaf_estimation_iterations": 2,
        # —— 连续特征分箱 —— #
        "border_count": 128,
        # —— 线程与日志 —— #
        "thread_count": 128,
        "verbose": False
    },
    # 统一控制：是否保存 CatBoost 的文本模型（只读不可加载）
    "save_cat_text_dump": True
})

# =========================
# ====== 依赖导入 ========
# =========================
import re
import json
import argparse
import sys, io, contextlib
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
# try:
#     from tqdm.notebook import tqdm
# except Exception:
#     from tqdm.auto import tqdm
from tqdm.auto import tqdm
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

# =========================
# ====== Tee 日志工具 =====
# =========================
class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)
    def flush(self):
        for st in self.streams:
            st.flush()

@contextlib.contextmanager
def tee_stdout_stderr(log_path: str, mode: str = "a"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, mode, encoding="utf-8") as f:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n\n===== RUN START {stamp} =====\n")
        f.flush()
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout = _Tee(sys.stdout, f)
            sys.stderr = _Tee(sys.stderr, f)
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err

# =========================
# ====== 工具函数（CL4）====
# =========================
def reduce_mem_usage(df: pd.DataFrame, verbose=True, desc="Downcast"):
    start = df.memory_usage().sum() / 1024**2
    for col in tqdm(list(df.columns), desc=desc, leave=False, dynamic_ncols=True):
        col_type = df[col].dtype
        if str(col_type).startswith(("int", "uint")):
            c_min, c_max = df[col].min(), df[col].max()
            if c_min >= 0:
                if c_max < np.iinfo(np.uint8).max:    df[col] = df[col].astype(np.uint8)
                elif c_max < np.iinfo(np.uint16).max: df[col] = df[col].astype(np.uint16)
                elif c_max < np.iinfo(np.uint32).max: df[col] = df[col].astype(np.uint32)
                else:                                  df[col] = df[col].astype(np.uint64)
            else:
                if np.iinfo(np.int8).min  < c_min < np.iinfo(np.int8).max:    df[col] = df[col].astype(np.int8)
                elif np.iinfo(np.int16).min < c_min < np.iinfo(np.int16).max: df[col] = df[col].astype(np.int16)
                elif np.iinfo(np.int32).min < c_min < np.iinfo(np.int32).max: df[col] = df[col].astype(np.int32)
                else:                                                         df[col] = df[col].astype(np.int64)
        elif str(col_type).startswith("float"):
            df[col] = df[col].astype(np.float32)
    end = df.memory_usage().sum() / 1024**2
    if verbose: print(f"[reduce_mem_usage] {start:.2f} -> {end:.2f} MB")
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

def freq_encode(series: pd.Series):
    vc = series.astype(str).value_counts(dropna=False)
    mapping = vc / vc.sum()
    return series.astype(str).map(mapping).astype("float32")

def to_3digit_postcode_str(series: pd.Series):
    s = series.astype(str).str.split(".").str[0]
    s = s.where(~s.isin(["nan","None","none","NaN"]), other="")
    return s.str.zfill(3)

def add_n_stats(df, n_cols):
    n_df = df[n_cols].apply(pd.to_numeric, errors="coerce")
    df["n_sum"]      = n_df.sum(axis=1).astype("float32")
    df["n_mean"]     = n_df.mean(axis=1).astype("float32")
    df["n_max"]      = n_df.max(axis=1).astype("float32")
    df["n_std"]      = n_df.std(axis=1).astype("float32")
    df["n_nonzero"]  = (n_df > 0).sum(axis=1).astype("float32")
    return df

def safe_cut(series, bins, labels=None):
    try: return pd.cut(series, bins=bins, labels=labels, include_lowest=True)
    except: return pd.Series(pd.Categorical([np.nan]*len(series)))

def catboost_cat_feature_indices(df: pd.DataFrame, feature_cols: list) -> list:
    """返回 CatBoost 需要的类别列索引（基于 feature_cols 的位置）"""
    idx = []
    for i, c in enumerate(feature_cols):
        if str(df[c].dtype) == "category":
            idx.append(i)
    return idx

# =========================
# ====== 数据读取/EDA ======
# =========================
def load_data(train_path, test_path):
    train = pd.read_csv(train_path, low_memory=False)
    test  = pd.read_csv(test_path, low_memory=False)
    print(f"[load] train={train.shape}, test={test.shape}")
    return train, test

def quick_eda(train: pd.DataFrame, target: str):
    print("[EDA] dtypes (head):")
    print(train.dtypes.head(10))
    if target in train.columns:
        pos_rate = train[target].mean()
        print(f"[EDA] target pos_rate={pos_rate:.4f}")

# =========================
# ===== 特征工程（CL5）=====
# =========================
def build_features(train: pd.DataFrame, test: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    train["__is_train__"] = 1
    test["__is_train__"]  = 0
    full = pd.concat([train, test], axis=0, ignore_index=True)
    full = reduce_mem_usage(full, desc="Downcast (before FE)")

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
        full["fico_mean"] = ((full["ficoRangeLow"] + full["ficoRangeHigh"]) / 2.0).astype("float32")

    if {"loanAmnt","annualIncome"}.issubset(full.columns):
        full["loan_income_ratio"] = (full["loanAmnt"] / (full["annualIncome"] + 1)).astype("float32")

    for col in ["employmentTitle","title"]:
        if col in full.columns:
            as_str = full[col].astype(str)
            full[f"{col}_cat"]  = as_str.astype("category")
            full[f"{col}_freq"] = freq_encode(as_str)

    if "postCode" in full.columns:
        pc3 = to_3digit_postcode_str(full["postCode"])
        full["postCode_cat"]  = pc3.astype("category")
        full["postCode_freq"] = freq_encode(pc3)

    low_card_int_as_cat = ["grade","subGrade","homeOwnership","verificationStatus","purpose","regionCode","initialListStatus","applicationType","policyCode"]
    for c in [x for x in low_card_int_as_cat if x in full.columns]:
        if str(full[c].dtype) != "category":
            full[c] = full[c].astype("category")

    n_cols = [f"n{i}" for i in range(15) if f"n{i}" in full.columns]
    if n_cols: full = add_n_stats(full, n_cols)

    num_cols_full = full.select_dtypes(include=["number"]).columns
    full[num_cols_full] = full[num_cols_full].replace([np.inf,-np.inf], np.nan)
    full[num_cols_full] = full[num_cols_full].fillna(full[num_cols_full].median())

    for col in ["issueDate_dt","earliest_dt"]:
        if col in full.columns: full.drop(columns=[col], inplace=True)

    full = reduce_mem_usage(full, desc="Downcast (after FE)")
    return full

# =========================
# ==== 高基数类别降级补丁 ====
# =========================
def downgrade_high_card_categories(full: pd.DataFrame, threshold: int):
    cat_cols_now = [c for c in full.columns if str(full[c].dtype)=="category"]
    card = {c: full[c].nunique(dropna=False) for c in cat_cols_now}
    high_card_cols = [c for c,v in card.items() if v > threshold]
    print(f"[speed] high-card categories: {high_card_cols}")
    for c in high_card_cols:
        freq_col = f"{c}_freq"
        if freq_col not in full.columns:
            full[freq_col] = freq_encode(full[c].astype(str))
        full[c] = full[c].astype(str)  # 防止被 cat_features 捕获
    return full

# =========================
# ====== 训练/推理 ========
# =========================
def select_features(train_fe, test_fe, target, id_col):
    from pandas.api.types import is_numeric_dtype, is_bool_dtype
    features_all = [c for c in train_fe.columns if c not in [target,id_col]]
    features = [c for c in features_all if is_numeric_dtype(train_fe[c]) or is_bool_dtype(train_fe[c]) or str(train_fe[c].dtype)=="category"]
    cat_features = [c for c in features if str(train_fe[c].dtype)=="category"]
    pos = train_fe[target].sum(); neg = len(train_fe)-pos
    spw = float(neg)/float(max(pos,1.0))
    print(f"[select] features={len(features)}, cat={len(cat_features)}, spw={spw:.3f}")
    return features, cat_features, spw

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def train_cv_lgbm_cpu(train_fe,test_fe,features,cat_features,target,params,
                      n_folds=5,num_boost_round=3000,early_stopping_rounds=100,
                      update_every=10,seed=2025,
                      save_models=False,model_save_dir=None,save_meta=True):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(train_fe)); test_pred = np.zeros(len(test_fe))
    importances=[]; fold_reports=[]
    if save_models and model_save_dir: ensure_dir(model_save_dir)

    fold_iter = tqdm(enumerate(skf.split(train_fe[features],train_fe[target]),1),
                     total=n_folds,desc="Folds",leave=False,dynamic_ncols=True)

    for fold,(tr_idx,va_idx) in fold_iter:
        X_tr,y_tr = train_fe.iloc[tr_idx][features], train_fe.iloc[tr_idx][target]
        X_va,y_va = train_fe.iloc[va_idx][features], train_fe.iloc[va_idx][target]
        dtrain=lgb.Dataset(X_tr,label=y_tr,categorical_feature=cat_features,free_raw_data=True)
        dvalid=lgb.Dataset(X_va,label=y_va,categorical_feature=cat_features,free_raw_data=True)

        pbar=tqdm(total=num_boost_round,desc=f"Fold {fold}",leave=False,dynamic_ncols=True)
        it={"i":0}
        def cb(env): it["i"]+=1; 
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds,verbose=False),
                   lgb.log_evaluation(period=0),cb]
        model=lgb.train(params,train_set=dtrain,num_boost_round=num_boost_round,
                        valid_sets=[dtrain,dvalid],valid_names=["train","valid"],
                        callbacks=callbacks)
        pbar.close()
        best_it = model.best_iteration or model.current_iteration()
        oof[va_idx]=model.predict(X_va,num_iteration=best_it)
        fold_auc=roc_auc_score(y_va,oof[va_idx])
        print(f"[Fold {fold}] AUC={fold_auc:.6f}")

        if save_models and model_save_dir:
            path=os.path.join(model_save_dir,f"lgbm_fold{fold:02d}.txt")
            model.save_model(path,num_iteration=best_it)
            print(f"[Fold {fold}] model saved -> {path}")

        imp=pd.DataFrame({"feature":model.feature_name(),
                          "gain":model.feature_importance("gain"),
                          "split":model.feature_importance("split"),
                          "fold":fold})
        importances.append(imp)
        fold_reports.append({"fold":fold,"auc":float(fold_auc),"best_iteration":int(best_it) if best_it is not None else None})
        test_pred+=model.predict(test_fe[features],num_iteration=best_it)/n_folds

    cv_auc=roc_auc_score(train_fe[target],oof)
    print(f"[CV] AUC={cv_auc:.6f}")
    if save_meta and model_save_dir:
        meta={"cv_auc":float(cv_auc),"folds":fold_reports,"params":params,
              "features_count": len(features), "cat_features_count": len(cat_features)}
        with open(os.path.join(model_save_dir,"meta.json"),"w",encoding="utf-8") as f: json.dump(meta,f,indent=2)
        print(f"[save] meta -> {os.path.join(model_save_dir,'meta.json')}")
    return {"oof":oof,"test_pred":test_pred,"importances":importances,"cv_auc":cv_auc}

def train_cv_xgb_cpu(train_fe, test_fe, features, cat_features, target, params,
                     n_folds=5, num_boost_round=10000, early_stopping_rounds=200,
                     update_every=10, seed=2025,
                     save_models=False, model_save_dir=None, save_meta=True,
                     fast_mode=False):
    """
    与 train_cv_lgbm_cpu 对齐的 XGBoost 版本（CPU）。
    - 支持 pandas category（params.enable_categorical=True）
    - tqdm 进度条在 fast_mode=False 时开启
    - 每折保存为 JSON 模型：models/xgb_foldXX.json
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(train_fe), dtype=np.float64)
    test_pred = np.zeros(len(test_fe), dtype=np.float64)
    importances = []
    fold_reports = []
    if save_models and model_save_dir:
        ensure_dir(model_save_dir)

    # 目标与特征
    y_all = train_fe[target].astype(float).values
    X_all = train_fe[features]
    X_test = test_fe[features]

    # DMatrix 构建：开启类别支持
    dm_test = xgb.DMatrix(X_test, enable_categorical=True, nthread=params.get("nthread"))

    fold_iter = tqdm(enumerate(skf.split(X_all, y_all), 1),
                     total=n_folds, desc="Folds", leave=False, dynamic_ncols=False)

    for fold, (tr_idx, va_idx) in fold_iter:
        X_tr, y_tr = X_all.iloc[tr_idx], y_all[tr_idx]
        X_va, y_va = X_all.iloc[va_idx], y_all[va_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True, nthread=params.get("nthread"))
        dvalid = xgb.DMatrix(X_va, label=y_va, enable_categorical=True, nthread=params.get("nthread"))
        watchlist = [(dtrain, "train"), (dvalid, "valid")]

        # 进度条回调（fast_mode 关闭则启用）
        callbacks = [xgb.callback.EarlyStopping(rounds=early_stopping_rounds, save_best=True, maximize=True)]
        if not fast_mode:
            class TqdmCallback(xgb.callback.TrainingCallback):
                def __init__(self, total, update_every=10):
                    self.total = total
                    self.update_every = update_every
                    self.pbar = tqdm(total=total, desc=f"Fold {fold}", leave=False, dynamic_ncols=False,
                                     miniters=update_every, mininterval=1.0)
                def after_iteration(self, model, epoch, evals_log):
                    if (epoch + 1) % self.update_every == 0:
                        self.pbar.update(self.update_every)
                    return False
                def after_training(self, model):
                    # 补齐
                    curr = model.best_iteration if hasattr(model, "best_iteration") else self.total
                    if curr and curr < self.total:
                        self.pbar.update(self.total - curr)
                    self.pbar.close()
                    return model
            callbacks.append(TqdmCallback(total=num_boost_round, update_every=update_every))

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=watchlist,
            callbacks=callbacks,
            verbose_eval=False
        )

        # best_iteration / 预测
        best_it = getattr(booster, "best_iteration", None)
        if best_it is None:
            # 未触发早停时，xgboost 可能没有 best_iteration
            best_it = num_boost_round - 1
        # 使用 iteration_range 进行预测（包含上界）
        oof_pred = booster.predict(dvalid, iteration_range=(0, best_it + 1))
        oof[va_idx] = oof_pred
        fold_auc = roc_auc_score(y_va, oof_pred)
        print(f"[Fold {fold}] AUC={fold_auc:.6f}  (best_it={best_it})")

        # 测试集预测
        test_pred += booster.predict(dm_test, iteration_range=(0, best_it + 1)) / n_folds

        # 保存模型
        if save_models and model_save_dir:
            path = os.path.join(model_save_dir, f"xgb_fold{fold:02d}.json")
            booster.save_model(path)
            print(f"[Fold {fold}] model saved -> {path}")

        # 特征重要性（gain / weight）
        fscore_gain = booster.get_score(importance_type="gain")
        fscore_weight = booster.get_score(importance_type="weight")
        imp = pd.DataFrame({
            "feature": list(fscore_gain.keys() | fscore_weight.keys()),
        })
        imp["gain"] = imp["feature"].map(fscore_gain).fillna(0.0)
        imp["weight"] = imp["feature"].map(fscore_weight).fillna(0.0)
        imp["fold"] = fold
        importances.append(imp)

        fold_reports.append({"fold": fold, "auc": float(fold_auc), "best_iteration": int(best_it)})

    cv_auc = roc_auc_score(y_all, oof)
    print(f"[CV] AUC={cv_auc:.6f}")

    # 保存 meta
    if save_meta and model_save_dir:
        meta = {"cv_auc": float(cv_auc), "folds": fold_reports, "params": params}
        with open(os.path.join(model_save_dir, "xgb_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[save] meta -> {os.path.join(model_save_dir, 'xgb_meta.json')}")

    return {"oof": oof, "test_pred": test_pred, "importances": importances, "cv_auc": cv_auc}

def train_cv_catboost(train_fe, test_fe, features, cat_features, target,
                      params: dict, n_folds=5, seed=2025,
                      save_models=True, model_save_dir=None, save_meta=True,
                      fast_mode=True):
    """
    CatBoost K折训练（CPU）
    - 输入：train_fe/test_fe（含 category dtype）、features（列名列表）、cat_features（列名列表）
    - 输出：oof、test_pred、importances（特征重要性DataFrame列表）、cv_auc
    - 保存：每折 .cbm 可加载；可选 .txt 文本树（仅便于审计）
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    ensure_dir(model_save_dir or "models")

    # 类别列索引（基于 features 的位置）
    cat_idx = catboost_cat_feature_indices(train_fe, features)

    y = train_fe[target].values.astype(float)
    X_all = train_fe[features]
    X_test = test_fe[features]

    # 类别不平衡：与 LGB/XGB 保持一致（neg/pos）
    pos = y.sum(); neg = len(y) - pos
    scale_pos_weight = float(neg) / float(max(pos, 1.0))

    # 两种方式选其一：AutoClassWeights 或 class_weights
    # 方式A（简单）：自动权重（v1.2+）——和 scale_pos_weight 含义相近
    use_auto_class_weights = True

    params = dict(params)  # 复制，避免外部引用被修改
    if use_auto_class_weights:
        params["auto_class_weights"] = "Balanced"
    else:
        params["class_weights"] = [1.0, scale_pos_weight]

    # 线程数兜底
    params["thread_count"] = params.get("thread_count", os.cpu_count() or 4)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(train_fe), dtype=np.float64)
    test_pred = np.zeros(len(test_fe), dtype=np.float64)
    importances = []
    fold_reports = []

    fold_iter = tqdm(enumerate(skf.split(X_all, y), 1),
                     total=n_folds, desc="Folds(CatBoost)", leave=False, dynamic_ncols=False)

    for fold, (tr_idx, va_idx) in fold_iter:
        X_tr, y_tr = X_all.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X_all.iloc[va_idx], y[va_idx]

        train_pool = Pool(X_tr, label=y_tr, cat_features=cat_idx)
        valid_pool = Pool(X_va, label=y_va, cat_features=cat_idx)
        test_pool  = Pool(X_test, cat_features=cat_idx)

        model = CatBoostClassifier(**params)
        # 日志控制：fast_mode=True 时每折静默；否则每200 iter 打印一条
        verbose = False if fast_mode else 200

        model.fit(
            train_pool,
            eval_set=valid_pool,
            verbose=verbose,
            use_best_model=True
        )

        # 预测（使用 best_iteration）
        oof_pred = model.predict_proba(valid_pool)[:, 1]
        oof[va_idx] = oof_pred
        fold_auc = roc_auc_score(y_va, oof_pred)
        print(f"[Fold {fold}] AUC={fold_auc:.6f}  best_it={model.tree_count_}")

        test_pred += model.predict_proba(test_pool)[:, 1] / n_folds

        # 保存模型
        if save_models and model_save_dir:
            cbm_path = os.path.join(model_save_dir, f"cat_fold{fold:02d}.cbm")
            model.save_model(cbm_path)  # ✅ 可加载
        
        if CONFIG.get("save_cat_text_dump", True):
            json_path = os.path.join(model_save_dir, f"cat_fold{fold:02d}.json")
            model.save_model(json_path, format="json")  # ✅ 文本可读，通常也可加载
            print(f"[Fold {fold}] model saved -> {cbm_path} (loadable), {json_path} (readable)")
        else:
            print(f"[Fold {fold}] model saved -> {cbm_path} (loadable)")

        # 特征重要性（PredictionValuesChange 更稳）
        imp = pd.DataFrame({
            "feature": features,
            "importance": model.get_feature_importance(train_pool, type="PredictionValuesChange")
        })
        imp["fold"] = fold
        importances.append(imp)

        fold_reports.append({"fold": fold, "auc": float(fold_auc), "best_iteration": int(model.tree_count_)})

    cv_auc = roc_auc_score(y, oof)
    print(f"[CV] AUC={cv_auc:.6f}")

    if save_meta and model_save_dir:
        meta = {"cv_auc": float(cv_auc), "folds": fold_reports, "params": params}
        with open(os.path.join(model_save_dir, "cat_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    return {"oof": oof, "test_pred": test_pred, "importances": importances, "cv_auc": cv_auc}

# =========================
# ==== 写/读处理后数据缓存（CSV）====
# =========================
import time

def save_processed_cache_csv(cfg: dict,
                             train_fe: pd.DataFrame,
                             test_fe: pd.DataFrame,
                             features: list,
                             cat_features: list):
    """
    将处理后的特征数据保存为 CSV，并写 meta.json 记录 features / cat_features。
    """
    cache_dir = Path(cfg.get("cache_dir", "data/processed_v1"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_csv = cache_dir / "train_fe.csv"
    test_csv  = cache_dir / "test_fe.csv"
    meta_js   = cache_dir / "meta.json"

    train_fe.to_csv(train_csv, index=False)
    test_fe.to_csv(test_csv, index=False)

    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "id_col": cfg["id_col"],
        "target": cfg["target"],
        "features": features,
        "cat_features": cat_features,
        "n_train": int(len(train_fe)),
        "n_test": int(len(test_fe)),
        "pandas": pd.__version__,
        "lightgbm": lgb.__version__,
        "format": "csv"
    }
    with open(meta_js, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[cache] saved (csv) -> {train_csv}, {test_csv}, {meta_js}")

def try_load_processed_cache(cfg: dict):
    """
    尝试从 cfg['cache_dir'] 加载 CSV 缓存：
      - train_fe.csv
      - test_fe.csv
      - meta.json
    成功则返回 dict(ok=True, train_fe=..., test_fe=..., features=[...], cat_features=[...])
    失败则返回 dict(ok=False)
    """
    cache_dir = Path(cfg.get("cache_dir", "data/processed_v1"))
    train_csv = cache_dir / "train_fe.csv"
    test_csv  = cache_dir / "test_fe.csv"
    meta_js   = cache_dir / "meta.json"

    if not (train_csv.exists() and test_csv.exists() and meta_js.exists()):
        print(f"[cache] not found at {cache_dir} — will build features as usual.")
        return {"ok": False}

    try:
        with open(meta_js, "r", encoding="utf-8") as f:
            meta = json.load(f)
        features = meta.get("features", [])
        cat_features = meta.get("cat_features", [])
        target = meta.get("target", cfg["target"])

        # 用 dtype=str 读入，随后按 meta 恢复
        train_fe = pd.read_csv(train_csv, dtype=str, low_memory=False)
        test_fe  = pd.read_csv(test_csv,  dtype=str, low_memory=False)

        # 恢复目标列与数值列
        if target in train_fe.columns:
            train_fe[target] = pd.to_numeric(train_fe[target], errors="coerce")

        non_cat = [c for c in features if c not in cat_features]
        for c in non_cat:
            if c in train_fe.columns:
                train_fe[c] = pd.to_numeric(train_fe[c], errors="coerce")
            if c in test_fe.columns:
                test_fe[c] = pd.to_numeric(test_fe[c], errors="coerce")

        # 恢复类别列
        for c in cat_features:
            if c in train_fe.columns:
                train_fe[c] = train_fe[c].astype("category")
            if c in test_fe.columns:
                test_fe[c] = test_fe[c].astype("category")

        print(f"[cache] loaded (csv) from {cache_dir}")
        print(f"[cache] train_fe={train_fe.shape}, test_fe={test_fe.shape}, #features={len(features)}, #cat={len(cat_features)}")

        return {"ok": True, "train_fe": train_fe, "test_fe": test_fe,
                "features": features, "cat_features": cat_features}
    except Exception as e:
        print(f"[cache] failed to load csv cache due to: {e} — will rebuild.")
        return {"ok": False}

# =========================
# ====== I/O & 主流程 ======
# =========================
def save_submission(ids: pd.Series, preds: np.ndarray, out_path: str):
    sub=pd.DataFrame({"id":ids.values,"isDefault":np.clip(preds,0,1)})
    sub.to_csv(out_path,index=False); print(f"[save] submission -> {out_path}")

def main(cfg: dict):
    ensure_dir(cfg["out_dir"])

    # 1) 读取原始 CSV
    train, test = load_data(cfg["train_path"], cfg["test_path"])
    quick_eda(train, cfg["target"])

    # 2) 尝试加载缓存
    cache_hit = False
    train_fe = test_fe = None
    features = cat_features = None
    if cfg.get("use_cache", True):
        cache = try_load_processed_cache(cfg)
        if cache.get("ok", False):
            cache_hit = True
            train_fe = cache["train_fe"]
            test_fe  = cache["test_fe"]
            features = cache["features"]
            cat_features = cache["cat_features"]

    # 3) 未命中缓存：可选抽样 -> 特征工程 -> 选择特征 -> 保存缓存
    if not cache_hit:
        if 0.0 < cfg["sample_frac"] < 1.0:
            train, _ = train_test_split(
                train, train_size=cfg["sample_frac"],
                stratify=train[cfg["target"]] if cfg["target"] in train.columns else None,
                random_state=cfg["random_state"]
            )
            print(f"[sample] train -> {train.shape} (frac={cfg['sample_frac']})")

        full = build_features(train.copy(), test.copy(), cfg)
        full = downgrade_high_card_categories(full, cfg["high_card_threshold"])

        train_fe = full[full["__is_train__"] == 1].drop(columns=["__is_train__"])
        test_fe  = full[full["__is_train__"] == 0].drop(columns=["__is_train__", cfg["target"]], errors="ignore")

        features, cat_features, spw = select_features(train_fe, test_fe, cfg["target"], cfg["id_col"])

        # 保存 CSV 缓存
        if cfg.get("use_cache", True):
            save_processed_cache_csv(cfg, train_fe, test_fe, features, cat_features)
    else:
        # 缓存命中：仍需计算 scale_pos_weight
        pos = train_fe[cfg["target"]].sum()
        neg = len(train_fe) - pos
        spw = float(neg) / float(max(pos, 1.0))

    # 4) 训练与落盘
    params = dict(cfg["lgbm_params"])
    params["seed"] = cfg["random_state"]
    params["scale_pos_weight"] = spw

    model_dir = os.path.join(cfg["out_dir"], cfg["model_dir"])
    ensure_dir(model_dir)

    with open(os.path.join(cfg["out_dir"], "features.txt"), "w") as f:
        f.write("\n".join(features))
    with open(os.path.join(cfg["out_dir"], "cat_features.txt"), "w") as f:
        f.write("\n".join(cat_features))
    print(f"[save] features list -> {os.path.join(cfg['out_dir'],'features.txt')}")
    print(f"[save] categorical list -> {os.path.join(cfg['out_dir'],'cat_features.txt')}")

    # out = train_cv_lgbm_cpu(
    #     train_fe, test_fe, features, cat_features, cfg["target"], params,
    #     n_folds=cfg["n_folds"], num_boost_round=cfg["num_boost_round"],
    #     early_stopping_rounds=cfg["early_stopping_rounds"],
    #     update_every=cfg["update_every"], seed=cfg["random_state"],
    #     save_models=cfg["save_models"], model_save_dir=model_dir, save_meta=cfg["save_meta"]
    # )

    # # 使用 XGBoost 训练
    # xgb_params = dict(cfg["xgb_params"])
    # xgb_params["seed"] = cfg["random_state"]
    # xgb_params["nthread"] = 32
    
    # model_dir = os.path.join(cfg["out_dir"], cfg["model_dir"])
    # ensure_dir(model_dir)
    
    # out = train_cv_xgb_cpu(
    #     train_fe, test_fe, features, cat_features, cfg["target"], xgb_params,
    #     n_folds=cfg["n_folds"], num_boost_round=cfg["num_boost_round"],
    #     early_stopping_rounds=cfg["early_stopping_rounds"], update_every=cfg["update_every"],
    #     seed=cfg["random_state"], save_models=cfg["save_models"],
    #     model_save_dir=model_dir, save_meta=cfg["save_meta"],
    #     fast_mode=cfg.get("fast_mode", False)
    # )

    # ====== CatBoost 训练 ======
    cat_params = dict(CONFIG["cat_params"])
    # 统一早停：以 CONFIG 的 early_stopping_rounds/num_boost_round 为上限覆盖
    if CONFIG.get("early_stopping_rounds") is not None:
        cat_params["early_stopping_rounds"] = CONFIG["early_stopping_rounds"]
    if CONFIG.get("num_boost_round") is not None:
        cat_params["iterations"] = CONFIG["num_boost_round"]
    # 线程透传
    cat_params["thread_count"] = cat_params.get("thread_count", os.cpu_count() or 4)

    model_dir = os.path.join(CONFIG["out_dir"], CONFIG["model_dir"])
    ensure_dir(model_dir)

    # 将 features 中的 category dtype 留给 CatBoost；数值列直接用
    out = train_cv_catboost(
        train_fe=train_fe, test_fe=test_fe,
        features=features, cat_features=cat_features,
        target=CONFIG["target"],
        params=cat_params,
        n_folds=CONFIG["n_folds"],
        seed=CONFIG["random_state"],
        save_models=CONFIG["save_models"],
        model_save_dir=model_dir,
        save_meta=CONFIG["save_meta"],
        fast_mode=CONFIG.get("fast_mode", True)
    )
    
    if cfg["save_oof"]:
        oof_df = train_fe[[cfg["id_col"], cfg["target"]]].copy()
        oof_df["oof_pred"] = out["oof"]
        oof_df.to_csv(os.path.join(cfg["out_dir"], "oof_lgb.csv"), index=False)
        print("[save] oof ->", os.path.join(cfg["out_dir"], "oof_lgb.csv"))

    if cfg["save_importance"]:
        # —— 通用特征重要性聚合：兼容 LGB/XGB/CatBoost —— #
        imp_list = out.get("importances", [])
        if imp_list:
            fi_all = pd.concat(imp_list, ignore_index=True)
        
            # 选择可用的分数列
            if "gain" in fi_all.columns:
                score_col = "gain"          # LGB / XGB(gain)
            elif "importance" in fi_all.columns:
                score_col = "importance"    # CatBoost(PredictionValuesChange)
            elif "weight" in fi_all.columns:
                score_col = "weight"        # XGB(split count)
            else:
                # 兜底：选一个数值列
                num_cols = [c for c in fi_all.columns if pd.api.types.is_numeric_dtype(fi_all[c])]
                score_col = [c for c in num_cols if c != "fold"][0] if num_cols else None
        
            if score_col is not None:
                fi_agg = (fi_all.groupby("feature", as_index=False)[score_col]
                          .mean()
                          .sort_values(score_col, ascending=False))
                fi_agg.rename(columns={score_col: "score"}, inplace=True)
                out_path = os.path.join(cfg["out_dir"], "feature_importance.csv")
                fi_agg.to_csv(out_path, index=False)
                print(f"[save] feature importance -> {out_path}")
            else:
                print("[warn] no numeric importance column found; skip saving feature importance.")
        else:
            print("[warn] importances list is empty; skip saving feature importance.")

    save_submission(test_fe[cfg["id_col"]], out["test_pred"], os.path.join(cfg["out_dir"], cfg["submission_name"]))
    print("[done] finished.")

def parse_args_to_config(cfg: dict) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default=cfg["train_path"])
    parser.add_argument("--test_path", type=str, default=cfg["test_path"])
    parser.add_argument("--target", type=str, default=cfg["target"])
    parser.add_argument("--id_col", type=str, default=cfg["id_col"])
    parser.add_argument("--out_dir", type=str, default=cfg["out_dir"])
    parser.add_argument("--n_folds", type=int, default=cfg["n_folds"])
    parser.add_argument("--random_state", type=int, default=cfg["random_state"])
    parser.add_argument("--num_boost_round", type=int, default=cfg["num_boost_round"])
    parser.add_argument("--early_stopping_rounds", type=int, default=cfg["early_stopping_rounds"])
    parser.add_argument("--update_every", type=int, default=cfg["update_every"])
    parser.add_argument("--sample_frac", type=float, default=cfg["sample_frac"])
    parser.add_argument("--use_cache", type=int, default=int(cfg["use_cache"]))
    parser.add_argument("--cache_dir", type=str, default=cfg["cache_dir"])

    args = parser.parse_args()

    cfg["train_path"] = args.train_path
    cfg["test_path"] = args.test_path
    cfg["target"] = args.target
    cfg["id_col"] = args.id_col
    cfg["out_dir"] = args.out_dir
    cfg["n_folds"] = args.n_folds
    cfg["random_state"] = args.random_state
    cfg["num_boost_round"] = args.num_boost_round
    cfg["early_stopping_rounds"] = args.early_stopping_rounds
    cfg["update_every"] = args.update_every
    cfg["sample_frac"] = args.sample_frac
    cfg["use_cache"] = bool(args.use_cache)
    cfg["cache_dir"] = args.cache_dir
    return cfg

if __name__ == "__main__":
    CONFIG = parse_args_to_config(CONFIG)
    ensure_dir(CONFIG["out_dir"])
    log_path = os.path.join(CONFIG["out_dir"], CONFIG["log_filename"])
    if CONFIG.get("log_to_file", True):
        with tee_stdout_stderr(log_path):
            print("[config]", json.dumps(CONFIG, indent=2, default=str))
            main(CONFIG)
    else:
        print("[config]", json.dumps(CONFIG, indent=2, default=str))
        main(CONFIG)