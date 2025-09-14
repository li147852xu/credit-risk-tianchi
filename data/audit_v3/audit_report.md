# FE v3 审计报告

## 基本信息
- train_fe: **(400000, 110)** | 数值列: 87 | 类别列: 12 | 目标正例率: 0.199680

## 缺失/无穷/类别基数
**缺失率 Top10（数值列）**

```
               col  nan_ratio
          loanAmnt        0.0
              term        0.0
      interestRate        0.0
       installment        0.0
   employmentTitle        0.0
      annualIncome        0.0
         isDefault        0.0
          postCode        0.0
               dti        0.0
delinquency_2years        0.0
```

**无穷值 Top10（数值列）**

```
               col  n_inf
          loanAmnt      0
              term      0
      interestRate      0
       installment      0
   employmentTitle      0
      annualIncome      0
         isDefault      0
          postCode      0
               dti      0
delinquency_2years      0
```

**类别基数 Top10**

```
             cat_col  nunique
          regionCode       51
            subGrade       35
            fico_bin       16
             purpose       14
employmentLength_cat       12
               grade        7
       homeOwnership        6
  verificationStatus        3
     applicationType        2
   initialListStatus        2
```

## 时间感知目标编码（te_*）无泄漏检测
                 te_col      mae     rmse  check_months  rows_compared
          te_grade_term 0.011814 0.012637            36         154011
te_employmentLength_cat 0.012303 0.013175            36         154011
            te_subGrade 0.012390 0.014277            36         154011
        te_purpose_home 0.013320 0.018019            36         154011
            te_fico_dti 0.022323 0.032440            36         154011
           te_iy_region 0.023236 0.032326            36         154011
        te_postCode_cat 0.029701 0.043340            36         154011

判定：若 `mae/rmse` 接近 0（1e-6 量级），说明实现与“仅用过去”一致；若明显偏大，可能存在实现偏差。

**疑似异常列（MAE>1e-6）Top5**

```
         te_col      mae     rmse  check_months  rows_compared
te_postCode_cat 0.029701 0.043340            36         154011
   te_iy_region 0.023236 0.032326            36         154011
    te_fico_dti 0.022323 0.032440            36         154011
te_purpose_home 0.013320 0.018019            36         154011
    te_subGrade 0.012390 0.014277            36         154011
```

## 历史滚动违约率（hist_rate_*）核验
              hist_col      key_col      mae     rmse  rows_compared
     hist_rate_purpose      purpose 0.029593 0.032644         154011
  hist_rate_regionCode   regionCode 0.031094 0.034510         154011
       hist_rate_grade        grade 0.035973 0.043437         154011
hist_rate_postCode_cat postCode_cat 0.047275 0.066952         154011

**疑似异常列（MAE>1e-6）Top5**

```
              hist_col      key_col      mae     rmse  rows_compared
hist_rate_postCode_cat postCode_cat 0.047275 0.066952         154011
       hist_rate_grade        grade 0.035973 0.043437         154011
  hist_rate_regionCode   regionCode 0.031094 0.034510         154011
     hist_rate_purpose      purpose 0.029593 0.032644         154011
```

## 按月 AUC（抽样列）
 time_key             col      auc     n
   201601 hist_rate_grade 0.721395  8316
   201602 hist_rate_grade 0.702745 10254
   201603 hist_rate_grade 0.699161 14575
   201604 hist_rate_grade 0.669703  7093
   201605 hist_rate_grade 0.679565  5261
   201606 hist_rate_grade 0.665689  6098
   201607 hist_rate_grade 0.660609  6453
   201608 hist_rate_grade 0.656105  6697
   201609 hist_rate_grade 0.674304  5029
   201610 hist_rate_grade 0.664945  5582
   201611 hist_rate_grade 0.646523  5713
   201612 hist_rate_grade 0.664834  5890
   201701 hist_rate_grade 0.670577  4876
   201702 hist_rate_grade 0.673538  4060
   201703 hist_rate_grade 0.667067  4988
   201704 hist_rate_grade 0.678726  3881
   201705 hist_rate_grade 0.656642  4871
   201706 hist_rate_grade 0.662620  4553
   201707 hist_rate_grade 0.668262  4366
   201708 hist_rate_grade 0.661268  4610

提示：关注某些月份 AUC 明显异常（过高/过低），可能存在时序分布差异或泄漏风险。