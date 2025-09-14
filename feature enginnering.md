太好了，你把字段+当前 dtypes都给了。我按“从原始特征→数据意义→常见取值/格式→建议的特征工程与衍生项（保持信息尽量全）”逐个讲，并在最后给一份特征映射清单方便你在 Notebook 里对照实现。



------





# 逐列说明（含工程化建议）







### 1) 

### id

### （int64）





- 含义：样本唯一标识。
- 格式：整数。
- 工程化：不作特征；仅用于对齐/排序/提交。





------





### 2) 

### loanAmnt

### （float64）





- 含义：贷款金额。

- 工程化

  

  - 原值保留。
  - 衍生：log_loanAmnt = log1p(loanAmnt)（稳定长尾）、loan_income_ratio = loanAmnt / (annualIncome+1)。
  - 分桶：按分位点/金额区间做 5–10 桶（可与 grade、term 交互）。

  

> [!WARNING]
>
> 直接保留了，应该暂时没做分桶？



------





### 3) 

### term

### （int64）





- 含义：贷款期限；你的 dtypes 是整数，多数数据即“月数”。（若有早期版本是“年”，可规则：≤10 视为年×12）

- 工程化

  

  - 原值保留（数值）。
  - 衍生：term_bin（短/中/长：<=36, 37–60, >60）、与 grade/subGrade 的交互（grade×term_bin）。
  - 单调约束（可选）：期限越长风险可能↑，但不必强加。

  

> [!WARNING]
>
> 这里不是月，应该改成年year，对应特征工程可能有问题



------





### 4) 

### interestRate

### （float64）





- 含义：年化利率（%）。

- 工程化

  

  - 原值保留。
  - 衍生：rate_to_grade_gap（把 grade/subGrade映射到平均利率后，计算样本相对偏差）。
  - 可能与 fico_mean、dti 交互（高利率 × 高 dti）。

  





------





### 5) 

### installment

### （float64）





- 含义：每期还款额。

- 工程化

  

  - 原值保留；log_installment。
  - 衍生：installment_income_ratio = installment / (annualIncome/12 + 1e-6)（月供/月收入）。
  - 与 loanAmnt/term 有共线性，模型会自行处理；也可构造 installment * term 近似总还款额。

  





------





### 6) 

### grade

### （object）





- 含义：贷款等级（A–G）。强序类别。

- 工程化

  

  - 转 category（保留原始序）。
  - 衍生：grade_ord（A=1..G=7）/分桶（高风险等级组）。
  - 交互：grade × term_bin、grade × purpose。

  

> [!WARNING]
>
> 注意这里的所有object没有做one-hot，而是转换为category输入模型



------





### 7) 

### subGrade

### （object）





- 含义：更细的等级（如 A1–G5）。强序类别。

- 工程化

  

  - 转 category。
  - 衍生：subGrade_ord（A1=1..G5=35）；subGrade_group（合并为粗级别 7 桶）。
  - 交互：subGrade × purpose / term_bin。

  

> [!WARNING]
>
> 这里的subgrade和grade应该有序列关系，可能需要subGrade_ord来组合起来



------





### 8) 

### employmentTitle

### （float64）





- 含义：就业职称。你这里是 float64，显然已做过某种编码（可能是 ID/频率）。

- 工程化

  

  - 若可回溯原文本更好；当前保留数值列。
  - 建议再补一个频数编码（employmentTitle_freq，在全量 train+test 上统计）来稳健化。
  - 可加一个目标编码（CV target encoding、防泄露），在后续提升阶段用。

  

> [!WARNING]
>
> 这里应该是竞赛方做了脱敏处理，编码为数据



------





### 9) 

### employmentLength

### （object）





- 含义：工作年限（如 10+ years/< 1 year/3 years/n/a）。

- 工程化

  

  - 解析为数值年：emp_len_year ∈ [0,10]。
  - 同时保留原字符串为 category：employmentLength_cat（如 <1y, 1y, 2y, …, 10+y, na），以保留“形态信息”。

  

> [!WARNING]
>
> 原数据是字符串，而且格式比较混乱，需要解析，特征工程是否完善有待确认



------





### 10) 

### homeOwnership

### （int64）





- 含义：房产拥有情况（通常类别：RENT/MORTGAGE/OWN/OTHER）。你这里是整数编码。

- 工程化

  

  - 转 category。
  - 可做粗集合并（如 OWN/MORTGAGE/RENT/OTHER 4 桶）。

  





------





### 11) 

### annualIncome

### （float64）





- 含义：年收入。

- 工程化

  

  - 原值 + log_annualIncome。
  - 比率：见上（与 loanAmnt、installment）。
  - 异常值/截断：对极大值 winsorize（如 99.5% 分位）。

  





------





### 12) 

### verificationStatus

### （int64）





- 含义：收入/信息是否验证（类别）。

- 工程化

  

  - 转 category。
  - 与 annualIncome、dti 交互（未验证+高 dti 可能更差）。

  





------





### 13) 

### issueDate

### （object）





- 含义：放款月份（如 2012/7/1 或 2012-07 等）。

- 工程化

  

  - 解析为日期列 issueDate_dt。
  - 衍生：issue_year、issue_month、issue_quarter、issue_ym = year*100+month。
  - 保留原字符串为 category（issueDate_cat）以防解析失败/保留样式信息。
  - 时间漂移：可加“时间窗风险均值”（rolling mean target encoding，CV 防泄露）。

  





------





### 14) 

### isDefault

### （int64）





- 目标：1=违约，0=正常。





------





### 15) 

### purpose

### （int64）





- 含义：贷款用途类别（整数编码）。

- 工程化

  

  - 转 category。
  - 与 subGrade/term_bin/loanAmnt_bin 交互。

  





------





### 16) 

### postCode

### （float64）





- 含义：邮编前三位。你这列是 float，建议转字符串。

- 工程化

  

  - postCode_str = postCode.astype(str).str.split('.').str[0].str.zfill(3)（避免 1→“001” 信息丢失）。
  - 双路：postCode_cat（category）+ postCode_freq（频数编码）。
  - 可再映射到州/省（若有外部映射，此比赛一般不允许外部数据，就先不用）。

  





------





### 17) 

### regionCode

### （int64）





- 含义：地区编码（通常更粗的区域）。

- 工程化

  

  - 转 category。
  - 与 postCode 冗余但粒度不同，可共存；也可做对齐/一致性检查。

  





------





### 18) 

### dti

### （float64）





- 含义：Debt-to-Income 债务收入比。

- 工程化

  

  - 原值 + 分桶（极端高值拉一桶）。
  - 与 loan_income_ratio/interestRate 交互。

  





------





### 19) 

### delinquency_2years

### （float64）





- 含义：近两年逾期 30+ 天的次数。

- 工程化

  

  - 原值（可截断 3+ 为一组）。
  - 分桶：0、1、2+。

  





------





### 20–21) 

### ficoRangeLow

###  / 

### ficoRangeHigh

### （float64）





- 含义：FICO 下/上限。

- 工程化

  

  - fico_mean = (low+high)/2、fico_diff = high-low。
  - 桶化：fico_bin（如 580-, 580–620, … 760+）。
  - 与 interestRate、grade 交互（利率-评分错配可提示风险）。

  





------





### 22) 

### openAcc

### （float64）





- 含义：未结信用额度数量。

- 工程化

  

  - 原值 + 分桶。
  - 与 totalAcc、revolBal 组合（见下）。

  





------





### 23) 

### pubRec

### （float64）





- 含义：负面公共记录数。

- 工程化

  

  - 原值 + 二值化（是否>0）。
  - 合并与 pubRecBankruptcies（见下）。

  





------





### 24) 

### pubRecBankruptcies

### （float64）





- 含义：破产记录数。

- 工程化

  

  - 原值 + 是否有破产（二值化）。
  - 与 pubRec 合并成 bad_public_records = (pubRec>0) | (bankruptcies>0)。

  





------





### 25) 

### revolBal

### （float64）





- 含义：循环信贷余额。

- 工程化

  

  - 原值 + log_revolBal。
  - 比率：revol_bal_per_acc = revolBal / (openAcc+1)。

  





------





### 26) 

### revolUtil

### （float64）





- 含义：循环额度利用率（%）。你这里已经是 float。

- 工程化

  

  - 原值 + 分桶（0–30、30–60、60–90、90+）。
  - 与 revolBal/openAcc 交互。

  





------





### 27) 

### totalAcc

### （float64）





- 含义：总信用账户数。

- 工程化

  

  - 原值 + 分桶。
  - 比率：open_to_total = openAcc / (totalAcc+1)。

  





------





### 28) 

### initialListStatus

### （int64）





- 含义：初始列表状态（类别）。

- 工程化

  

  - 转 category；若几乎常量可丢弃。

  





------





### 29) 

### applicationType

### （int64）





- 含义：申请类型（个人/联合）。

- 工程化

  

  - 转 category；与 dti/loanAmnt 交互（联合申请但高 dti？）。

  





------





### 30) 

### earliesCreditLine

### （object）





- 含义：最早授信时间（字符串如 Apr-01/Dec-06/Apr-2001）。

- 工程化

  

  - 解析为日期 earliest_dt。
  - 与 issueDate_dt 计算：credit_hist_mon = (issueDate - earliest_dt)/30。
  - 保留原字符串为 category：earliesCreditLine_cat。

  





------





### 31) 

### title

### （float64）





- 含义：贷款标题。你这里是 float（大概率预编码）。

- 工程化

  

  - 保留原值；再做一个频数编码 title_freq（稳健）。
  - 有原文本的话可做文本清洗/主题聚类；当前先保持数值+freq 双路。

  





------





### 32) 

### policyCode

### （float64）





- 含义：策略代码；很多比赛中为常量=1。

- 工程化

  

  - 若单值列直接删除；否则转 category。

  





------





### 33–47) 

### n0 … n14

### （float64）





- 含义：匿名行为计数特征（信息很关键）。

- 工程化

  

  - 全部保留原值；异常值剪裁（如 99.9% winsorize）。
  - 衍生统计：n_sum/n_mean/n_max/n_std/n_nonzero、n_q25/n_q75。
  - （可选）KMeans 聚类标签、PCA 前 2–5 维（后期提分）。
  - 与关键变量交互：n_mean × dti、n_sum × fico_bin（后期）。

  





------





# 建议的“特征映射清单”（便于你实现）



| 原始列                   | 保留/转换      | 新特征（示例）                                               |
| ------------------------ | -------------- | ------------------------------------------------------------ |
| id                       | 保留（不入模） | —                                                            |
| loanAmnt                 | 保留           | log_loanAmnt, loan_income_ratio, loanAmnt_bin                |
| term                     | 保留           | term_bin（<=36/37–60/>60）, 交互：grade×term_bin             |
| interestRate             | 保留           | rate_to_grade_gap（相对等级均值）                            |
| installment              | 保留           | log_installment, installment_income_ratio                    |
| grade (obj)              | → category     | grade_ord（A..G→1..7）                                       |
| subGrade (obj)           | → category     | subGrade_ord（A1..G5→1..35）, subGrade_group                 |
| employmentTitle (float)  | 保留           | employmentTitle_freq（全量频数编码）                         |
| employmentLength (obj)   | 解析+保留原    | emp_len_year（数值0..10）, employmentLength_cat（category）  |
| homeOwnership (int)      | → category     | homeOwnership_grp（合并4桶，可选）                           |
| annualIncome             | 保留           | log_annualIncome                                             |
| verificationStatus (int) | → category     | 与 dti 交互                                                  |
| issueDate (obj)          | 解析+保留原    | issue_year/issue_month/issue_quarter/issue_ym + issueDate_cat |
| purpose (int)            | → category     | 与 subGrade/term_bin 交互                                    |
| postCode (float)         | 转字符串+双路  | postCode_cat（category，零填充3位）、postCode_freq           |
| regionCode (int)         | → category     | 与 postCode 对齐分析                                         |
| dti                      | 保留           | dti_bin（极端高值一桶，可选）                                |
| delinquency_2years       | 保留           | delinq_bin（0/1/2+）                                         |
| ficoRangeLow/High        | 保留           | fico_mean, fico_diff, fico_bin                               |
| openAcc                  | 保留           | 分桶；revol_bal_per_acc                                      |
| pubRec                   | 保留           | pubRec>0（bool）                                             |
| pubRecBankruptcies       | 保留           | bankrupt>0（bool）, bad_public_records（与 pubRec 合并）     |
| revolBal                 | 保留           | log_revolBal, revol_bal_per_acc                              |
| revolUtil                | 保留           | revolUtil_bin（可选）                                        |
| totalAcc                 | 保留           | open_to_total                                                |
| initialListStatus (int)  | → category     | 若常量则删                                                   |
| applicationType (int)    | → category     | 与 dti/loanAmnt 交互                                         |
| earliesCreditLine (obj)  | 解析+保留原    | credit_hist_mon, earliesCreditLine_cat                       |
| title (float)            | 保留           | title_freq                                                   |
| policyCode (float)       | 检查单值       | 单值删；否则→category                                        |
| n0..n14                  | 保留           | n_sum/n_mean/n_max/n_std/n_nonzero/n_q25/n_q75、（可选）KMeans/PCA |



------





## 实施注意点（避免信息丢失与数据泄露）





1. 双路保留：对“原来是 object、现在转成 category”的列，同时保留数值化版本（频数编码/目标编码CV）——信息更全面。
2. 所有编码统计（如频数/均值参照）在 train+test 合并后计算，但不使用标签；目标编码要严格用 CV 内部拟合。
3. 日期解析失败要兜底：保留 issueDate_cat / earliesCreditLine_cat 以免丢失可用模式。
4. 极端值处理：收入、余额等长尾变量建议 log 或 winsorize，避免少数极端样本主导分裂。
5. 冗余交互可以先不开，等 baseline 收敛后逐步加入，观察 AUC/线上线下差异。





------



如果你愿意，我可以把这份“特征映射清单”直接落成 Notebook 两格代码：



- 一格生成所有上述衍生特征（不丢原始信息）
- 一格安全选列（仅数值/布尔/category）并列出最终 features 清单（还能打印每类特征数量）





说一声“要代码版”，我就给你可直接运行的 Cells（带 tqdm 进度条）。