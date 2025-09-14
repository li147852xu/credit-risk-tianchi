### 指标

confuse matrix

真正类 TP true postive 正类判正

假负类 FN false negative 正类判负

假正类 FP false positive 负类判正

真负类 TN true negative 负类判负

Accuracy: TP + TN / ALL 也就是model全部判对样本/样本总数

Precision: TP / TP + FP 判正的样本里多少个判对了【查准率】

Recall: TP / TP + FN 原来的正样本里多少判对了【查全率】

F1-Score: 2 /  ( 1 / Precision + 1 / recall) 兼顾精准率和召回率两个此消彼长的指标

PR曲线: Precision-Recall Curve

ROC空间

X轴为假正例率 FPR = FP / FP + TN 也就是 原本的负样本中错判为正样本的概率

Y轴为真正例率 FPR = TP / TP + FN 也就是 原本的正样本中判对为正样本的概率

而AUC就是ROC曲线和坐标轴围成的面积 AUC[0.5, 1] 越接近1模型真实越高

KS统计量鉴定区分能力 KS = max( TPR − FPR ) 也就是TPR和FPR都作为Y轴，X轴为选定的阈值【越大越好，注意过拟合】

| KS（%） | 好坏区分能力         |
| ------- | -------------------- |
| 20以下  | 不建议采用           |
| 20-40   | 较好                 |
| 41-50   | 良好                 |
| 51-60   | 很强                 |
| 61-75   | 非常强               |
| 75以上  | 过于高，疑似存在问题 |



### 数据

train.csv 80w+ testA.csv 20w + testB.csv 20w (private)

字段表

|     **Field**      |                       **Description**                        |
| :----------------: | :----------------------------------------------------------: |
|         id         |                为贷款清单分配的唯一信用证标识                |
|      loanAmnt      |                           贷款金额                           |
|        term        |                       贷款期限（year）                       |
|    interestRate    |                           贷款利率                           |
|    installment     |                         分期付款金额                         |
|       grade        |                           贷款等级                           |
|      subGrade      |                        贷款等级之子级                        |
|  employmentTitle   |                           就业职称                           |
|  employmentLength  |                        就业年限（年）                        |
|   homeOwnership    |              借款人在登记时提供的房屋所有权状况              |
|    annualIncome    |                            年收入                            |
| verificationStatus |                           验证状态                           |
|     issueDate      |                        贷款发放的月份                        |
|      purpose       |               借款人在贷款申请时的贷款用途类别               |
|      postCode      |         借款人在贷款申请中提供的邮政编码的前3位数字          |
|     regionCode     |                           地区编码                           |
|        dti         |                          债务收入比                          |
| delinquency_2years |       借款人过去2年信用档案中逾期30天以上的违约事件数        |
|    ficoRangeLow    |            借款人在贷款发放时的fico所属的下限范围            |
|   ficoRangeHigh    |            借款人在贷款发放时的fico所属的上限范围            |
|      openAcc       |              借款人信用档案中未结信用额度的数量              |
|       pubRec       |                      贬损公共记录的数量                      |
| pubRecBankruptcies |                      公开记录清除的数量                      |
|      revolBal      |                       信贷周转余额合计                       |
|     revolUtil      | 循环额度利用率，或借款人使用的相对于所有可用循环信贷的信贷金额 |
|      totalAcc      |              借款人信用档案中当前的信用额度总数              |
| initialListStatus  |                      贷款的初始列表状态                      |
|  applicationType   |       表明贷款是个人申请还是与两个共同借款人的联合申请       |
| earliesCreditLine  |              借款人最早报告的信用额度开立的月份              |
|       title        |                     借款人提供的贷款名称                     |
|     policyCode     |      公开可用的策略_代码=1新产品不公开可用的策略_代码=2      |
|   n系列匿名特征    |        匿名特征n0-n14，为一些贷款人行为计数特征的处理        |























