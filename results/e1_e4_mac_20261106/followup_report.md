# E1／E3／E4 新增受控補證完成

完成 330 個正式模型，每模型 210 次污染評估；全部預測重算與訓練紀錄稽核通過。

E3策略計數程式於外層測試前修正，恢復原Stage1相對No-Defense勝場定義；修正前已查看部分內層fold結果，完整揭露於selection_rule_amendment.json，不稱完全事前註冊。
作者於已查看部分inner scores後、外層測試前核准v4：單折雜訊完全平手各計勝票，最終勝票平手並列，跨折平均F1僅描述、不參與平手處理；缺失min-win及total-win均平手並列。所有並列策略均評估，所有不重複策略配對納入同一E3 Holm family，不依test挑選。詳tie_rule_amendment_v4.json及E3_co_selection_report.json。
單一固定外層切分，10 個訓練 seed；每個 seed 先平均 10 次污染。不混用舊 CPU／MPS 結果。所有 epoch 與策略決定先於本輪外層評估。
E3 使用可重建的新五折群組切分，不聲稱重現歷史 folds。Combined 使用明載的 weight decay 下限，部分基線 E2 已有較高 weight decay，正則化候選可能重複。
RecAux 採一通道編碼器。Yu3−RecAux 同時涉及 pass 次數、分類 loss 與輸入介面；不能視為純殘差因果效果。ZeroFB／ShuffleFB 是另列的敏感度控制。

乾淨及全部污染絕對 F1、Dabs、Drel 見 summary.csv；10seed逐筆平均見 per_seed_averaged.csv；配對差、效果量、區間與兩個完整 Holm families 見 paired_tests.csv。大量檢定的解析度有限，未顯著不能當作等效。
相對 CE 逐 seed 依原 Word 公式計算；分母不穩定或非正值明列。隨機缺失 CE 不予解讀。原強度與延伸連續缺失網格分列；30–50%僅測試、不訓練。

| 策略 | 模型 | Clean F1% | 連續缺失20% F1% | 連續缺失20% Dabs pp |
|---|---|---:|---:|---:|

## E3並列策略
| 模型 | 雜訊 | 缺失 |
|---|---|---|
| Yu3 | S3 | combined |
| ResNet18 | S3 / S4 | augmentation / combined |
| MobileNetV3 | S3 / S4 | augmentation / combined |
| ViT-Tiny | S3 / S4 | augmentation / combined |

## 共同策略結果
| 策略 | 模型 | Clean F1% | 連續缺失20% F1% | 連續缺失20% Dabs pp |
|---|---|---:|---:|---:|
| no_defense | M1 | 70.14±4.06 | 66.72±3.61 | 3.42±1.38 |
| no_defense | RecAux | 71.11±5.40 | 66.44±3.26 | 4.67±2.96 |
| no_defense | M3 | 69.79±4.70 | 64.31±4.23 | 5.48±2.01 |
| no_defense | Yu3 | 71.63±3.60 | 66.61±2.74 | 5.02±2.22 |
| S3 | M1 | 71.35±4.47 | 66.21±3.09 | 5.14±1.68 |
| S3 | RecAux | 71.53±5.33 | 66.58±4.29 | 4.94±2.33 |
| S3 | M3 | 70.21±4.57 | 64.72±3.97 | 5.49±1.84 |
| S3 | Yu3 | 72.11±5.15 | 67.28±3.35 | 4.83±2.74 |
| combined | M1 | 71.41±4.67 | 67.89±3.95 | 3.52±2.84 |
| combined | RecAux | 70.43±4.15 | 65.92±3.97 | 4.52±2.02 |
| combined | M3 | 65.41±5.79 | 62.37±5.82 | 3.04±2.56 |
| combined | Yu3 | 71.65±4.30 | 65.87±4.47 | 5.78±2.78 |

![連續缺失退化](consecutive_degradation.png)

結果需連同 absolute F1、paired差、校正及限制解讀。