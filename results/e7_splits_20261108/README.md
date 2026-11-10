# E7：多組 group 切分與各類別分析

最新交付狀態 2026-11-08 12:59：Combined與S3各60/60訓練／完整評估及兩completed_audit均全過，writer已退出；同一份21頁18表4圖E7回應稿、集中副本及ZIP已完成，入口為`../../e7_full_review_20261108/README.md`。獨立重算2,890,800筆、逐頁QA及CRC/SHA全過。下方啟動／等待／暫停描述是歷史快照，不要重啟writer或queue。原稿未改。

2026-11-08（台灣）。作者本對話指示「開始對話，接續 E7 實驗」。

## 狀態與入口

- E7-B 已完成：`part_b/README.md`、132列逐類別摘要、12張600dpi混淆矩陣PNG及SVG、計數CSV、Yu3−M1類別差異表。四模型×三條件，10個訓練seed；重算183,960筆預測，`analysis_audit.json`及另用逐類TP/FP/FN的`independent_prf_audit.json`通過。此處比原規畫B多保留RecAux，沒有新增訓練。
- E7-A 已啟動：5組新切分×3seed（42/123/2026）×Yu3/M1/RecAux/ResNet18，共60次正式訓練。即時狀態以`progress.json`、`current_run.json`、`run.log`、`stderr.log`與本機程序為準，不能以本說明當即時監控。
- 四模型MPS真實訓練一輪及checkpoint重載通過；原切分seed42四模型各31條件預測與E4完全相同。`preflight_audit.json`粗估純訓練約133分鐘，前測同時有B繪圖CPU工作，估值含冷啟動且未含評估／分析，不是完成時間承諾。
- 全部60模型及評估結束後，runner自動產出`analysis/`與`completed_audit.json`。Word回覆與原稿修改不是本audit的完成項，仍待結果後另做。

## 凍結設計與證據

以`protocol.json`為準。原始規畫為`../../Yu3重投_第一階段實驗修訂_1102.docx` E7段；審稿原文為`../../Yu3_審稿意見列表_中英對照.docx` R3-8(b)、R3 minor-9。新測試結果產生前凍結來源hash、split、settings與分析規則，`source_snapshot/`保留程式。

| 模型 | 策略 | 固定epoch | lr | weight decay | dropout |
| --- | --- | ---: | ---: | ---: | ---: |
| Yu3 | Combined | 36 | .0005 | .0001 | .5 |
| M1 | Combined | 42 | .0005 | .0001 | .5 |
| RecAux | Combined | 35 | .0005 | .0001 | .5 |
| ResNet18 | Augmentation／Combined等價設定 | 48 | .0001 | .001 | .2 |

基線只依E2 inner-validation分數選ResNet18（.65947）；不依E7 test選模型。M1/RecAux沿用Yu3的E3缺失策略。四模型原切分42–51十seed的epoch／設定均已核對；ResNet18兩策略的本輪31條件保存預測完全一致。

切分採原程式每類2000次候選設備子集、最接近20%樣本、嚴格改善才換子集；新seed1–5。原loader使用os.listdir而歷史機台插入順序無法復原，因此新切分明確以字典序排序group，不能宣稱只是替換原seed、逐位重現原切分生成程序。原切分直接沿用已驗證855/219映射；六組test ID清單各異，類別齊全，train/test group不交疊。`split_class_counts.csv`與`split_overlap.csv`保留數量及重疊。

`group`是原loader的「類別×住宅location」，不是已驗證的唯一實體機台ID；同一住宅不同類別仍可能跨train/test，不能稱整屋隔離。metadata與1074個raw sample group全部核對通過。

每組新切分只用自身train fit global normalization，固定原切分的HPO／策略／epoch；不在新test重新調參、挑epoch或策略。原設定已利用過原資料；新split的test可能曾在舊split參與訓練／調參。這是對既定設定的切分敏感度分析，不冒稱全新外部驗證或nested-CV。

測試為clean一次，AWGN10dB、連續缺失20%/50%各10個污染seed1000–1009。先在每個訓練seed內平均污染實現，再平均同split的訓練seed。

原規畫要求五新加原共六split及hierarchical bootstrap。此實作保留六split結果，並以五新split為主要摘要、六split為敏感度；原split有10seed，新split各3seed，兩者分開揭露。配對重抽split及同split共同seed索引10000次、seed20261108，報平均差、pointwise percentile區間及勝出split數，不做Wilcoxon/Holm顯著性主張。split間測試group有重疊且共用資料，bootstrap區間僅是條件描述，不能當獨立split母群95%涵蓋率、確認性推論、等效或多重比較控制的證據。原規畫預期的「不同split波動便證實依賴」也需限縮為描述觀察。

## 程式與操作

從`Code/RevisedCode/`使用本機已驗證venv Python執行：

```text
python experiments/control_e7_splits.py status
python experiments/control_e7_splits.py pause
python experiments/control_e7_splits.py resume
```

`pause`在epoch或評估item保存邊界退出，必須確認`PAUSED.json`及本機程序退出才搬動／闔蓋；`resume`限作者授權。kernel writer lock及PID命令驗證只防本機重複writer，不是跨主機分散鎖。Google Drive雲端同步未獨立驗證。

| 入口 | 目的 | 輸出 |
| --- | --- | --- |
| `experiments/e7_split_core.py` | 共用固定模型／條件、2000子集split、指標與來源驗證 | 固定group split與讀取舊預測 |
| `experiments/prepare_e7_splits.py` | metadata／split／設定／等價驗證，凍結協定與來源 | protocol、split CSV、preparation audit |
| `experiments/preflight_e7_splits.py` | 四模型真實MPS一epoch，原31條件逐筆預測核對 | preflight audit、保守估時；scratch刪除 |
| `experiments/run_e7_splits.py` | 單writer鎖、逐模型固定輪數訓練與31條件評估、自動分析／稽核 | runs、逐epoch進度、checkpoint與逐筆預測 |
| `experiments/analyze_e7_splits.py` | 原切分B與完成後A的逐筆重算、逐類／混淆／paired bootstrap | CSV、600dpi圖、analysis audit |
| `experiments/control_e7_splits.py` | start/status/pause/resume | 本機PID、鎖、暫停請求 |

以上程式路徑以`Code/RevisedCode/`為基準。結果不取代學生CUDA獨立重跑；未改投稿原稿、既有研究來源或前輪結果。
