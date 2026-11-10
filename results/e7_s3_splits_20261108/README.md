# E7 S3 多切分雜訊補充

最新交付狀態 2026-11-08 12:59：Combined與S3各60/60訓練／完整評估及兩completed_audit均全過，writer已退出；同一份21頁18表4圖E7回應稿、集中副本及ZIP已完成，入口為`../../e7_full_review_20261108/README.md`。獨立重算2,890,800筆、逐頁QA及CRC/SHA全過。下方啟動／等待／暫停描述是歷史快照，不要重啟writer或queue。原稿未改。

2026-11-08 09:54更新：作者明確說「繼續」，已在AC99%恢復同目錄，驗證首組Yu3由epoch49接至50再52、log連續且PAUSED/STOP清除。S3恢復執行，後續以本機程序、progress與log為準；下一段09:07暫停為歷史。

最新狀態快照：2026-11-08 09:07作者要求搬動前暫停。Combined已60/60完成且audit全過；S3前測通過、自動接續後安全暫停於split1 Yu3 seed42 epoch49/52，0/60完整run。resume.pt含model/optimizer可載入、49輪log連續、writer與queue均退出，見`travel_pause_audit_20261108.json`。等待作者抵達後明確授權才resume；不得自行恢復或重建queue。本次新前測純訓練外推約168分鐘，未含評估與報告。以下為原安排與設計說明。

2026-11-08。作者授權來源為`../../e7_s3_handoff_20261108.md`；本補充保留Combined E7所有結果，補齊原S3雜訊主結果對R3-8(b)與R3 minor-9的證據，不替換Combined或重選策略。

## 本次準備與自動接續

- 協定、來源hash、相同六組split清單、四模型原S3設定與epoch已凍結於`protocol.json`；新訓練為相同五新split×三seed42/123/2026×四模型，共60組。
- 當前排程狀態讀`queue_status.json`，正式訓練狀態讀`progress.json`及`current_run.json`，不要以本README作即時監控。
- `queue_e7_s3.py`等Combined的`completed_audit.json`通過、progress completed、程序退出與kernel lock釋放後，才執行S3真實MPS前測及啟動獨立writer。提前執行preflight或controller start會被拒絕，已實測此門檻。Combined pause/failed或S3 STOP_REQUESTED會終止接續安排，不能繞過作者後續暫停。
- S3前測會做四模型一個真實train epoch／checkpoint重載，以及原切分seed42四模型各101條件預測與E4一致性核對；前測scratch不混入研究。通過後才啟動60組。
- `historical_scheduling_estimate.json`以已完成原S3十seed訓練實測外推，純訓練約157分鐘，未含本次新grid評估／分析／報告。這是舊split實測外推；本次新split前測要待Combined退出才執行，其實際估時另存`preflight_audit.json`，不能把兩者混稱。

## 固定範圍

|模型|策略|epoch|lr|weight decay|dropout|
|---|---|---:|---:|---:|---:|
|Yu3|S3|52|.0005|0|.5|
|M1|S3|42|.0005|0|.5|
|RecAux|S3|42|.0005|0|.5|
|ResNet18|S3／S4有效設定等價|66|.0001|.001|.2|

設定及epoch均由原E1/E3/E4的40個S3正式run核對。ResNet18基線依E2內層驗證選擇；M1/RecAux轉移Yu3策略。ResNet18原S3/S4十seed的全部本輪條件保存預測完全相同，不依test挑策略。

條件為clean，AWGN30/20/15/10/5/0dB及椒鹽1/5/10/20%，污染各10seed1000–1009，每模型101項。採完整既有雜訊網格，不只選有利強度。

## 既有原切分 B

`part_b/`已從原切分既有S3預測重算884,760筆：四模型×11條件×10訓練seed（污染每條件10實現），484列逐類P/R/F1 mean±SD與test group數，44張600dpi混淆矩陣PNG/SVG及計數CSV、Yu3−M1類別差異。`analysis_audit.json`與獨立boolean TP/FP/FN的`independent_prf_audit.json`通過。此完成不表示新split訓練或正式Word完成；圖表仍需交付版視覺QA。

## 完成後分析與交付

runner自動分析全部原切分與五新split；污染先在每個訓練seed內平均，split內再平均seed。輸出F1、clean-minus-corrupt退化pp、逐類PRF、全部split混淆計數，以及Yu3對M1/RecAux/ResNet18配對差與描述性split/seed bootstrap10000次。五新split為主要摘要，含原split的六組為敏感度；原10seed與新各3seed分開揭露。

group為類別×住宅location代理，非已驗證的唯一實體機台；不同類別仍可能同屋跨train/test。沿用原設定的split敏感度不是nested／新外部驗證，新test可能曾在舊split參與訓練／調參。split互有重疊，bootstrap區間只作條件重抽描述，不作母群95%涵蓋率、確認性顯著、等效或多重比較控制主張。不追加seed、改強度或校正來追求顯著。

正式回應稿待Combined與S3完整audit後依現行E2/E5模式整理：逐題英中原文、紅字英中Response、Methods/Results/Supplementary實際表圖與完整來源；清楚區分原Combined與新增S3理由。Word逐頁渲染QA後才交付；原稿不修改，不公開GitHub。

## 入口

以`Code/RevisedCode/`為基準：

|程式|目的|
|---|---|
|`experiments/e7_s3_split_core.py`|獨立S3條件、來源核對、原S3預測讀取及split演算法|
|`experiments/prepare_e7_s3_splits.py`|凍結相同split、40run原S3設定與完整grid，保存source snapshot|
|`experiments/queue_e7_s3.py`|Combined完成及退出門檻，一次性依序執行前測與啟動；非定期提醒automation|
|`experiments/preflight_e7_s3_splits.py`|四模型真實MPS前測／101條件原預測核對／新估時|
|`experiments/run_e7_s3_splits.py`|獨立鎖與writer，60模型訓練、101條件評估及最終audit|
|`experiments/analyze_e7_s3_splits.py`|F1／退化／逐類／混淆／描述性bootstrap|
|`experiments/control_e7_s3_splits.py`|start/status/pause/resume，啟動前再次驗證Combined門檻|

writer與queue使用本機鎖，不是跨機分散鎖；Google Drive同步未獨立驗證。暫停搬動須等PAUSED及程序退出，不能靠caffeinate宣稱闔蓋可跑。
