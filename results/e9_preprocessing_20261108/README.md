# 後續回應稿已交付

20頁13表2主圖英中Word已完成，含主文Discussion及補充材料方法／結果；入口`../../e9_full_review_20261108/README.md`，集中副本`../../0_E1-E13實驗結果文稿/E9結果回應稿.docx`。舊研究audit的Word未完成旗標及下文為研究交付當時快照，不回寫；原稿未改。

# 最新完成狀態

本輪8pilot、40正式訓練與41條件評估完成；research／獨立raw／完整成像診斷audit通過。結果見`E9結果報告.md`、`summary.csv`及`paired_factor_changes.csv`。正式Word與原稿尚未處理。以下啟動段落保留歷史。

# E9 獨立前處理敏感度補測

2026-11-08作者授權，資料及真實MPS前測通過，正式writer已啟動。本段是啟動快照；實際狀態查`progress.json`、`current_run.json`、`FAILED.json`與`completed_audit.json`。

四條件×M1/Yu3×五seed，共40正式訓練；8組inner-validation epoch pilots另計。此為獨立補測，不替換舊實驗前處理。定案見`protocol.json`；設計沿革見`../../e9_design_20261108.md`。

共同合格965/1074；原outer成員取交集為train765/test200，inner616/149。排除109筆集中Laptop，完整`coverage.csv`與`window_manifest.csv`保留。無fallback，不依分類結果調門檻。

從Yu3專案根目錄，使用具Torch/MPS的Python執行：

```sh
python Code/RevisedCode/experiments/control_e9.py status
python Code/RevisedCode/experiments/control_e9.py pause
python Code/RevisedCode/experiments/control_e9.py resume
```

pause須等PAUSED與writer退出；resume依作者指示，不重複start。程序為本機程序，不隨Drive遷移。

完成後來源：`E9結果報告.md`、`summary.csv`、`paired_factor_changes.csv`、`diagnostics_summary.csv`、`figures/`、`independent_audit.json`及`completed_audit.json`。正式Word、原稿整合及E13舊分析處置另列。
