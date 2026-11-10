# E8 電腦端效率量測交付

2026-11-08 06:36（Asia/Taipei）正式計時與記憶體量測結束，程序已退出、可恢復使用電腦；從作者開始指示至結束約23分鐘。其後整理報告不要求電腦閒置。

- 閱讀入口：`E8電腦端效率量測報告.md`。
- 數據：`latency_summary.csv`、`raw_accepted_timings.csv`、`memory_summary.csv`；完整原始紀錄為`timing_*.json`、`memory_*.json`。
- 圖：`E8_latency.png`（600dpi）／`E8_latency.svg`。
- 完整性：`independent_audit.json`及`measurement_audit.json`均all_checks_passed=true；主要60,000筆、20個設定、20個獨立記憶體程序，無預定義漂移旗標。
- 透明修訂：初批PyTorch四執行緒受負載干擾，40次嘗試僅23輪有效，完整保留`initial_incomplete_timing_pytorch_4.json`及獨立摘要；依`rerun_amendment.json`在30秒低負載後只做一次整組重測，30/30有效。不可隱去此追加，或把它稱原先已預註冊。
- 溯源：`protocol.json`、`execution_source_manifest.json`、`source/`、`environment_packages.json`；ONNX及合成輸入保留於本目錄，原checkpoint仍在前輪結果。

本機M2 CPU、FP32、batch=1、合成64×64輸入；四模型S3 seed42，未重訓或使用test選擇。只量模型forward，未涵蓋前處理、真實資料端到端延遲、MACs／FLOPs、其他硬體或部署。記憶體是共同runtime匯入後RSS與程序lifetime peak，不是activation-only峰值。正式Word審稿回覆、效能與效率取捨圖及原稿修訂尚未完成；不能將量測完成寫成整題審稿問題已結案。

重現及工具入口以`Code/RevisedCode/experiments/`為準；本次run-specific腳本有覆寫防護，既有輸出須保留，不能直接重啟同一結果writer。未獨立驗證Google Drive同步完成。
