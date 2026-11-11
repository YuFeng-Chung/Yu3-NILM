# E12 研究結果入口

先讀 `E12結果報告.md` 與 `completed_audit.json`。本輪實驗／分析／獨立稽核完成。

全部合成條件，沒有重訓或部署／現場驗證。完整設計見 `protocol.json`；啟動與安全暫停由 `../../experiments/control_e12.py` 管理。完成後不要重啟。

打包檔不重複封裝舊模型與原資料，需與原E1/E3/E4和PLAID來源搭配；`partial/`是程序中間結果，正式完整輸出以 `predictions/` 為準。
