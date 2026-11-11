# E10 WHITED：夜間自動接續入口

2026-11-08作者要求休息期間接續E12，明早可讀取成果。此目錄為獨立E10研究工作，不改既有模型與實驗。

## 先看目前狀態

- `queue_status.json`：等待E12、開始E10、完成或暫停。
- `progress.json`：目前pilot／正式訓練、epoch或研究完成。
- `queue_failed.json`／`FAILED.json`／`stderr.log`：失敗原因（若有）。
- `local_queue_process.json`、`local_process.json`：只對記錄host上的程序有效，不能以同步PID判斷其他電腦。
- 完成以`completed_audit.json`和`delivery_audit.json`全通過為準，不能僅靠ZIP或本README稱完成。

## 已準備的真實資料與協定

官方下載包`../../data/whited_v11/WhiteD.zip`，2,061,689,222 bytes，SHA256 `88e6d5409a550645f7775e2470910b6620f2857fbe6ea7b56e0430919770a242`。頂層1339 FLAC；依包內README r1–r8的1259筆與官網數量一致，另80筆r9未記載，先排除。官方未提供可用來驗證相位修正版的發布checksum，本次凍結實際下載包而不冒稱已逐筆證明發布版本。

各類至少3個設備名稱群組才可分開inner-train、validation、outer-test，保留15類／76群組／759筆。群組依類別＋設備名稱＋地區，移除kit及timestamp，未獨立查證實體序號；非house-disjoint。每類分三個互斥test折，所有納入設備各當一次test，內層每類保留一群組作validation。

15組單seed epoch pilot，之後M1–M5×三折×seed42/43/44共45組正式訓練，只有clean。所有pilot完成並凍結epoch、全部模型訓練完成後，才開outer test。Adam lr.0005、batch32、dropout.5、無weight decay；最長100輪，最早20輪選epoch，patience15/min_delta.001。無新HPO，固定既有架構與loss；M5原detached residual兩pass。完整`protocol.json`為執行依據。

取最大電流一階差分附近一個額定電網週期，前置20%，V/I共同polyphase反混疊重採樣到500點，再逐波形各自z-score、64×64 .1 occupancy飽和與uint8量化。759筆成像與原PLAID成像函式逐筆完全一致（在相同抽取波形下），不代表原始取窗已驗證為所有開啟事件或穩態。通道0=電壓、1=電流，依kit因子核對納入資料的60–300Vrms及45–65Hz；README並未明文標示通道次序，不能隱藏此推定來源。

## 接續與保護

純本機Python queue，每20秒讀本機狀態，不呼叫AI/API，不依token重置啟動。E12研究推論、分析、獨立稽核、ZIP完成、progress為research_complete、完成hash相符、同host writer退出且鎖可取得後，才做E10 MPS前測及正式訓練。E12正式Word不包含在其研究runner的自動產物，不能冒稱Word已完成。

E12或E10出現STOP／PAUSED／FAILED即停止接續，不繞過暫停、不自動重跑。等待最多12小時；來源或資料hash變更則拒絕啟動。queue與writer各有互斥鎖。每epoch原子保存模型／optimizer，真實8筆M5 CPU兩輪中斷重載與不中斷權重／BN完全一致；所有五模型CPU梯度／重載／optimizer下一步一致。MPS真實前測只在E12退出後進行，若失敗停下。

## 完成後自動產物

- `E10結果報告.md`、`summary.csv`、`per_fold_seed.csv`、`per_class_prf.csv`、`confusion_counts.csv`、`paired_differences.csv`。
- 五模型×三seed的OOF逐筆預測、所有run checkpoint／epoch log、15pilot選epoch記錄。
- `figures/`內600dpi PNG與SVG；輸出仍待目視QA。
- `independent_audit.json`重算11,385筆保存test預測的指標／PRF／整數混淆，並驗證split／ID／來源。
- `../../E10_results_20261108.zip`：資料與小型來源、報告、圖、預測及SHA manifest；不重複包全部權重與optimizer。
- 自動更新共享入口的E10狀態與程式總覽E10主表列；歷史狀態保留。

三seed平均±SD只描述訓練變異；三折訓練資料重疊、類別納入限制及全部負面結果均保留，不作確認性顯著或等效主張。合成分析前測僅在臨時目錄，不作研究結果。


## 程式入口（相對本目錄）

`../../experiments/prepare_e10_whited.py`、`e10_core.py`、`run_e10_whited.py`、`analyze_e10_whited.py`、`queue_e10_whited.py`。實際完整腳本保存在`source_snapshot/`；資料來源及納入清單見`sample_manifest.json`、`class_coverage.csv`、`waveform_diagnostics.csv`。

電腦需持續接電且不要闔蓋／關機；queue期間caffeinate防閒置睡眠，無法保證闔蓋後仍運作。
