# E14 NoiseFB 與 PermFB

本輪狀態以 progress.json 與 completed_audit.json 為準；報告交付另核對 qa/delivery_audit.json 與 qa/visual_review.json。

## 協定與來源

原規格：`../../../../Revised/Claude修訂_20261109/E14_NoiseFB_實驗規格_給Codex.md`（由本目錄算相對路徑）。只做 No-Defense；NoiseFB 為主要、PermFB 為次要。原控制 checkpoint 由初版 `run_revision_followup.py` 與共同 `revision_followup_core.py` 於11/06訓練；後續整輪由 v4 接續，v1/v4 家族 pilot/final AST 相同。詳細證據 `runner_provenance.json`。既有研究來源、模型與結果不修改。

E14 執行程式全部在 `source/`；`e14_core.py` 為隔離副本，新增 FeedbackControl 模式、build/logits/loss 名稱與必要亂數 context／ID 傳遞，不改原有模式邏輯。`core_changes.diff` 可逐行核對。依原分組內層切分選 epoch，NoiseFB42、PermFB49；新規格 patience30，舊控制20，並非完全相同的 pilot patience 或固定計算預算比較。每模型 seed42–51，共20正式模型；相同原855/219與train-only尺度、E2 Yu3設定。

## 執行順序

使用原已具 torch/MPS、NumPy/SciPy/pandas 環境。在本目錄執行 `python source/run_e14.py pilot` 完成各自 epoch；作者已收到 `prelaunch_report.md` 後，`python source/run_e14.py formal` 才執行正式訓練、推論與研究分析。程式由 writer.lock 排他鎖保護；已有結果不重訓。`STOP_REQUESTED` 為原引擎停止訊號，執行中要求暫停須等 epoch／條件保存後退出，不以舊 PID 當作目前程序。

當前一次正式程序 metadata 在 local_process.json；所有即時判斷仍需核對本機 ps 與 log。完成後不應重新啟動；這不是自動重啟授權。不要再執行 source/seal_launch.py，該檔是本次首次啟動來源，會重建保護清單，不能作日常resume工具。

## 結果與推論邊界

- `final/`：新模型完整checkpoint、run_spec、train log、evaluation、outer_ids_labels、outer_predictions與first_pass_predictions。
- `reference_evaluation/`：舊Yu3／ZeroFB／ShuffleFB只推論副本；原生輸出逐值對照原E4保存預測，原權重不重訓。
- `per_seed_condition_realization.csv`：每模型、訓練seed、污染實現與端點四指標。`per_seed_averaged.csv` 先平均10實現；`summary.csv` 跨10訓練seed均值及樣本SD。CSV的accuracy／PRF為0–1，差值欄pp為百分點。
- `paired_tests_prespecified.csv`：主NoiseFB及次PermFB各3強度×2對照，共12項。每模型、每強度內對Yu3／ZeroFB兩項Holm，n10雙尾精確Wilcoxon；不跨強度或跨主／次模型校正。
- `paired_tests_all.csv`：包含ShuffleFB、其他強度及原生輸出；非事先指定者標記exploratory_unadjusted，p_holm空白，不混充主要顯著證據。
- `prespecified_interpretation.csv`：規格第5節逐強度機械套用，3pp為描述性條件，非等效檢定。若強度間不一致，不任選有利強度作總結。

NoiseFB仍保留逐樣本殘差mean/std，PermFB保留逐樣本殘差值分布；不能推論完全沒有樣本資訊或任何變動通道皆足夠。歷史test已查看且本輪假設由舊診斷引出；單一split、不同epoch及Mac/MPS平台限制保留，非學生CUDA獨立重現。

## 隨機性與稽核

train穩定種子為SHA256(seed,epoch,zero-based batch index)，eval為SHA256(sampleID)；UTF-8 pipe分隔、前8 bytes little-endian、mod(2^63−1)。CPU torch.Generator，不消耗全域torch亂數。Noise採population std correction=0，逐像素Gaussian後clamp(min=0)；clamp後輸出矩不要求與R完全相同。PermFB每樣本獨立排列。推論同樣本ID跨污染條件重用底層亂數圖，污染仍由原E4公式／seed1000–1009產生。

`completed_audit.json`包含獨立Boolean F1重算與所有p值符號排列核對。`preflight_audit.json`、`aux_semantics_audit.json`、`evaluation_timing_train_only.json`為前測；`qa/training_audit.json`與`qa/delivery_audit.json`包含完整epoch／設定／ID／normalization／BN／舊來源保護及Word內容核對。`qa/visual_review.json`記錄Word逐頁視覺檢查。`protected_sources_sha256.json`凍結6,694個相關舊來源及結果；`e14_source_sha256.json`為正式啟動時E14來源。後續報告／稽核程式可新增，研究來源若變動必須披露，不重寫凍結清單。

## 推論批次數值修正

正式結果全部使用原E4的batch32。首批曾使用batch219，於ZeroFB seed46的SP20%、pollution seed1008發現1/219筆分類不同後，guard立即停止。首批全部保存於numerical_audit/batch219，另記inference_batch_amendment.json與INITIAL_BATCH219_FAILURE.json。source/resume_evaluation.py只重做推論／分析，不呼叫train；所有20模型不重訓。正式evaluator唯一變更為batch回到32，原啟動SHA清單保留，修正前來源副本與新SHA另存供驗證。前測batch219估時與部分舊描述屬修正前歷史，不當作正式結果。
