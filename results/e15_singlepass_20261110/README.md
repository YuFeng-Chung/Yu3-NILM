# E15 單次推論與變動第二通道

2026-11-09：作者確認後已完成20個正式模型、790訓練epoch、每模型兩輸入×211條件、完整分析及實驗報告。研究完成以`completed_audit.json`、`qa/training_audit.json`為準，文件另查`qa/delivery_audit.json`／`qa/visual_review.json`；最新總狀態見`progress.json`。

- `supplied_spec.md`：作者在本對話提供的規格保存，含事先判讀規則。
- `noise_scale.json`／`noise_scale_seal.json`：十個E14 NoiseFB final模型、855筆訓練樣本的固定尺度與SHA；不得改動。
- `calibration_moments.npz`：每seed每樣本的mean／population std與ID。
- `source/e15_core.py`：E14隔離副本，只新增兩個單次模式；`source/core_changes.diff`與`source/original_e14_core.py`供比較。
- `source/prepare_pilot.py`：只校準／前測／pilot／train-only估時，不含正式訓練入口。重複執行會核對封存來源與既有run設定，不重訓已有result的pilot。尚需重驗證來源SHA與writer狀態，不能依舊PID認定活躍。
- `pilots/`、`split_manifest.json`、`epoch_settings.json`：內層選輪、ID、checkpoint、完整log與結果；驗證採主要[X,0]。
- `preflight_audit.json`、`prelaunch_audit.json`：前向／loss／梯度／亂數／舊模式等價與獨立保存資料核對。
- `protected_sources_sha256.json`：準備前既有程式／原始資料／E14產物的保護清單；前後逐檔核對，不以新版清單覆寫舊版。
- `prelaunch_report.md`：固定尺度、各模型fixed_epochs與估時，供作者確認。

亂數：SHA256 UTF8 pipe分隔，前8 bytes little-endian mod(2^63−1)；training key=(seed,epoch,zero-based batch)，evaluation key=(sampleID)。CPU局部torch.Generator。Mix每batch先產生全部Gaussian，再獨立抽每樣本Bernoulli（<.5為noise），推論[X,N]各樣本固定Gaussian。各eval batch32。

校準使用規格指定E14全855訓練checkpoint，因此包含後續inner-validation樣本；pilot是既定轉移尺度下的選輪，非完整nested模型選擇。未以外層test校準、選輪或估時。

正式研究已完成：兩模型各10seed，211條件，兩種測試輸入，主要／全部檢定與逐強度判讀、CSV及實驗報告。既有程式／結果不覆寫。不因此README自動啟動任何程序。


## 主要結果

主要[X,0]、脈衝5／10／20%之Macro-F1均值：

| 模型與保存端點 | 5% | 10% | 20% |
|---|---:|---:|---:|
| 1P-NoiseCh zero | 56.27 | 48.88 | 39.22 |
| 1P-Mix zero | 59.75 | 50.16 | 41.11 |
| E14 NoiseFB pass1 | 64.21 | 57.17 | 48.61 |
| E14 ZeroFB reported | 50.49 | 41.15 | 32.27 |

兩模型在三強度全部依原規格判為「部分或中間效果」，既未達單次足夠、也未達兩次必要分支。兩者六個條件皆顯著低於NoiseFB；Mix各強度均顯著高於ZeroFB（Holm p=.003906）。NoiseCh對ZeroFB在5%不顯著（p=.105469），10／20%顯著（p=.009766／.003906）。此為各強度各模型內兩比較Holm，非跨強度／跨模型校正。3pp非等效門檻，不能據此宣稱兩次推論普遍必要。

## 程式與完整交付

- `source/run_e15.py`：固定尺度／epoch授權後的正式runner；`source/evaluate_e15.py`兩輸入batch32推論，全部影像與E14一致。正式權重、完整訓練log及8,440個新預測檔保留在`final/`。
- `source/analyze_e15.py`：從新模型與50個保存對照來源重算18,990預測檔／4,158,810標籤的四指標；輸出逐實現、每seed與摘要CSV。全部440精確p另以符號排列核對，12項事先指定對比及逐強度判讀單列。
- `source/verify_training.py`：790正式epoch、IDs／normalization、單次BN計數、checkpoint重載及兩輸入完整性；`source/verify_delivery.py`：封存來源、CSV／Holm／決策、Word表格／數值／原文／紅字／模板與嵌圖QA。
- `confusion_counts/`：新模型兩輸入逐條件11×11整數混淆計數。`reference_manifest.json`列明E14／E1E4保存來源，沒有新訓練或forward對照。
- `analysis_plan.json`：在E15新外層評估前凍結的比較與分析來源。`formal_source_sha256.json`／`final_results_sha256.json`分別封存正式來源與新模型checkpoint／logs／預測。
- `E15結果與程式整理包.zip`：報告、CSV、圖、來源、協定、校準統計、混淆計數、train log及QA；CRC與逐檔SHA見`package_audit.json`。大型checkpoint、逐筆npy預測與成像快取保留結果目錄，舊對照保留原E14／E1E4。整理包不是獨立全訓練環境。

`protocol.json`／`prelaunch_audit.json`及`local_process.json`是準備階段歷史，當時正式授權為false、正式模型數0，未回寫。後續作者有效確認見`authorization.json`；正式本機程序歷史見`formal_local_process.json`，現在writer已退出。`progress.json`與完成QA為後續狀態。不要因舊PID或讀取本檔自行重新啟動。首個PyTorch唯讀mmap訊息保留stderr並解釋於`qa/readonly_array_note.md`；輸入未作in-place寫入，研究來源未為隱藏訊息而變更。
