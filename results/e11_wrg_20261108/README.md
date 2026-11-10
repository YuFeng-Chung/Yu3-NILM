# E11 Yu3 與 WRG-CNN 公平比較

作者2026-11-08明確授權正式比較與估時。正式writer於台灣14:12:38啟動，14:51:52完成，約39.24分鐘（不含前期實作／前測與後續QA）。18HPO、30CV、5pilots、50正式訓練與50×211完整評估完成；獨立重算3,696,720筆，completed_audit全過，本Macwriter及caffeinate均已退出。PID40781為歷史，不重啟本輪。正式結果見`E11結果報告.md`。

## 執行設計

- WRG依E2同一18組Adam搜尋、batch32、682/173 inner群組切分，seed42，min30/max150、patience20；不額外給優化器或表徵搜尋機會。
- E3使用既有相同五折group及六種訓練recipe，noise與missing依作者確認v4勝票規則；所有並列均保留。
- 正式WRG包括No-Defense、S3、Combined及所有額外選中策略；每個實際訓練recipe依inner pilot固定epoch，再在855筆上以seed42–51訓練。S0/S1/S2共用同一No-Defense權重，保留各策略獨立推論。
- Yu3重用來源hash已凍結的E2/E3/E4搜尋、CV及No-Defense/S3/Combined十seed結果，不重訓或重選。
- 共用500點波形、污染函式、污染seed1000–1009；先污染後各自成像。完整clean＋21強度×10實例共211條件：AWGN30/20/15/10/5/0dB、椒鹽1/5/10/20%、隨機缺失1/5/10/20%、連續缺失1/5/10/20/30/40/50%。
- 所有WRG搜尋、選策、epoch及final訓練完成後，才開本輪外層test評估。完整報告同策略比較及各自選策的所有配對，保留負面結果。

## 表徵與公平性的明示調整

WRG使用作者CNN、current-only mean-PAA、50×50距離圖、eps10/delta20；不把WRG CNN套V–I圖。共同500點視窗替代原WRG多週期萃取，Adam/batch32/可調dropout依本研究同等搜尋程序，非原SGD/batch16/.25；明示為共同評估協定下重現方法，不稱原論文數值重現。原文median／作者code mean與上游MLP初始化的吞例外行為保留、揭露。

此為兩整套方法比較；不同輸入資訊與表徵不單獨歸因於CNN或回饋機制。WRG2020也不獨自滿足「近期方法」要求，本次依作者指定只比較Yu3/WRG。

## 估時與前測

`preflight_audit.json`已通過132次作者圖逐值核對、訓練增強及S0/S1/S2濾波的共同波形→VI逐值核對，並完成3種recipe×855筆×2輪真實MPS訓練與重載。推論前測用219筆train作形狀代理，未用test。

平均前測epoch約0.34秒；估算中心運算約48分鐘、保守約124分鐘，另加整理／稽核餘裕，向作者回報1–2.5小時。估算假設及各量測值完整保留JSON；實際early-stop、並列策略、檔案IO與負載可改變時間，不是完成承諾。

## 程式架構與狀態

| 入口（相對RevisedCode） | 目的／流程 | 產出 |
|---|---|---|
| `experiments/prepare_e11_wrg.py` | source/split稽核→train-only前測→凍結 | protocol、preflight及source_snapshot |
| `experiments/e11_wrg_core.py` | 作者CNN＋WRG成像＋共用波形策略接合 | 模型與訓練／推論張量 |
| `experiments/run_e11_wrg.py` | 18HPO→30CV→pilots→final→完整外層→分析 | checkpoint、預測、log、選策與狀態 |
| `experiments/analyze_e11_wrg.py` | 從保存預測獨立Boolean重算→全條件配對→圖表 | CSV、confusion_counts、圖、E11結果報告及completed_audit |
| `experiments/control_e11_wrg.py` | 核對本機程序並status/pause/resume | safe pause與本機程序紀錄 |

最後完成以`completed_audit.json`的all_checks_passed及writer退出核對，不以訓練完成取代分析完成。本輪已完成；不再啟動writer。額外核對S3/S4與Augmentation/Combined各十seed的全部權重及4,220對預測檔逐值一致，報表保留標籤對照。

## 推論範圍

主要是完整描述性Macro-F1、Dabs與Drel，污染先在訓練seed內平均，再配對十個seed。bootstrap僅描述固定歷史split的訓練隨機性，不作母群信賴區間或確認性顯著主張；不以縮family、追加seed或test挑策略迴避前輪統計限制。歷史test已查看、group為類別×住宅代理、single-seed HPO／非nested及Mac非CUDA等限制保留。

已產出完整CSV、七組600dpi PNG/SVG曲線、逐條件混淆整數NPZ、Markdown結果報告及獨立audit。112列配對比較含完整有利／不利條件；Yu3 S3對WRG S3/S4在AWGN20至0dB較低、椒鹽較高；Combined連續缺失50%為WRG平均略高，其餘完整結果見報告。整理包`../../E11完整結果整理包_20261108.zip`由`finalize_e11_delivery.py`產生並驗證CRC及逐檔SHA；checkpoint與逐筆預測留在原results，不重複打包。未修改原稿、後續已完成16頁16表2圖正式E11結果回應稿，入口`../../e11_full_review_20261108/README.md`、未公開上傳，不宣稱Drive同步完成。
