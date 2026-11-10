# E15 開跑前回報

日期：2026-11-09（Asia/Taipei）。目前僅完成校準、前測與兩個pilot，等待作者確認；正式20模型尚未訓練，未開外層test評分。

## 固定雜訊尺度

- μ0 = 0.013518345542252064
- σ0 = 0.021488790400326252
- 十個E14 NoiseFB final checkpoint seed42–51，各855筆train、batch32、eval模式第一次推論絕對殘差；逐樣本像素mean／population std（correction=0），8550值各自取中位數。
- 尺度SHA256：`5321bebe21c36dff9407115103ecf4052e1afc175208c5e3bb4270c4a2b8f6a6`；`noise_scale.json`與`noise_scale_seal.json`已封存，不更動。`calibration_moments.npz`保存原始逐樣本統計並以獨立NumPy公式核對。

## Pilot 選輪

兩模型seed42，inner train 682／validation 173，共同group split；min30／patience30／max150／delta.001，從第30輪起最高clean validation Macro-F1（[X,0]）選fixed epoch，同分取最早。均保留原E14引擎選輪／早停／checkpoint邏輯。

| 模型 | fixed_epochs | 停止輪次 | validation F1 | 實測秒／epoch |
|---|---:|---:|---:|---:|
| 1P-NoiseCh | 42 | 72 | 0.6772594541 | 0.786 |
| 1P-Mix | 37 | 67 | 0.6940268778 | 0.785 |

## 正式規畫與估時

正式兩模型×seed42–51，共20模型；855train／219test；Adam lr.0005／WD0／dropout.5／batch32。全部211條件（clean＋210污染），兩種輸入[X,0]／[X,N]，推論batch32。NoiseFB／ZeroFB重用E14保存數值，不重訓。

依pilot訓練秒數按855/682外推：純正式訓練 **13.0分鐘**。同train的219筆實測、兩種輸入推論外推 **11.0分鐘**。估時不使用test影像或分數；不含首次完整污染成像／IO／分析。保守估正式研究 **35–55分鐘**，加完整Word／圖表／文件QA約 **60–90分鐘**；為目前估計，非完成承諾。

## 邊界與稽核

- 單次encoder／classifier／decoder、CE＋0.5MSE、所有梯度、兩通道參數結構、局部CPU Generator、Mix逐樣本Bernoulli、固定樣本ID推論噪聲均通過前測。
- Mix先產生整批Gaussian，再取每樣本獨立Bernoulli（<.5為noise）；同E14 SHA256 key與映射。驗證固定[X,0]，次要[X,N]不參與選輪。
- 按規格以E14 full855訓練之模型校準，包含後續inner validation樣本。pilot為既定轉移尺度下的選輪，不宣稱完整nested獨立選模。外層test未參與校準／選輪／估時。
- 主要比較兩模型各自對NoiseFB第一次推論／ZeroFB原生輸出；SP5/10/20每強度內兩項精確雙尾Wilcoxon n10＋Holm；1P-Mix獨立次要家族。逐強度3pp判讀保留規格原文，非等效或普遍必要性證明。
- 36379個既有來源／E14產物SHA256在pilot前後一致； inherited train、predict、FeedbackControl與選輪函式AST完全相同；舊八模式前向值一致。
- E15程式／產物全部在本目錄；`source/prepare_pilot.py`僅有校準與pilot，沒有正式訓練入口。writer排他鎖、STOP_REQUESTED沿用E14。正式執行須等待本次作者明確確認，不能依舊對話授權自動啟動。
