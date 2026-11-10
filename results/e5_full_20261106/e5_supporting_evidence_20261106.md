# E5審稿佐證實驗：10個配對seed（2026-11-06）

本輪保留原Yu3架構，補受控消融作為佐證，不依held-out結果改版。共11變體×10訓練seed＝110次訓練；每次固定19輪，Adam 0.0005、batch32、dropout0.5。原855／219切分不變。

第一輪seed42–44的33次訓練原樣沿用，再補seed45–51的77次；同環境、同程式與同輸入雜湊。未刪除低分run。初期3seed已被查看，分析計畫在其餘seed完成前固定；這不是完全未看資料的預註冊。

## 乾淨資料結果

| 變體 | Macro-F1（%，mean ± SD） | 相對原Yu3差值（百分點） | 配對p（全12對比Holm） |
|---|---:|---:|---:|
| 原Yu3：誤差圖detach／共用BN | 68.46 ± 3.68 | +0.00 | — |
| 誤差圖不detach | 65.02 ± 5.10 | -3.45 | 1.0000 |
| 影像回饋不detach | 65.44 ± 3.84 | -3.03 | 1.0000 |
| 影像回饋detach | 69.30 ± 3.80 | +0.84 | 1.0000 |
| 無第一階段分類loss | 67.21 ± 3.12 | -1.25 | 1.0000 |
| 無重建loss | 59.26 ± 14.01 | -9.21 | 0.1172 |
| 重建loss權重1.0 | 68.11 ± 3.70 | -0.36 | 1.0000 |
| RecAux（E5匹配版本） | 69.94 ± 2.51 | +1.48 | 1.0000 |
| Three-Pass | 67.26 ± 4.20 | -1.20 | 1.0000 |
| 各pass獨立BN | 69.85 ± 3.45 | +1.39 | 0.4082 |
| 原No-FB（無重建） | 69.39 ± 2.51 | +0.93 | 1.0000 |

n=10是訓練seed數，不是110個獨立seed。非顯著不代表等效，也不排除尚未偵測到的差異。完整12個對比（包括interaction contrast及RecAux vs No-FB）的未校正／全體校正／家族校正p值見paired_tests_clean.csv。

## BatchNorm與回饋的判讀

本輪未檢出共用／獨立BN在clean Macro-F1的全體校正後差異；可作補充對照，不能寫成「兩者等效」或「共用BN完全沒有影響」。

校正範圍須透明：獨立BN對照原始p=0.0371，D家族校正p=0.0371，全12對比主要校正p=0.4082；移除重建loss對照原始p=0.0098，B家族校正p=0.0293，全12對比校正p=0.1172。1102原規畫採家族內校正，本輪另固定更保守的全12對比為主要判讀；兩套完整保留。不能概括說『所有分析都未顯著』，也不能看結果後只選較有利的一套。
本輪未檢出原Yu3相對RecAux的clean優勢。這延續乾淨資料上不強調架構優勢的定位；回饋的穩健性貢獻需看退化量。

原Yu3同一checkpoint的一次／二次推論平均Macro-F1為67.72%／68.46%；平均錯→對4.4筆、對→錯3.3筆。這是同一已訓練模型內的描述，不是獨立No-FB對照。

BN輸入矩診斷保存於各原Yu3 run的bn_diagnostic.json，並彙整於bn_input_summary.csv及bn_pass_distribution_difference.csv：eval模式分別記錄各pass、各層輸入分布與凍結running stats；差距大小本身不能證明效能影響。獨立BN同時分開affine參數及running stats（卷積仍共用），不能把差異只歸因於running stats。

Yu3與RecAux亦同時改變推論次數與分類loss組成（Yu3為0.5 CE1+CE2+0.5 MSE；RecAux為CE1+0.5 MSE），屬整體系統對照，不能單憑此比較歸因為純粹回饋訊號效果；訊號／detach及loss分別另有A／B組控制。

## 波形級擾動：次要佐證

公開PLAID 2014經發布者MD5驗證，且原波形重建的1074張clean影像與既有PNG逐像素／群組匹配後才評估。資料來源：[PLAID 2014](https://doi.org/10.6084/m9.figshare.11605074.v1)。

E5仍採原每波形z-score；AWGN使用各通道去均值變異數決定noise power；連續缺失對兩通道使用同一100/500點區段並線性內插。污染seed1000–1009依sample ID固定，所有模型完全相同。這與Exp4固定全域正規化及防禦訓練不同。

順序是先從乾淨原波形擷取事件視窗，再在該500點視窗加擾動，最後正規化與建圖；未測試雜訊下事件偵測本身是否可靠，不能稱為端到端量測系統驗證。

| 條件 | 變體 | 擾動Macro-F1（%） | 相對自身clean退化（百分點） |
|---|---|---:|---:|
| awgn_10db | 原Yu3：誤差圖detach／共用BN | 15.18 ± 2.98 | 53.28 ± 4.32 |
| awgn_10db | 誤差圖不detach | 11.48 ± 4.87 | 53.54 ± 4.19 |
| awgn_10db | 影像回饋不detach | 11.38 ± 3.85 | 54.06 ± 4.45 |
| awgn_10db | 影像回饋detach | 14.03 ± 2.50 | 55.27 ± 3.55 |
| awgn_10db | 無第一階段分類loss | 16.28 ± 2.32 | 50.93 ± 4.18 |
| awgn_10db | 無重建loss | 9.61 ± 7.04 | 49.65 ± 13.94 |
| awgn_10db | 重建loss權重1.0 | 15.93 ± 2.16 | 52.17 ± 4.71 |
| awgn_10db | RecAux（E5匹配版本） | 12.92 ± 2.35 | 57.02 ± 3.81 |
| awgn_10db | Three-Pass | 15.08 ± 3.11 | 52.19 ± 4.83 |
| awgn_10db | 各pass獨立BN | 14.42 ± 2.78 | 55.44 ± 3.74 |
| awgn_10db | 原No-FB（無重建） | 12.49 ± 2.97 | 56.90 ± 4.22 |
| consecutive_missing_20pct | 原Yu3：誤差圖detach／共用BN | 62.09 ± 2.46 | 6.37 ± 3.35 |
| consecutive_missing_20pct | 誤差圖不detach | 57.03 ± 3.20 | 7.99 ± 3.31 |
| consecutive_missing_20pct | 影像回饋不detach | 56.71 ± 2.57 | 8.73 ± 3.36 |
| consecutive_missing_20pct | 影像回饋detach | 62.86 ± 2.58 | 6.44 ± 2.20 |
| consecutive_missing_20pct | 無第一階段分類loss | 60.37 ± 2.88 | 6.84 ± 1.95 |
| consecutive_missing_20pct | 無重建loss | 49.04 ± 13.13 | 10.22 ± 3.85 |
| consecutive_missing_20pct | 重建loss權重1.0 | 63.23 ± 1.12 | 4.87 ± 3.28 |
| consecutive_missing_20pct | RecAux（E5匹配版本） | 63.05 ± 2.02 | 6.90 ± 2.04 |
| consecutive_missing_20pct | Three-Pass | 61.28 ± 2.75 | 5.98 ± 2.65 |
| consecutive_missing_20pct | 各pass獨立BN | 63.01 ± 1.77 | 6.84 ± 2.70 |
| consecutive_missing_20pct | 原No-FB（無重建） | 62.77 ± 1.71 | 6.63 ± 1.40 |

每個訓練seed先平均10個污染seed的指標，統計n仍為10。退化量越小代表相對自身clean下降越少，負值代表該條件平均分數較clean高；不因此把該條件當作提升方法。

退化量全部24個次要對比Holm校正及全部36個對比的敏感度校正，見corrupted_evaluation/paired_tests_degradation.csv。主次端點依預定12／24組校正判讀，不在看結果後更換family或改單側檢定。

檢定解析度限制：10對非零差值的雙側完整符號置換最小p為2/1024≈0.001953。若把全部36個對比合併Holm，最小首步校正p至少為36×2/1024≈0.0703，因此此樣本數下全36組的敏感度校正無法達0.05；不能把其未顯著當作無影響／等效證據。零差值會使有效配對數更小，解析度不會改善。

### 對核心回饋主張的直接影響

- awgn_10db：原Yu3相對RecAux的退化量差（Yu3−RecAux）為-3.74百分點；原始p=0.0273，24組次要對比Holm後p=0.6289。負值只是平均下降較少，若校正後未顯著，不據此宣稱回饋有明確穩健性增益。
- consecutive_missing_20pct：原Yu3相對RecAux的退化量差（Yu3−RecAux）為-0.52百分點；原始p=0.6953，24組次要對比Holm後p=1.0000。負值只是平均下降較少，若校正後未顯著，不據此宣稱回饋有明確穩健性增益。
- 此E5設定下四個主要對照在AWGN10dB的Macro-F1均低，應承認未防禦模型在此條件的侷限，不用小幅相對差距宣稱可有效抗強雜訊。
- 獨立BN在clean平均較高，但AWGN10dB平均Macro-F1並未同時提高，不構成各條件全面較佳的替代版。保留原架構作研究對象與佐證對照是可行策略，不能推論原設定最優。
- 這輪主要補上實作與消融證據。若要保留「回饋在原稿防禦策略下有額外貢獻」的核心主張，下一個關鍵是Exp4相同正規化、訓練／防禦條件下的Yu3與RecAux等對照；本輪不能替代。

## 對審稿與論文的最小修改建議

- R3-5：明確說明原No-FB不執行decoder且只計CE，新增RecAux補上重建輔助對照。本輪RecAux保留E5匹配的2-channel encoder及零輔助輸入，不混稱最終E1的1-channel M1版本。
- R3-m5：提供完整訊號×detach控制與interaction contrast；原Img-FB與Err-FB比較同時改變兩因素，不能單歸因訊號類型。
- R3-7b：交代encoder／BN共用、train每batch更新兩次、eval凍結；附獨立BN對照與逐pass統計，避免以未顯著聲稱無影響。
- R2-9：附loss權重消融。移除重建loss時decoder不再受重建監督，且detach阻止第二階段分類loss訓練decoder；該誤差圖不再有已學會重建殘差的意義。
- 優先在Supplement增加一張完整受控消融表／配對圖，Methods與Discussion各補短段引用。保留原主文數字，不將不同環境的新舊run混合平均。
- 新結果若反對某項原主張，應誠實收斂文字；本輪未自動改模型或主文。
- R3-2／R3-3的baseline架構、同等調參及各自選防禦策略仍未由E5解決；E1–E4及多切分／跨資料集驗證仍屬後續工作。

## 驗證與重現資料

110個run的19輪log、219筆clean預測、指標重算與checkpoint BN更新次數皆經verify_e5_outputs.py核對。來源快照、環境版本、原切分與資料雜湊、配對差值CSV均保留。新環境為Apple M2／MPS，不能聲稱精確數值重現原CUDA結果。

統計方法來源：[SciPy Wilcoxon文件](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html)。使用雙側完整符號置換，零差異按wilcox處理（全零p=1），差值先四捨五入至小數12位以避免浮點誤差造成假tie差異。

## 配對結果圖

![Paired seed evidence](e5_supporting_evidence.png)
