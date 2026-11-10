# E14 開跑前回報

2026-11-09 Asia/Taipei。正式訓練前已向作者回報。

- 原runner：`Code/RevisedCode/experiments/run_revision_followup_v4.py`，原引擎`revision_followup_core.py`。protocol_v4各source SHA全部通過。
- E14隔離入口：`source/run_e14.py`，隔離引擎`source/e14_core.py`；差異`source/core_changes.diff`。既有程式不改寫。
- NoiseFB pilot seed42：best epoch42，stop72，validation Macro-F1 .6955289342；fixed_epochs42。
- PermFB pilot seed42：best epoch49，stop79，validation Macro-F1 .7085294634；fixed_epochs49。
- min30／patience30／max150／delta .001，最早30輪起選最高clean validation F1。舊v4 No-Defense patience20；E14依新規格30，非完全相同的pilot patience。
- 正式各10 seed42–51，855 train／219 test，共20模型，lr .0005／WD0／dropout.5／Adam／batch32／MPS。
- 前測純訓練外推1416.52秒約24分鐘；同train資料219筆推論外推約15分鐘，保守含成像IO／統計為20–30分鐘。研究約45–60分鐘，含Word／QA預估60–90分鐘，非完成承諾。
- 評估所有E4條件：乾淨＋210污染、10實現。同樣本ID同污染公式。新模型／Yu3／ZeroFB之eval batch219，ShuffleFB保留原batch32循環替換；BN eval無批次統計更新。新增模型與batch32預測一致另查train-only benchmark；舊模型全部原生預測與保存E4逐值核對。
- Noise population std correction=0；CPU torch.Generator產生亂數後送MPS；固定SHA256映射，不用Python不穩定hash。train key(seed,epoch,batch)，eval key(sampleID)。
- 未在選epoch前讀取E14 test結果。主要端點pass1 SP5/10/20，每強度內NoiseFB對Yu3／ZeroFB兩項精確雙尾Wilcoxon＋Holm；PermFB相同但另列次要；ShuffleFB其餘比較探索性未校正。

## Runner provenance correction after deeper chronology audit

Old ZeroFB/ShuffleFB No-Defense checkpoints were produced on 11/06 by initial run_revision_followup.py and the identical shared revision_followup_core.py, before v4 was created. Final active queue was v4; nested pilot/final functions are AST-identical between v1 and v4. Old ZeroFB32 / ShuffleFB36 epochs; pilot patience20. E14 continues the same engine via isolated source, following new patience30 specification. This corrects shorthand naming only; no experiment design or training change. Evidence: runner_provenance.json.
