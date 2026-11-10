"""Build E8 reading report from independently audited measurements; no measurement."""
import csv
import html
import json
from pathlib import Path

ROOT=Path.cwd()/'Code/RevisedCode'
OUT=ROOT/'results/e8_cpu_20261108'

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
    p=json.loads((OUT/'protocol.json').read_text())
    a=json.loads((OUT/'measurement_audit.json').read_text())
    independent=json.loads((OUT/'independent_audit.json').read_text())
    s=json.loads((OUT/'latency_summary.json').read_text())
    m=json.loads((OUT/'memory_summary.json').read_text())
    assert independent['all_checks_passed']
    lines=['# E8 電腦端效率量測報告','', '日期：2026-11-08（Asia/Taipei）。本報告為本Mac CPU量測交付，不代表原稿及逐題審稿回覆已修訂。', '',
    '## 完成與稽核', '',
    f"四組runtime／threads、20個設定、{independent['accepted_calls']:,}筆正式有效逐次計時；20個獨立記憶體程序。獨立原始重算通過。量測穩定性稽核 all_checks_passed={a['all_checks_passed']}，漂移旗標={len(a['drift_flags'])}。所有被拒輪次仍保存，非依速度刪除離群值。", '',
    '## 硬體與方法', '',
    f"Apple M2、24 GiB；Python {p['versions']['python']}、PyTorch {p['versions']['torch']}、ONNX Runtime {p['versions']['onnxruntime']}，CPUExecutionProvider。CPU intra-op=1或4、interop=1，eval與inference_mode；固定FP32、batch=1、1×64×64。ORT採sequential、停用intra-op spinning、預設圖最佳化。兩runtime設定分段執行，未跨runtime隨機交錯；因此跨runtime差異亦可能受時段影響。", '',
    '沿用四模型S3 seed42的已訓練checkpoint，不重訓、不依test挑選。合成16張uniform[0,1]輸入固定循環使用，不含V–I建圖、磁碟I/O、前處理與資料傳輸。此為固定幾何的模型計算成本測試，非真實資料端到端延遲。', '',
    '每條路徑50次暖機，每個區塊另3次暖機，30個交錯輪次×每模型100次。每5輪各模型在五個位置均各出現一次，六個cycle採固定seed重排。報告保留逐次median／p95及區塊median平均與SD，重複計時不當成獨立訓練seed或資料集樣本，不作顯著性檢定。', '',
    '正式輪次依預先凍結規則檢查接電及外部CPU負載：整機比例平均≤15%、最高≤35%；不合格整輪拒收，保留原始嘗試，每設定最多額外10輪。外部CPU以整機使用量減量測程序使用量估計，不保證完全隔離OS排程。last10／first10區塊median比值超過1.2或低於1/1.2標記漂移，不以速度結果追刪資料。macOS未提供本次可用的溫度實測，pmset無已記錄警告不等於溫控已獨立驗證。', '',
    'PyTorch／ONNX以1-thread各16張、4-thread各4張輸入核對，atol=rtol=1e-4，所有argmax一致；這只驗證固定輸入範圍內的數值一致，不是全資料集準確度再驗證。', '',
    '## 延遲（毫秒）', '',
    table(['Runtime','Threads','路徑','逐次median','逐次p95','區塊median平均 ± SD','末／初比','漂移'],[
        [r['runtime'],r['threads'],r['path'],f"{r['median_ms']:.4f}",f"{r['p95_ms']:.4f}",f"{r['block_median_mean_ms']:.4f} ± {r['block_median_sd_ms']:.4f}",f"{r['last10_first10_ratio']:.3f}",r['drift_flag']] for r in s]),'',
    '## Yu3 第二pass額外成本', '',
    'Pass1_only使用同一模型，執行初次encoder與classifier，跳過decoder與第二pass。完整減Pass1包含重建、殘差、第二次encoder與classifier，是端到端額外路徑成本，不是孤立第二次encoder的微量測；跨區塊差值也受排程影響。ORT可裁掉不影響最終logits的初次classifier；PyTorch eager仍會執行，因此runtime成本邊界明列。', '']
    if (OUT/'rerun_amendment.json').exists():
        amendment=json.loads((OUT/'rerun_amendment.json').read_text())
        lines += ['## 追加重測與透明揭露', '', f"初批PyTorch四執行緒在40次嘗試後僅有{amendment['initial_accepted_rounds']}個有效輪次，原因為預定CPU負載規則拒收，未依延遲／模型排序決定。原檔initial_incomplete_timing_pytorch_4.json完整保留，另提供initial_incomplete_latency_summary.json。其後新增一次完整整組重測，在連續30秒CPU≤15%後使用相同30輪、每輪100次、交錯順序與最多40次嘗試。這是量測開始後的排程修訂，記於rerun_amendment.json，不冒稱原先已預註冊。主要60,000筆摘要採完整重測時段；初批不完整時段另列，不混成同一輪次或只選較快數據。", '']
    extra=[]
    for runtime,threads in p['setting_order']:
        full=next(r for r in s if (r['runtime'],r['threads'],r['path'])==(runtime,threads,'Yu3'))
        first=next(r for r in s if (r['runtime'],r['threads'],r['path'])==(runtime,threads,'Yu3_pass1_only'))
        extra.append([runtime,threads,f"{full['mean_ms']-first['mean_ms']:.4f}",f"{full['mean_ms']/first['mean_ms']:.3f}"])
    lines += [table(['Runtime','Threads','完整−Pass1平均ms','完整／Pass1平均'],extra),'', '## 參數與檔案', '',
        '以下為已載入完整模型的總參數及tensor storage；Pass1沿用完整模型，decoder雖不執行仍保留權重，不能把總參數當成Pass1活躍參數。checkpoint包含研究metadata，ONNX為匯出後計算圖，兩檔大小的差異不等於壓縮率。', '',
        table(['路徑','完整模型總參數','參數MiB','state tensors MiB','ONNX MiB','ONNX節點'],[[n,v['total_architecture_parameters'],f"{v['parameter_bytes']/2**20:.3f}",f"{v['state_tensor_bytes']/2**20:.3f}",f"{v['onnx_bytes']/2**20:.3f}",v['onnx_nodes']] for n,v in p['models'].items()]),'',
        '## 記憶體（MiB）', '',
        '每一runtime／threads／路徑使用新程序；baseline為共同匯入PyTorch及ORT後的RSS，load包含模型載入及保留的配置器記憶體，warmup後RSS是當下觀測。peak為macOS getrusage程序全生命週期RSS高水位，含import與載入暫存，不是推論activation峰值。未量測系統總記憶體，沒有把0當成缺失值。每設定只有一次程序，故只做描述，不提供虛構SD。', '',
        table(['Runtime','Threads','路徑','Baseline','載入後','暖機後','載入增量','Lifetime peak'],[[r['runtime'],r['threads'],r['path']]+[f"{r[k]/2**20:.2f}" for k in ['baseline_rss_bytes','loaded_rss_bytes','after_warmup_rss_bytes','load_delta_rss_bytes','peak_lifetime_rss_bytes']] for r in m]),'',
        '## 解讀邊界與來源', '',
        '結果只限此Mac、CPU、固定輸入與指定runtime／threads；不宣稱嵌入式、edge deployment、耗能或其他電腦上的排序。未新做MACs／FLOPs、CUDA／MPS或訓練速度，也未由這次時間量測推論F1優劣；這些不能標成已由E8解決。準確度與效率取捨圖、正式英中審稿回覆及Word稿仍可另依原始證據整理，原投稿未修改。', '',
        '`protocol.json`為量測前凍結設定及版本／checkpoint／source SHA；`timing_*.json`含全部accepted／rejected輪次及CPU紀錄；`memory_*.json`為獨立程序記錄；`latency_summary.csv/json`、`memory_summary.json`為摘要；`independent_audit.json`獨立重算所有60,000筆、20摘要及記憶體算術；`measurement_audit.json`另列漂移。`additional_parity_4threads.json`記錄4-thread一致性。ONNX圖及合成輸入一併保留；checkpoint仍在前輪來源。', '',
        '估時前測位於另一個`results/e8_schedule_20261108/`，未混入本次正式數據。']
    (OUT/'E8電腦端效率量測報告.md').write_text('\n'.join(lines)+'\n')
    with (OUT/'raw_accepted_timings.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['runtime','threads','round','attempt','path','position','call','latency_ns'])
        for runtime,threads in p['setting_order']:
            data=json.loads((OUT/f'timing_{runtime}_{threads}.json').read_text())
            for attempt in data['attempts']:
                if attempt['accepted']:
                    for block in attempt['blocks']:
                        for i,ns in enumerate(block['ns']):w.writerow([runtime,threads,attempt['round_index'],attempt['attempt'],block['path'],block['position'],i,ns])
    print('report and raw CSV saved')

if __name__=='__main__':main()
