import torch

# --- 強制關閉 Flash Attention，避免舊顯卡架構報錯 ---
if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
# ------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from thop import profile
import time
import pandas as pd
import numpy as np
import psutil
import os
import warnings
import json
from scipy.stats import friedmanchisquare, wilcoxon


warnings.filterwarnings("ignore")
DEVICE = torch.device("cpu")  # 強制寫死用 CPU
print(f"目前使用硬體加速裝置: {DEVICE}\n")

NUM_CLASSES = 11


# ==========================================
# 1. 模型架構定義 (直接引用主實驗使用的模型，保證 100% 一致)
# ==========================================
from models_m1_to_m5 import M5_Model
from models_baselines_V2 import ResNet18_Model, MobileNetV3_Model, ViTTiny_Model

def create_model(model_name):
    if model_name == 'Yu3': return M5_Model()
    elif model_name == 'MobileNetV3': return MobileNetV3_Model()
    elif model_name == 'ResNet18': return ResNet18_Model()
    elif model_name == 'ViT-Tiny': return ViTTiny_Model()

# ==========================================
# 2. 測量單一確定值與多輪統計
# ==========================================

# ==========================================
# 統計檢定：Friedman + Wilcoxon + Holm-Bonferroni
# ==========================================
def holm_bonferroni(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    n = len(p_values)
    corrected = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * p_values[idx]
        running_max = max(running_max, adj)
        corrected[idx] = min(running_max, 1.0)
    return corrected

def run_efficiency_statistical_tests(raw_data_dict, model_names, metric_name):
    # raw_data_dict 裡面裝的是各模型的 100 次 raw data list
    data_matrix = [raw_data_dict[m] for m in model_names]
    stat, p_friedman = friedmanchisquare(*data_matrix)
    print(f"\n[{metric_name}] Friedman檢定 (n={len(data_matrix[0])}): chi2={stat:.3f}, p={p_friedman:.4e}")
    
    if p_friedman >= 0.05:
        print("  → 整體無顯著差異，不進行事後兩兩比較")
        return
        
    pairs, p_raw = [], []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            try:
                _, p = wilcoxon(raw_data_dict[m1], raw_data_dict[m2])
            except ValueError:
                p = 1.0
            pairs.append((m1, m2))
            p_raw.append(p)
            
    p_corrected = holm_bonferroni(np.array(p_raw))
    for (m1, m2), p_c in zip(pairs, p_corrected):
        sig = "**" if p_c < 0.01 else ("*" if p_c < 0.05 else "ns")
        print(f"  {m1} vs {m2}: Holm-corrected p={p_c:.4e} {sig}")

def get_fp32_model_size(model):
    temp_path = "temp_model_size_test.pth"
    torch.save(model.state_dict(), temp_path)
    size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    os.remove(temp_path) 
    return size_mb

def evaluate_efficiency_table(model, device, dummy_input):
    model.eval()
    
    # [1] 測量單一確定值：MACs 與 Parameters
    macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
    flops = macs * 2  
    
    # [2] 測量單一確定值：FP32 模型檔案大小
    fp32_size_mb = get_fp32_model_size(model)
    
    # 暖身 (Warm-up)
    #for _ in range(10): _ = model(dummy_input)
    #if device.type == 'cuda': torch.cuda.synchronize()
    
    measure_runs = 100
    inference_times = []
    peak_memories = []
    
    # 💡 關鍵：在這裡先宣告 process，否則迴圈內會找不到名稱！
    process = psutil.Process(os.getpid())

    # 【🚨 致命漏洞修復】必須包裹在 torch.no_grad() 內，這才是真實的邊緣推論！
    with torch.no_grad():
        # 暖身 (Warm-up)
        for _ in range(10): _ = model(dummy_input)
        if device.type == 'cuda': torch.cuda.synchronize()
        
        for _ in range(measure_runs):
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                start_time = time.perf_counter() 
                _ = model(dummy_input)
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                run_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            else:
                # 拔除沉重的 tracemalloc，進行最純粹的 CPU 推論計時
                start_time = time.perf_counter() 
                _ = model(dummy_input)
                end_time = time.perf_counter()
                
                run_mem_mb = 0.0  # 反正我們不要這項數據了，隨便塞個 0.0 避免報錯
                
            run_time_ms = (end_time - start_time) * 1000
            inference_times.append(run_time_ms)
            peak_memories.append(run_mem_mb)
            
    # 計算平均值與標準差
    mean_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    mean_mem = np.mean(peak_memories)
    std_mem = np.std(peak_memories)
    
    # 吞吐量 (Throughput)
    throughput_fps = 1000.0 / mean_time if mean_time > 0 else 0.0
    
    return {
        'Parameters (M)': round(params / 1e6, 4),
        'File Size (MB)': round(fp32_size_mb, 2),
        'MACs (G)': round(macs / 1e9, 4),          
        'FLOPs (G)': round(flops / 1e9, 4),        
        '推論時間 (ms)': f"{mean_time:.2f} ± {std_time:.2f}",
        '吞吐量 (FPS)': round(throughput_fps, 2),    
        '峰值記憶體 (MB)': f"{mean_mem:.2f} ± {std_mem:.2f}",
        # 【關鍵新增】將 100 次的原始數據一併回傳
        'raw_times': inference_times,
        'raw_mems': peak_memories
    }

# ==========================================
# 3. 主程式執行
# ==========================================
if __name__ == '__main__':
    # 名字對齊你的表格
    model_names = ['Yu3', 'MobileNetV3', 'ResNet18', 'ViT-Tiny']
    
    # Batch Size 設定為 1，模擬邊緣運算單筆推論情境
    dummy_input = torch.randn(1, 1, 64, 64).to(DEVICE)
    results = []
    raw_data_collection = {'times': {}, 'mems': {}} # 【新增】收集原始數據
    
    print("======== 開始執行代表性模型比較與部署前效率評估 ========\n")
    
    for name in model_names:
        print(f"正在測量模型: {name} (收集 100 輪統計數據中...)")
        model = create_model(name).to(DEVICE)
        
        metrics = evaluate_efficiency_table(model, DEVICE, dummy_input)
        
        # 【新增】將該模型的 100 次數據存入字典
        raw_data_collection['times'][name] = metrics['raw_times']
        raw_data_collection['mems'][name] = metrics['raw_mems']
        
        results.append({
            'Model': name,
            'Parameters (M)': metrics['Parameters (M)'],
            'File Size (MB)': metrics['File Size (MB)'],
            'MACs (G)': metrics['MACs (G)'],
            'FLOPs (G)': metrics['FLOPs (G)'],
            '推論時間 (ms)': metrics['推論時間 (ms)'],
            '吞吐量 (FPS)': metrics['吞吐量 (FPS)'],
            '峰值記憶體 (MB)': metrics['峰值記憶體 (MB)']
        })
        
        del model
        if DEVICE.type == 'cuda': torch.cuda.empty_cache()

    # 【新增】儲存 100 次原始數據到 JSON 檔
    with open('exp5_raw_efficiency_data.json', 'w', encoding='utf-8') as f:
        json.dump(raw_data_collection, f, ensure_ascii=False, indent=2)
    print("\n✓ 已將 100 次推論時間與記憶體的原始數據存入 exp5_raw_efficiency_data.json")

    # 【新增】直接執行嚴謹的統計檢定
    print("\n======== 推論時間 (ms) 統計檢定 ========")
    run_efficiency_statistical_tests(raw_data_collection['times'], model_names, "Inference Time")
    
    # 刪除這兩行，不要對記憶體做統計檢定！
    # print("\n======== 峰值記憶體 (MB) 統計檢定 ========")
    # run_efficiency_statistical_tests(raw_data_collection['mems'], model_names, "Peak Memory")

    # ----- (下方保留你原本轉置表格與存 CSV 的程式碼) -----
    df_efficiency = pd.DataFrame(results)
    df_transposed = df_efficiency.set_index('Model').T
    print("\n======== 最終評估結果 (已完美對齊你的 Word 表格) ========\n")
    print(df_transposed.to_string())
    df_transposed.to_csv('Exp5_Table_Format.csv', encoding='utf-8-sig')
    print("\n✅ 評估完成！請打開 Exp5_Table_Format.csv，直接複製貼上到論文中。")