import os
import json
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from tqdm import tqdm
# ==========================================
# 1. 設定區 (Config)
# ==========================================
RAW_DATA_PATH = r"C:\Users\user\plaid_project\plaid_2014"
OUTPUT_DIR = r"./dataset"
try:
    from plaid_loader_vi import load_plaid_data
except ImportError:
    print("錯誤：找不到 plaid_loader_vi.py，請確認此程式在正確的目錄下。")
    exit()
# ==========================================
# 2. 存檔函數 (包含修正邏輯)
# ==========================================
def save_data_to_folders(data_list, split_name):
    print(f"正在處理 {split_name} 資料集...")

    skip_count = 0

    for i, item in enumerate(tqdm(data_list)):
        label = str(item['label']).strip()
        img_array = item['image']

        # --- 影像處理 ---
        # 1. 確保是 Numpy 格式
        img_array = np.array(img_array)

        # [關鍵修正] 2. 移除多餘的維度 (例如: (1, 64, 64) -> (64, 64))
        img_array = np.squeeze(img_array)

        # [防呆檢查] 如果 squeeze 後不是 2D (例如變成 1D 或是形狀怪異)，就跳過
        if img_array.ndim != 2:
            # 只有前幾次錯誤會印出來，避免洗版
            if skip_count < 5:
                print(f"\n[警告] 跳過樣本 {i}: 形狀異常 {img_array.shape} (預期應為 2D 矩陣)")
            skip_count += 1
            continue

        # 3. 數值正規化 (0~1 -> 0~255)
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)

        # 4. 轉成圖片物件
        try:
            img = Image.fromarray(img_array)
        except Exception as e:
            print(f"\n[錯誤] 無法轉換圖片 {i}: {e}")
            continue

        # 5. 確保是灰階 (L mode)
        if img.mode != 'L':
            img = img.convert('L')
        # --- 存檔 ---
        target_dir = os.path.join(OUTPUT_DIR, split_name, label)
        os.makedirs(target_dir, exist_ok=True)

        # [新增] 檔名嵌入機台group id，讓後續(pilot切分等)可從檔名還原機台歸屬，
        # 避免在train資料夾內部再切分時，又發生同一台機器樣本跨切分的洩漏。
        safe_group = str(item.get('group', 'unknown')).replace(os.sep, '_').replace(' ', '_')
        save_path = os.path.join(target_dir, f"{safe_group}__{i}.png")
        img.save(save_path)

    if skip_count > 0:
        print(f"注意：共跳過了 {skip_count} 筆形狀異常的資料。")
# ==========================================
# 3. 主程式
# ==========================================
if __name__ == "__main__":
    print("=== 開始製作 dataset 資料夾結構 (修正版 + 最佳比例 Group-Stratified) ===")

    # 1. 讀取
    print(f"正在從 {RAW_DATA_PATH} 讀取原始數據...")
    raw_data = load_plaid_data(RAW_DATA_PATH, limit=None)

    if not raw_data:
        print("錯誤：讀不到任何數據。")
        exit()
    # 2. 切分：自動尋找最接近 80:20 的實體機台組合
    print("正在執行「最佳比例」Group-Stratified 切分 (嚴格逼近 20% 測試樣本)...")

    import random
    random.seed(42)

    train_data = []
    val_data = []
    unique_labels = sorted(list(set([item['label'] for item in raw_data])))
    for lbl in unique_labels:
        lbl_samples = [item for item in raw_data if item['label'] == lbl]
        total_lbl_samples = len(lbl_samples)
        target_val_count = total_lbl_samples * 0.2  # 理想的 20% 測試樣本數

        # 整理這個類別的機台字典 {機台ID: [這台機器的所有樣本清單]}
        machine_dict = {}
        for item in lbl_samples:
            grp = item['group']
            if grp not in machine_dict:
                machine_dict[grp] = []
            machine_dict[grp].append(item)

        machines = list(machine_dict.keys())

        # 尋找最佳機台組合
        best_val_machines = []
        best_diff = float('inf')

        # 讓電腦快速嘗試 2000 種不同的機台分配組合，挑出最完美的那一種
        for _ in range(2000):
            # 隨機挑選至少 1 台、至多 (總台數-1) 台機台作為測試集
            num_val_machines = random.randint(1, max(1, len(machines) - 1))
            val_candidates = random.sample(machines, num_val_machines)

            # 計算這個組合加起來的「總樣本數」
            val_count = sum(len(machine_dict[m]) for m in val_candidates)

            # 計算這個組合與理想 20% 樣本數的誤差
            diff = abs(val_count - target_val_count)

            # 如果這個組合的誤差更小，就記錄下來
            if diff < best_diff:
                best_diff = diff
                best_val_machines = val_candidates

        # 根據找到的最佳組合，將資料正式分發到 Train 與 Val
        for m in machines:
            if m in best_val_machines:
                val_data.extend(machine_dict[m])
            else:
                train_data.extend(machine_dict[m])
    # ---------------------------------------------------------
    # [新增] 驗證 train/val 機台完全不重疊（fail-fast，資料切完立刻檢查）
    train_groups = set(d['group'] for d in train_data)
    val_groups = set(d['group'] for d in val_data)
    overlap = train_groups & val_groups
    assert len(overlap) == 0, f"[嚴重錯誤] train/val 機台重疊，共 {len(overlap)} 台: {overlap}"
    print(f"✓ 驗證通過：train {len(train_groups)} 台、val {len(val_groups)} 台，完全不重疊\n")
    # ---------------------------------------------------------
    stats = []
    for lbl in unique_labels:
        t_data = [d for d in train_data if d['label'] == lbl]
        v_data = [d for d in val_data if d['label'] == lbl]

        t_machines = len(set(d['group'] for d in t_data))
        v_machines = len(set(d['group'] for d in v_data))
        t_samples = len(t_data)
        v_samples = len(v_data)

        total_samples = t_samples + v_samples
        test_ratio = (v_samples / total_samples * 100) if total_samples > 0 else 0

        stats.append({
            "電器類別": lbl,
            "訓練機台數": t_machines,
            "測試機台數": v_machines,
            "訓練樣本數": t_samples,
            "測試樣本數": v_samples,
            "實際測試比例": f"{test_ratio:.1f}%"
        })
    # 計算總計
    total_t_machines = len(set(d['group'] for d in train_data))
    total_v_machines = len(set(d['group'] for d in val_data))
    total_t_samples = len(train_data)
    total_v_samples = len(val_data)
    total_test_ratio = (total_v_samples / (total_t_samples + total_v_samples)) * 100

    stats.append({
        "電器類別": "總計",
        "訓練機台數": total_t_machines,
        "測試機台數": total_v_machines,
        "訓練樣本數": total_t_samples,
        "測試樣本數": total_v_samples,
        "實際測試比例": f"{total_test_ratio:.1f}%"
    })
    # 印出完美表格
    df_stats = pd.DataFrame(stats)
    print("\n" + "="*70)
    print(" 表2. 最佳比例 Group-Stratified切分後,各類別實際訓練/測試組成")
    print("="*70)
    print(df_stats.to_string(index=False))
    print("="*70 + "\n")
    # ---------------------------------------------------------
    # 3. 清理舊資料與存檔
    if os.path.exists(OUTPUT_DIR):
        print(f"清理舊資料夾 {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)

    # [新增] 保存本次切分結果（機台 ID 清單），供版控與可重現性稽核
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    split_record = {
        "random_seed": 42,
        "train_appliance_count": len(train_groups),
        "val_appliance_count": len(val_groups),
        "train_appliance_ids": sorted(train_groups),
        "val_appliance_ids": sorted(val_groups),
    }
    split_record_path = os.path.join(OUTPUT_DIR, "split_record.json")
    with open(split_record_path, "w", encoding="utf-8") as f:
        json.dump(split_record, f, ensure_ascii=False, indent=2)
    print(f"✓ 切分紀錄已存至 {split_record_path}")

    save_data_to_folders(train_data, 'train')
    save_data_to_folders(val_data, 'val')

    print("\n 資料集製作完成")
