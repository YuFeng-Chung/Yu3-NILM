import os
import shutil
import json
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from plaid_loader_vi import load_plaid_data

# ==========================================
# 參數設定
# ==========================================
RAW_DATA_PATH = r"C:\Users\user\plaid_project\plaid_2014"
OUT_DIR = r"./dataset_random"
SEEDS = list(range(42, 52))

def save_split_data(items, split_dir):
    for i, it in enumerate(items):
        img = np.squeeze(np.array(it["image"]))
        if img.ndim != 2:
            continue
            
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        
        label_dir = os.path.join(split_dir, str(it["label"]).strip())
        os.makedirs(label_dir, exist_ok=True)
        
        group_id = str(it["group"]).replace(os.sep, "_").replace(" ", "_")
        save_path = os.path.join(label_dir, f"{group_id}__{i}.png")
        Image.fromarray(img, mode="L").save(save_path)

if __name__ == "__main__":
    print("=== 開始製作「隨機切分 (Random Split)」資料集並計算洩漏率 ===")
    
    # 讀取原始數據並過濾異常維度
    raw_data = [s for s in load_plaid_data(RAW_DATA_PATH, limit=None) if np.squeeze(np.array(s["image"])).ndim == 2]
    labels = np.array([s["label"] for s in raw_data])
    groups = np.array([s["group"] for s in raw_data])
    
    os.makedirs(OUT_DIR, exist_ok=True)
    stats = []
    
    for seed in SEEDS:
        print(f"處理 Seed {seed}...")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, test_idx = next(sss.split(np.zeros(len(raw_data)), labels))
        
        base_path = os.path.join(OUT_DIR, f"seed{seed}")
        shutil.rmtree(base_path, ignore_errors=True)
        
        # 存檔
        save_split_data([raw_data[i] for i in train_idx], os.path.join(base_path, "train"))
        save_split_data([raw_data[i] for i in test_idx], os.path.join(base_path, "val"))
        
        # 計算資料洩漏率：測試集中的機台，有多少比例曾出現在訓練集
        train_groups = np.unique(groups[train_idx])
        test_groups = groups[test_idx]
        leakage_mask = np.isin(test_groups, train_groups)
        leakage_pct = float(leakage_mask.mean() * 100)
        
        stats.append({
            "seed": seed, 
            "n_train": int(len(train_idx)), 
            "n_test": int(len(test_idx)),
            "test_from_seen_appliances_pct": leakage_pct
        })
        
    stats_path = os.path.join(OUT_DIR, "leakage_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        
    print(f"✓ 已產出 10 組隨機切分資料集，洩漏統計已存至 {stats_path}")