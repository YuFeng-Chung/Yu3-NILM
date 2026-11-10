import os
import json
import numpy as np
import pandas as pd
import cv2

# ==========================================
# 1. 數據增強函數 (新增)
# ==========================================
def augment_data(current, voltage):
    """
    數據增強：隨機平移與縮放
    """
    # 1. 隨機縮放 (0.9 ~ 1.1倍)
    scale = np.random.uniform(0.9, 1.1)
    current = current * scale
    
    # 2. 隨機平移 (-20 ~ +20 點)
    shift = np.random.randint(-20, 20)
    if shift > 0:
        current = np.pad(current, (shift, 0), 'constant')[:len(current)]
        voltage = np.pad(voltage, (shift, 0), 'constant')[:len(voltage)]
    else:
        current = np.pad(current, (0, -shift), 'constant')[-len(current):]
        voltage = np.pad(voltage, (0, -shift), 'constant')[-len(voltage):]
        
    # 3. 加入微量雜訊 (Gaussian Noise)
    noise = np.random.normal(0, 0.05, current.shape)
    current = current + noise
    
    return current, voltage

# ==========================================
# 2. 輔助工具函數
# ==========================================
def get_activation_window(current, voltage, window_len=500):
    diff = np.diff(current)
    event_idx = np.argmax(np.abs(diff))
    start = event_idx - 100
    end = start + window_len
    
    if start < 0:
        start = 0
        end = window_len
    if end > len(current):
        end = len(current)
        start = end - window_len
        if start < 0:
            pad_len = window_len - len(current)
            current = np.pad(current, (0, pad_len), 'constant')
            voltage = np.pad(voltage, (0, pad_len), 'constant')
            return current, voltage

    return current[start:end], voltage[start:end]

def generate_vi_image(current, voltage, size=64):
    current = np.nan_to_num(current)
    voltage = np.nan_to_num(voltage)
    
    # 標準化
    if np.std(current) > 1e-5:
        current = (current - np.mean(current)) / np.std(current)
    if np.std(voltage) > 1e-5:
        voltage = (voltage - np.mean(voltage)) / np.std(voltage)

    v_norm = ((voltage + 3) / 6 * (size - 1)).astype(int)
    i_norm = ((current + 3) / 6 * (size - 1)).astype(int)
    
    v_norm = np.clip(v_norm, 0, size-1)
    i_norm = np.clip(i_norm, 0, size-1)
    
    img = np.zeros((size, size), dtype=np.float32)
    for i, v in zip(i_norm, v_norm):
        img[i, v] += 0.1 
    
    img = np.clip(img, 0, 1.0)
    return img[np.newaxis, :, :]

# ==========================================
# 3. 主讀取函數 (含增強邏輯)
# ==========================================
def load_plaid_data(data_dir, meta_name="meta_2014.json", img_size=64, limit=None):
    meta_path = os.path.join(data_dir, meta_name)
    if not os.path.exists(meta_path):
        print(f"Meta file not found: {meta_path}")
        return []

    with open(meta_path, "r", encoding="utf-8") as f:
        meta_list = json.load(f)

    id2meta = {}
    for item in meta_list:
        sid = str(item.get("id"))
        meta = item.get("meta", item)
        appliance = meta.get("appliance", {})
        atype = appliance.get("type")
        
        # [關鍵正確修正]：使用 location (房屋地點) 來作為機台的綁定基準
        location = meta.get("location", "unknown")
        
        # 實體機器的唯一 ID = 電器種類 + 房屋地點
        group_id = f"{atype}_house{location}"
        
        id2meta[sid] = {"type": atype, "group": group_id}

    samples = []
    csv_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
    
    print(f"Found {len(csv_files)} csv files. Processing with Augmentation...")
    
    count = 0
    for fname in csv_files:
        if limit and count >= limit: break
        
        sid = os.path.splitext(fname)[0]
        if sid not in id2meta: continue
            
        csv_path = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(csv_path, header=None)
            if df.shape[1] < 2: continue
                
            current = df.iloc[:, 0].astype(float).values
            voltage = df.iloc[:, 1].astype(float).values
            
            # 1. 對齊事件
            c_crop, v_crop = get_activation_window(current, voltage, window_len=500)
            
            # 2. 生成原始 VI 圖
            vi_img = generate_vi_image(c_crop, v_crop, size=img_size)
            
            # 原始波形標準化
            c_crop_norm = c_crop.copy()
            if np.std(c_crop_norm) > 1e-5:
                c_crop_norm = (c_crop_norm - np.mean(c_crop_norm)) / np.std(c_crop_norm)

            # === 加入原始數據 ===
            sample = {
                "label": id2meta[sid]["type"],
                "group": id2meta[sid]["group"], # 新增：記錄該筆數據屬於哪台實體機器
                "wave": c_crop_norm[np.newaxis, :].astype(np.float32), 
                "image": vi_img
            }
            samples.append(sample)

            # # === 您的程式碼插入位置：新增 2 筆增強數據 ===
            # for _ in range(2): 
            #     # 這裡傳入 copy() 很重要，避免改到原始數據
            #     c_aug, v_aug = augment_data(c_crop.copy(), v_crop.copy())
            #     vi_aug = generate_vi_image(c_aug, v_aug, size=img_size)
                
            #     # 標準化
            #     if np.std(c_aug) > 1e-5:
            #         c_aug = (c_aug - np.mean(c_aug)) / np.std(c_aug)
                
            #     sample_aug = {
            #         "label": id2meta[sid]["type"],
            #         "group": id2meta[sid]["group"], # 增強數據也要繼承相同的機器標籤
            #         "wave": c_aug[np.newaxis, :].astype(np.float32),
            #         "image": vi_aug
            #     }
            #     samples.append(sample_aug)
            # # ============================================

            count += 1
            
        except Exception as e:
            pass

    print(f"Loaded {len(samples)} samples (Original + Augmented).")
    return samples