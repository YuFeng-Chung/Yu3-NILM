"""
Exp4 專用資料管線：從原始CSV波形出發，而不是從Exp1已經產生好的PNG圖檔出發。
==========================================================================
這是根據手稿(Materials and Methods 第(3)節「雜訊與訊號擾動穩健性測試」)修正後
最關鍵的架構調整，跟一開始我寫的post-hoc在PNG上加雜訊的版本完全不同，原因如下：

  1. 手稿明確指出：所有擾動(AWGN/Salt-Pepper/隨機缺失/連續缺失)都必須加在
     「500點重取樣後的原始電壓、電流波形上」，且是在正規化與V-I軌跡影像建構
     『之前』完成，而不是加在已經畫好的64x64圖片上。所以Exp4不能只在Exp1
     產生的PNG上做手腳，必須回到原始CSV重新算。

  2. 手稿明確指出：「正規化時僅使用由訓練資料計算所得並預先固定的統計參數」——
     也就是說，Exp4的Z-score不能像Exp1(plaid_loader_vi.py的generate_vi_image())
     一樣「每個樣本各自用自己的mean/std做normalize」，而要改成「用整個訓練集
     算出來、固定住的一組全域mean/std」。原因：如果還是用「每個樣本自己的
     mean/std」，那麼加了雜訊之後，正規化步驟會把雜訊的能量也一起吸收進
     mean/std裡，等於normalize這一步會「自動幫忙稀釋掉部分雜訊強度」，導致
     不同樣本、不同嚴重度之間的「有效雜訊強度」不可比——這正是手稿要求改用
     固定統計量的理由。

  3. 椒鹽雜訊的上下限同理，要用「訓練資料所定義之有效量測範圍」預先固定，
     不能用test資料重新估計。

跟Exp1共用的部分（不重新發明）：
  - 機台層級的80:20 group-stratified切分：直接讀取make_dataset_split.py
    產生的 dataset/split_record.json，裡面已經存好 train_appliance_ids /
    val_appliance_ids，Exp4直接照著這份清單把原始CSV分成train/test，
    確保跟Exp1「完全相同的機台層級切分」(手稿第214行的要求)。
  - CSV讀取、暫態事件偵測、500點窗口擷取的邏輯：直接複用
    plaid_loader_vi.py 的 get_activation_window()，避免另外維護一份漂移。
"""

import os
import json

import numpy as np
import pandas as pd

from plaid_loader_vi import get_activation_window  # 沿用Exp1同一套事件偵測+窗口擷取邏輯

RAW_DATA_PATH = r"C:\Topics\plaid2014"       # 跟make_dataset_split.py一致
SPLIT_RECORD_PATH = r"C:\Users\User\plaid_project\實驗一程式碼\dataset\split_record.json"
META_NAME = "meta_2014.json"
WINDOW_LEN = 500


# ==========================================
# 1. 讀取原始（未正規化）波形
# ==========================================
def load_plaid_raw_samples(data_dir=RAW_DATA_PATH, meta_name=META_NAME, limit=None):
    """回傳list of dict：{'label', 'group', 'current', 'voltage'}，
    current/voltage是500點、『尚未』做任何正規化的原始波形。
    邏輯跟plaid_loader_vi.load_plaid_data()前半段完全一致，只是在
    get_activation_window()之後就停手，不做Z-score、不建VI圖。"""
    meta_path = os.path.join(data_dir, meta_name)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta_list = json.load(f)

    id2meta = {}
    for item in meta_list:
        sid = str(item.get("id"))
        meta = item.get("meta", item)
        appliance = meta.get("appliance", {})
        atype = appliance.get("type")
        location = meta.get("location", "unknown")
        group_id = f"{atype}_house{location}"
        id2meta[sid] = {"type": atype, "group": group_id}

    samples = []
    csv_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
    count = 0
    for fname in csv_files:
        if limit and count >= limit:
            break
        sid = os.path.splitext(fname)[0]
        if sid not in id2meta:
            continue
        try:
            df = pd.read_csv(os.path.join(data_dir, fname), header=None)
            if df.shape[1] < 2:
                continue
            current = df.iloc[:, 0].astype(float).values
            voltage = df.iloc[:, 1].astype(float).values
            c_crop, v_crop = get_activation_window(current, voltage, window_len=WINDOW_LEN)

            samples.append({
                "label": id2meta[sid]["type"],
                "group": id2meta[sid]["group"],
                "current": c_crop.astype(np.float32),   # 原始值，未正規化
                "voltage": v_crop.astype(np.float32),
            })
            count += 1
        except Exception:
            pass

    print(f"[exp4_raw_data] 已讀取 {len(samples)} 筆原始（未正規化）波形樣本")
    return samples


# ==========================================
# 2. 依Exp1的機台切分紀錄，切成train/test
# ==========================================
def split_by_exp1_record(raw_samples, split_record_path=SPLIT_RECORD_PATH):
    """讀取make_dataset_split.py存的split_record.json，用裡面的
    train_appliance_ids / val_appliance_ids把raw_samples分成兩份，
    確保跟Exp1使用『完全相同』的機台層級切分（手稿214行的要求），
    而不是在Exp4重新做一次隨機切分。"""
    with open(split_record_path, "r", encoding="utf-8") as f:
        record = json.load(f)
    train_ids = set(record["train_appliance_ids"])
    val_ids = set(record["val_appliance_ids"])

    train_samples = [s for s in raw_samples if s["group"] in train_ids]
    test_samples = [s for s in raw_samples if s["group"] in val_ids]

    unmatched = [s for s in raw_samples if s["group"] not in train_ids and s["group"] not in val_ids]
    if unmatched:
        print(f"[警告] {len(unmatched)} 筆樣本的group不在split_record.json的清單內，已略過。"
              f"請確認RAW_DATA_PATH與make_dataset_split.py用的是同一批原始資料。")

    print(f"[exp4_raw_data] train={len(train_samples)}筆／{len(train_ids)}台機台，"
          f"test={len(test_samples)}筆／{len(val_ids)}台機台")
    return train_samples, test_samples


# ==========================================
# 3. 固定的（訓練集算出來的）正規化統計量
# ==========================================
def compute_fixed_norm_stats(train_raw_samples):
    """把訓練集所有樣本的current/voltage全部攤平pooled在一起，
    算出一組全域mean/std（供Z-score用）與min/max（供salt-pepper上下限用）。
    這組統計量算完就『凍結』，Exp4後續所有VI影像建構、椒鹽雜訊上下限，
    不論是clean還是任何corruption條件、不論train或test，一律套用同一組。"""
    all_current = np.concatenate([s["current"] for s in train_raw_samples])
    all_voltage = np.concatenate([s["voltage"] for s in train_raw_samples])

    stats = {
        "i_mean": float(np.mean(all_current)), "i_std": float(np.std(all_current)),
        "v_mean": float(np.mean(all_voltage)), "v_std": float(np.std(all_voltage)),
        "i_min": float(np.min(all_current)), "i_max": float(np.max(all_current)),
        "v_min": float(np.min(all_voltage)), "v_max": float(np.max(all_voltage)),
    }
    print(f"[exp4_raw_data] 固定正規化統計量（僅由訓練集算出）: {stats}")
    return stats


# ==========================================
# 4. 用固定統計量建VI影像（取代plaid_loader_vi.generate_vi_image()
#    裡「每個樣本各自算自己mean/std」的版本）
# ==========================================
def _zscore_fixed(x, mean, std):
    if std > 1e-5:
        return (x - mean) / std
    return x - mean


def _scatter_vi_image(current_z, voltage_z, size=64):
    """跟plaid_loader_vi.generate_vi_image()後半段（已標準化之後的部分）邏輯相同，
    直接複用同一套(v+3)/6映射到[0,size-1]再累加的做法，維持跟Exp1一致的視覺化定義。"""
    v_norm = ((voltage_z + 3) / 6 * (size - 1)).astype(int)
    i_norm = ((current_z + 3) / 6 * (size - 1)).astype(int)
    v_norm = np.clip(v_norm, 0, size - 1)
    i_norm = np.clip(i_norm, 0, size - 1)

    img = np.zeros((size, size), dtype=np.float32)
    for i, v in zip(i_norm, v_norm):
        img[i, v] += 0.1
    img = np.clip(img, 0, 1.0)
    return img[np.newaxis, :, :]


def generate_vi_image_fixed_norm(current, voltage, stats, size=64):
    """Exp4版的VI影像建構：用compute_fixed_norm_stats()算出的固定統計量做Z-score，
    而不是像Exp1一樣每個樣本自己算mean/std。"""
    current = np.nan_to_num(current)
    voltage = np.nan_to_num(voltage)
    c_z = _zscore_fixed(current, stats["i_mean"], stats["i_std"])
    v_z = _zscore_fixed(voltage, stats["v_mean"], stats["v_std"])
    return _scatter_vi_image(c_z, v_z, size=size)


# ==========================================
# 5. 波形層級的擾動函式（AWGN / Salt-Pepper / 隨機缺失 / 連續缺失）
#    ⚠️ 全部依「表6」定義，加在current/voltage原始波形上，尚未正規化之前
# ==========================================
def add_awgn_waveform(current, voltage, snr_db, rng):
    """公式(1)-(3)：per-channel獨立產生AWGN，用該樣本自己的訊號功率P_x算雜訊變異數
    （這一步不涉及跨樣本統計，所以不是手稿說的洩漏來源；洩漏風險在正規化那一步，
    已經用固定統計量處理掉了）。"""
    def _add(x):
        p = np.var(x.astype(np.float64))
        if p <= 1e-12:
            return x.copy()
        noise_power = p / (10 ** (snr_db / 10.0))
        noise = rng.normal(0.0, np.sqrt(noise_power), size=x.shape)
        return (x + noise).astype(np.float32)
    return _add(current), _add(voltage)


def add_salt_pepper_waveform(current, voltage, amount, stats, rng):
    """依污染比例隨機選點，以相同機率替換為對應通道的上限或下限。
    上下限用compute_fixed_norm_stats()算出的訓練集min/max，不得用test資料重估。"""
    def _add(x, xmin, xmax):
        x = x.copy()
        n = len(x)
        k = int(round(amount * n))
        if k <= 0:
            return x
        idx = rng.choice(n, size=k, replace=False)
        vals = np.where(rng.random(k) < 0.5, xmax, xmin)
        x[idx] = vals
        return x
    return (_add(current, stats["i_min"], stats["i_max"]),
            _add(voltage, stats["v_min"], stats["v_max"]))


def _interpolate_missing(x, mask):
    """線性插值補值；缺失段在頭尾時，np.interp對範圍外的點自動取端點值，
    等同手稿要求的『以最鄰近有效取樣值進行邊界延伸』。"""
    x = x.copy()
    idx = np.arange(len(x))
    valid = ~mask
    if valid.sum() == 0 or valid.sum() == len(x):
        return x
    x[mask] = np.interp(idx[mask], idx[valid], x[valid])
    return x


def add_missing_random_waveform(current, voltage, missing_ratio, rng):
    """隨機資料遺失：電壓電流『同步』在相同時間位置遺失。"""
    n = len(current)
    mask = rng.random(n) < missing_ratio
    return _interpolate_missing(current, mask), _interpolate_missing(voltage, mask)


def add_missing_consecutive_waveform(current, voltage, missing_ratio, rng):
    """連續資料遺失：隨機選一個起點，建立長度=missing_ratio*500的連續缺失區段，
    電壓電流同步遺失同一段時間位置。"""
    n = len(current)
    seg_len = max(1, int(round(missing_ratio * n)))
    seg_len = min(seg_len, n)
    start = int(rng.integers(0, max(1, n - seg_len + 1)))
    mask = np.zeros(n, dtype=bool)
    mask[start:start + seg_len] = True
    return _interpolate_missing(current, mask), _interpolate_missing(voltage, mask)


CORRUPTION_FUNCS = {
    "awgn": add_awgn_waveform,
    "salt_pepper": add_salt_pepper_waveform,
    "missing_random": add_missing_random_waveform,
    "missing_consecutive": add_missing_consecutive_waveform,
}


def apply_waveform_corruption(current, voltage, corruption_type, severity, stats, rng):
    """統一介面。corruption_type='clean'時原樣回傳。"""
    if corruption_type == "clean":
        return current.copy(), voltage.copy()
    if corruption_type == "salt_pepper":
        return add_salt_pepper_waveform(current, voltage, severity, stats, rng)
    func = CORRUPTION_FUNCS.get(corruption_type)
    if func is None:
        raise ValueError(f"未知的 corruption_type: {corruption_type}")
    return func(current, voltage, severity, rng)


# ==========================================
# 6. 表6：正式的嚴重度網格
#    ⚠️ AWGN「30、20、15、10、5、0 dB」在手稿表格中因排版被拆成兩行文字，
#       這裡採用最合理的還原讀法：六個等級 30/20/15/10/5/0 dB。
#       請學生對照manuscript表6原始版面（非pandoc轉出的markdown）再次確認，
#       如果實際上是其他組合，只要改這個dict即可，其餘程式碼不需要動。
# ==========================================
SEVERITY_GRIDS = {
    "awgn": [30, 20, 15, 10, 5, 0],            # dB，SNR越低雜訊越強
    "salt_pepper": [0.01, 0.05, 0.10, 0.20],   # 受污染取樣點比例
    "missing_random": [0.01, 0.05, 0.10, 0.20],
    "missing_consecutive": [0.01, 0.05, 0.10, 0.20],
}


# ==========================================
# 7. 共用的 PyTorch Dataset：raw waveform → （可選擾動/動態增強）→ 固定正規化VI影像
# ==========================================
import torch
from torch.utils.data import Dataset


def build_label_map(raw_samples):
    labels = sorted(set(s["label"] for s in raw_samples))
    return {lbl: i for i, lbl in enumerate(labels)}


class Exp4WaveformDataset(Dataset):
    """統一給pilot search、Stage1、Stage2共用的Dataset。

    三種用法：
      1. clean（預設）：不做任何擾動，直接建VI影像 —— 用於Stage1/Stage2的
         『訓練資料』（S0/S1/S2策略，或評估clean condition時）。
      2. corruption_type/severity 指定：對每個樣本套用『固定』的擾動條件
         （加上種子可控制隨機性）—— 用於Stage2評估某個corruption condition時
         的test set。
      3. dynamic_augment_fn 指定：訓練時每次__getitem__都重新、隨機決定
         要不要加擾動、加哪種、加多強 —— 用於S3/S4/combined策略的訓練資料。
      2和3只能擇一，不能同時指定。
    """
    def __init__(self, raw_samples, stats, label_map,
                 corruption_type='clean', severity=None, seed=None,
                 dynamic_augment_fn=None, inference_filter=None):
        assert not (corruption_type != 'clean' and dynamic_augment_fn is not None), \
            "corruption_type（固定擾動）跟dynamic_augment_fn（訓練期動態增強）不能同時指定"
        self.raw_samples = raw_samples
        self.stats = stats
        self.label_map = label_map
        self.corruption_type = corruption_type
        self.severity = severity
        self.dynamic_augment_fn = dynamic_augment_fn
        self.inference_filter = inference_filter  # 'S1' / 'S2' / None，見exp4_defense_strategies.py
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, idx):
        item = self.raw_samples[idx]
        c, v = item["current"], item["voltage"]

        if self.dynamic_augment_fn is not None:
            c, v = self.dynamic_augment_fn(c, v, self.stats, self.rng)
        elif self.corruption_type != 'clean':
            c, v = apply_waveform_corruption(c, v, self.corruption_type, self.severity, self.stats, self.rng)

        if self.inference_filter is not None:
            from exp4_defense_strategies_V2 import apply_inference_filter
            c, v = apply_inference_filter(c, v, self.inference_filter)

        img = generate_vi_image_fixed_norm(c, v, self.stats)
        label_idx = self.label_map[item["label"]]
        return torch.from_numpy(img).float(), label_idx


def group_aware_pilot_split_raw(train_raw_samples, val_ratio=0.2, seed=42):
    """跟Exp1 pilot_epoch_search.py的group_aware_pilot_split()同樣邏輯，
    只是作用在raw_samples list（依'group'欄位)，而不是ImageFolder的檔名。"""
    from sklearn.model_selection import GroupShuffleSplit

    groups = np.array([s["group"] for s in train_raw_samples])
    indices = np.arange(len(train_raw_samples))

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(indices, groups=groups))

    overlap = set(groups[train_idx]) & set(groups[val_idx])
    assert len(overlap) == 0, f"[嚴重錯誤] pilot-train/pilot-val機台重疊: {overlap}"

    print(f"  ✓ Pilot內部切分：train機台{len(set(groups[train_idx]))}台 / "
          f"val機台{len(set(groups[val_idx]))}台，完全不重疊")
    return train_idx, val_idx
