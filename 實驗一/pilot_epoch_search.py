import os
import re
import json
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit

from models_m1_to_m5 import M1_Model, M2_Model, M3_Model, M4_Model, M5_Model

warnings.filterwarnings("ignore")

# ==========================================
# 1. 設定區 (Config) —— 與 plaid_m1_to_m5.py 保持一致
# ==========================================
DATA_DIR = r'C:\Users\user\plaid_project\實驗一程式碼\dataset'
NUM_CLASSES = 11
BATCH_SIZE = 32
MAX_EPOCHS = 100          # pilot最多跑幾個epoch（早停通常會提早結束）
PATIENCE = 15             # 連續多少個epoch沒有明顯進步就早停
MIN_DELTA = 0.001         # 「明顯進步」的門檻（macro-F1，0.001 = 0.1個百分點）
PILOT_VAL_RATIO = 0.2     # 從80%訓練集中，再切多少比例出來當pilot-val
SEED = 42                 # pilot只跑單一seed，不是要拿來報告效能數字，只是找收斂點
OUTPUT_JSON = "best_epochs.json"
OUTPUT_CURVE_DIR = "pilot_curves"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = {
    'M1': M1_Model, 'M2': M2_Model, 'M3': M3_Model, 'M4': M4_Model, 'M5': M5_Model
}

# ==========================================
# 2. 從檔名還原機台group id
#    make_dataset_split.py已修改為存成 "{group}__{i}.png"
# ==========================================
def extract_group_from_path(path):
    fname = os.path.basename(path)
    m = re.match(r"^(.*)__\d+\.png$", fname)
    if not m:
        raise ValueError(
            f"檔名 {fname} 找不到group前綴，請確認 make_dataset_split.py 是否已更新為"
            f" '{{group}}__{{i}}.png' 的存檔格式，並重新產生 dataset/train 資料夾。"
        )
    return m.group(1)


def group_aware_pilot_split(image_folder_dataset, val_ratio=0.2, seed=42):
    """在ImageFolder的樣本上，依檔名還原的group做GroupShuffleSplit，
    確保同一台機器的樣本不會同時出現在pilot-train與pilot-val。"""
    groups = [extract_group_from_path(p) for p, _ in image_folder_dataset.samples]
    groups = np.array(groups)
    indices = np.arange(len(groups))

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(indices, groups=groups))

    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])
    overlap = train_groups & val_groups
    assert len(overlap) == 0, f"[嚴重錯誤] pilot-train/pilot-val機台重疊，共{len(overlap)}台: {overlap}"

    print(f"  ✓ Pilot內部切分：train機台{len(train_groups)}台 / val機台{len(val_groups)}台，完全不重疊")
    return train_idx, val_idx


# ==========================================
# 3. 單一模型的pilot訓練：早停 + 記錄F1曲線
# ==========================================
def run_pilot_for_model(name, ModelClass, pilot_train_loader, pilot_val_loader):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model = ModelClass().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_recon = nn.MSELoss()

    f1_curve = []
    best_f1 = -1.0
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for inputs, labels in pilot_train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            if name == 'M5':
                p1, p2, recon = outputs
                loss = 0.5 * criterion_cls(p1, labels) + 1.0 * criterion_cls(p2, labels) \
                       + 0.5 * criterion_recon(recon, inputs)
            else:
                loss = criterion_cls(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in pilot_val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                final_pred = outputs[1] if name == 'M5' else outputs
                _, predicted = torch.max(final_pred.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        _, _, mac_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        f1_curve.append(mac_f1)

        if mac_f1 > best_f1 + MIN_DELTA:
            best_f1 = mac_f1
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"  [{name}] epoch {epoch:3d}  pilot-val macro-F1={mac_f1*100:.2f}%  "
              f"(best={best_f1*100:.2f}% @epoch{best_epoch}, no_improve={epochs_no_improve})")

        if epochs_no_improve >= PATIENCE:
            print(f"  [{name}] 早停：連續{PATIENCE}個epoch macro-F1未提升超過{MIN_DELTA*100:.1f}個百分點，"
                  f"於epoch {epoch}停止，採用收斂點 epoch {best_epoch}")
            break

    return best_epoch, best_f1, f1_curve


# ==========================================
# 4. 主程式
# ==========================================
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    train_dir = os.path.join(DATA_DIR, 'train')
    if not os.path.exists(train_dir):
        print(f"錯誤：找不到 {train_dir}！請先執行 make_dataset_split.py。")
        exit()

    full_train_ds = datasets.ImageFolder(train_dir, transform)
    print(f"完整80%訓練集共 {len(full_train_ds)} 張影像")

    print("\n正在依機台group做Pilot內部切分（不動用正式20% test set）...")
    train_idx, val_idx = group_aware_pilot_split(full_train_ds, val_ratio=PILOT_VAL_RATIO, seed=SEED)

    pilot_train_ds = Subset(full_train_ds, train_idx)
    pilot_val_ds = Subset(full_train_ds, val_idx)
    print(f"Pilot-train樣本數: {len(pilot_train_ds)}  Pilot-val樣本數: {len(pilot_val_ds)}\n")

    os.makedirs(OUTPUT_CURVE_DIR, exist_ok=True)
    results = {}

    for name, ModelClass in MODELS.items():
        print(f"=== [{name}] 開始Pilot訓練（單一seed={SEED}，最多{MAX_EPOCHS} epoch，"
              f"patience={PATIENCE}）===")

        g = torch.Generator()
        g.manual_seed(SEED)
        pilot_train_loader = DataLoader(pilot_train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
        pilot_val_loader = DataLoader(pilot_val_ds, batch_size=BATCH_SIZE)

        best_epoch, best_f1, f1_curve = run_pilot_for_model(
            name, ModelClass, pilot_train_loader, pilot_val_loader
        )
        results[name] = {"fixed_epoch": best_epoch, "pilot_val_f1": best_f1}

        curve_path = os.path.join(OUTPUT_CURVE_DIR, f"{name}_pilot_f1_curve.json")
        with open(curve_path, "w", encoding="utf-8") as f:
            json.dump({"f1_curve": f1_curve, "best_epoch": best_epoch}, f, ensure_ascii=False, indent=2)
        print(f"  → 已存曲線: {curve_path}\n")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(" Pilot結果彙總（各模型固定epoch數）")
    print("=" * 60)
    for name, r in results.items():
        print(f"  {name}: fixed_epoch={r['fixed_epoch']}  (pilot-val macro-F1={r['pilot_val_f1']*100:.2f}%)")
    print(f"\n✓ 已寫入 {OUTPUT_JSON}，plaid_m1_to_m5.py 會自動讀取此檔案。")
