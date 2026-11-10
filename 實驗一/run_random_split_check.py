import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from models_m1_to_m5 import M1_Model, M5_Model

# ==========================================
# 嚴格控制變因設定
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = {"M1": 14, "M5": 19}  # 完全沿用 Group-Split 找出的最佳收斂點
MODELS = {"M1": M1_Model, "M5": M5_Model}
BATCH_SIZE = 32
LR = 0.0005

# 影像前處理完全一致
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

def run_experiment(model_name, ModelClass, seed):
    # 強制鎖定亂數種子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_dir = f"./dataset_random/seed{seed}/train"
    test_dir = f"./dataset_random/seed{seed}/val"
    
    train_ds = datasets.ImageFolder(train_dir, transform)
    test_ds = datasets.ImageFolder(test_dir, transform)
    
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = ModelClass().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_recon = nn.MSELoss()

    # 訓練階段 (依指定 Epochs 盲跑)
    model.train()
    for epoch in range(EPOCHS[model_name]):
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x)
            
            if model_name == "M5":
                p1, p2, recon = outputs
                loss = 0.5 * criterion_cls(p1, y) + 1.0 * criterion_cls(p2, y) + 0.5 * criterion_recon(recon, x)
            else:
                loss = criterion_cls(outputs, y)
                
            loss.backward()
            optimizer.step()

    # 測試階段
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            outputs = model(x)
            final_pred = outputs[1] if model_name == "M5" else outputs
            _, predicted = torch.max(final_pred, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.numpy())

    acc = accuracy_score(all_labels, all_preds) * 100
    _, _, mac_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    
    return acc, mac_f1 * 100

if __name__ == "__main__":
    print("=== 開始執行隨機切分效能驗證 (嚴格固定 Epochs) ===")
    results = {}
    
    for name, ModelClass in MODELS.items():
        print(f"\n--- 測試模型: {name} (固定 Epochs: {EPOCHS[name]}) ---")
        results[name] = []
        acc_list, f1_list = [], []
        
        for seed in range(42, 52):
            acc, f1 = run_experiment(name, ModelClass, seed)
            results[name].append({"seed": seed, "acc": acc, "macro_f1": f1})
            acc_list.append(acc)
            f1_list.append(f1)
            print(f"  Seed {seed:2d} -> Acc: {acc:.2f}%, F1: {f1:.2f}%")
            
        print(f"[{name} 彙總] Acc: {np.mean(acc_list):.2f} ± {np.std(acc_list, ddof=1):.2f}% | F1: {np.mean(f1_list):.2f} ± {np.std(f1_list, ddof=1):.2f}%")

    with open("random_split_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("\n✓ 測試完成，結果已存入 random_split_results.json")