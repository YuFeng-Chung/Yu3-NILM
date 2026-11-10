import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.stats import friedmanchisquare
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import json
import time
import warnings

from models_m1_to_m5 import BaseEncoder, NUM_CLASSES

warnings.filterwarnings("ignore")

# ==========================================
# 1.實驗核心參數設定
# ==========================================
CLEAN_TRAIN_DIR = r'C:\Users\user\plaid_project\dataset\train'
CLEAN_VAL_DIR = r'C:\Users\user\plaid_project\dataset\val'
BEST_EPOCHS_JSON = "best_epochs.json"
OUTPUT_FIG_DIR = 'paper_figures_300dpi'

NUM_RUNS = 10
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)
MODES = ['No-FB', 'Img-FB', 'Err-FB', 'Three-Pass']

# ==========================================
# 2.繼承 M5_Model 架構的消融模型
# ==========================================
class Yu3AblationModel(nn.Module):
    def __init__(self, mode='Err-FB'):
        super(Yu3AblationModel, self).__init__()
        self.mode = mode
        
        #  複製 models_m1_to_m5.py 中 M5_Model 的定義
        self.encoder = BaseEncoder(in_channels=2) 
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1) 
        self.dec_bn1 = nn.BatchNorm2d(64)
        self.dec_conv2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1) 
        self.dec_bn2 = nn.BatchNorm2d(32)
        self.dec_conv3 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1) 
        self.fc = nn.Linear(128 * 8 * 8, 256) 
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        # --- Pass 1 ---
        dummy_r = torch.zeros_like(x)
        feat1 = self.encoder(torch.cat([x, dummy_r], dim=1))
        flat1 = feat1.view(feat1.size(0), -1)
        p1 = self.classifier(self.dropout(F.relu(self.fc(flat1))))

        if self.mode == 'No-FB':
            return p1

        #  完全複製 M5_Model 的解碼層邏輯
        recon1 = F.relu(self.dec_bn1(self.dec_conv1(feat1)))
        recon1 = F.relu(self.dec_bn2(self.dec_conv2(recon1)))
        x_hat1 = torch.sigmoid(self.dec_conv3(recon1))
        
        if self.mode == 'Img-FB':
            aux1 = x_hat1
        elif self.mode in ['Err-FB', 'Three-Pass']:
            #  關鍵修正：套用真實 M5 的 .detach() 保護機制
            aux1 = torch.abs(x - x_hat1).detach()

        # --- Pass 2 ---
        feat2 = self.encoder(torch.cat([x, aux1], dim=1))
        flat2 = feat2.view(feat2.size(0), -1)
        p2 = self.classifier(self.dropout(F.relu(self.fc(flat2))))

        if self.mode in ['Img-FB', 'Err-FB']:
            return p1, p2, x_hat1

        # --- Pass 3 (僅 Three-Pass 使用) ---
        recon2 = F.relu(self.dec_bn1(self.dec_conv1(feat2)))
        recon2 = F.relu(self.dec_bn2(self.dec_conv2(recon2)))
        x_hat2 = torch.sigmoid(self.dec_conv3(recon2))
        
        aux2 = torch.abs(x - x_hat2).detach()
        feat3 = self.encoder(torch.cat([x, aux2], dim=1))
        flat3 = feat3.view(feat3.size(0), -1)
        p3 = self.classifier(self.dropout(F.relu(self.fc(flat3))))

        return p1, p2, p3, x_hat1, x_hat2

# ==========================================
# 3. 損失函數與推論擷取
# ==========================================
def compute_ablation_loss(outputs, labels, inputs, mode, criterion_cls, criterion_recon):
    if mode == 'No-FB':
        return criterion_cls(outputs, labels)
    elif mode in ['Img-FB', 'Err-FB']:
        y1, y2, x_hat1 = outputs
        # 🚨 完全對齊 plaid_m1_to_m5.py 中的損失比例
        return 0.5*criterion_cls(y1, labels) + 1.0*criterion_cls(y2, labels) + 0.5*criterion_recon(x_hat1, inputs)
    elif mode == 'Three-Pass':
        y1, y2, y3, x_hat1, x_hat2 = outputs
        loss_c = 0.5*criterion_cls(y1, labels) + 0.5*criterion_cls(y2, labels) + 1.0*criterion_cls(y3, labels)
        loss_r = 0.25*criterion_recon(x_hat1, inputs) + 0.25*criterion_recon(x_hat2, inputs)
        return loss_c + loss_r

def get_final_pred(outputs, mode):
    if mode == 'No-FB': return outputs
    elif mode in ['Img-FB', 'Err-FB']: return outputs[1]
    elif mode == 'Three-Pass': return outputs[2]

# ==========================================
# 4. 訓練核心 (100% 複製實驗一流程)
# ==========================================
def train_and_eval_ablation(mode, fixed_epochs, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    #  完全複製實驗一的 transforms，未做額外手腳
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    train_ds = datasets.ImageFolder(CLEAN_TRAIN_DIR, transform)
    test_ds = datasets.ImageFolder(CLEAN_VAL_DIR, transform)
    
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = Yu3AblationModel(mode=mode).to(DEVICE)
    #  採用與實驗一完全相同的 Adam lr=0.0005 與 MSELoss
    optimizer = optim.Adam(model.parameters(), lr=0.0005) 
    criterion_cls = nn.CrossEntropyLoss()
    criterion_recon = nn.MSELoss() 

    start_time = time.time()
    
    # --- 訓練階段 ---
    for epoch in range(fixed_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = compute_ablation_loss(outputs, labels, inputs, mode, criterion_cls, criterion_recon)
            loss.backward()
            optimizer.step()

    # --- 評估階段 (僅考一次) ---
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            final_pred = get_final_pred(outputs, mode)
            _, predicted = torch.max(final_pred.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    run_time = time.time() - start_time
    acc = accuracy_score(all_labels, all_preds) * 100
    mac_prec, mac_rec, mac_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    
    return acc, mac_prec * 100, mac_rec * 100, mac_f1 * 100, run_time

# ==========================================
# 5. 主程式流程與表格輸出
# ==========================================
if __name__ == '__main__':
    if os.path.exists(BEST_EPOCHS_JSON):
        with open(BEST_EPOCHS_JSON, "r", encoding="utf-8") as f:
            best_epochs = json.load(f)
            fixed_epochs = best_epochs.get('M5', {}).get('fixed_epoch', 19)
    else:
        fixed_epochs = 19
        print(f"警告：找不到 {BEST_EPOCHS_JSON}，使用預設 Epoch=19")

    print(f" 開始執行 Experiment 3: Yu3 回授策略消融實驗 (Epoch={fixed_epochs}, n={NUM_RUNS})")
    
    results = {m: {'Acc': [], 'Prec': [], 'Rec': [], 'F1': [], 'TotalTime': 0.0} for m in MODES}

    for run in range(NUM_RUNS):
        seed = 42 + run
        print(f"\n--- Run {run+1}/10 (Seed {seed}) ---")
        for mode in MODES:
            acc, prec, rec, f1, r_time = train_and_eval_ablation(mode, fixed_epochs, seed)
            results[mode]['Acc'].append(acc)
            results[mode]['Prec'].append(prec)
            results[mode]['Rec'].append(rec)
            results[mode]['F1'].append(f1)
            results[mode]['TotalTime'] += r_time
            print(f"  └ [{mode:10s}] Acc: {acc:.2f}%, P: {prec:.2f}%, R: {rec:.2f}%, F1: {f1:.2f}%")

    print("\n\n" + "="*95)
    print(" 📊 Table 20: Performance of feedback strategies in the Yu3 ablation study")
    print("="*95)
    print(f"| {'Feedback strategy':<32} | {'Accuracy (%)':<14} | {'Macro Precision (%)':<20} | {'Macro Recall (%)':<17} | {'Macro F1 (%)':<15} |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    
    for mode in MODES:
        acc_m, acc_s = np.mean(results[mode]['Acc']), np.std(results[mode]['Acc'], ddof=1)
        pre_m, pre_s = np.mean(results[mode]['Prec']), np.std(results[mode]['Prec'], ddof=1)
        rec_m, rec_s = np.mean(results[mode]['Rec']), np.std(results[mode]['Rec'], ddof=1)
        f1_m, f1_s = np.mean(results[mode]['F1']), np.std(results[mode]['F1'], ddof=1)
        
        mode_names = {
            'No-FB': 'No feedback',
            'Img-FB': 'Reconstructed-image feedback',
            'Err-FB': 'Reconstruction-error (one pass)',
            'Three-Pass': 'Reconstruction-error (three passes)'
        }
        print(f"| {mode_names[mode]:<32} | {acc_m:5.2f} ± {acc_s:4.2f} | {pre_m:5.2f} ± {pre_s:4.2f}       | {rec_m:5.2f} ± {rec_s:4.2f}    | {f1_m:5.2f} ± {f1_s:4.2f}  |")
    print("="*95)

    print("\n\n" + "="*95)
    print(" 📊 Table 21: Friedman test results across the four feedback strategies")
    print("="*95)
    print("| Metric             | χ²(3)    | p-value  | Kendall's W  | Interpretation       |")
    print("| :---               | :---     | :---     | :---         | :---                 |")

    metrics_list = [('Acc', 'Accuracy'), ('Prec', 'Macro Precision'), ('Rec', 'Macro Recall'), ('F1', 'Macro F1-score')]
    for metric_key, metric_name in metrics_list:
        data_for_test = [results[m][metric_key] for m in MODES]
        stat, p_val = friedmanchisquare(*data_for_test)
        w_val = stat / (NUM_RUNS * (len(MODES) - 1))
        interpretation = "Significant" if p_val < 0.05 else "Not significant"
        print(f"| {metric_name:<18} | {stat:<8.2f} | {p_val:<8.3f} | {w_val:<12.3f} | {interpretation:<20} |")
    print("="*95 + "\n")

    print(" 正在生成 3 張獨立 300 DPI 高畫質消融比較圖...")
    plt.rcParams['font.family'] = 'sans-serif'
    x_positions = np.arange(len(MODES))

    def plot_independent_metric(metric_key, title, ylabel, filename):
        plt.figure(figsize=(7, 5), dpi=300)
        ax = plt.gca()
        for run in range(NUM_RUNS):
            run_vals = [results[m][metric_key][run] for m in MODES]
            ax.plot(x_positions, run_vals, color='#bdc3c7', alpha=0.6, marker='o', markersize=5, zorder=1)
        means = [np.mean(results[m][metric_key]) for m in MODES]
        stds = [np.std(results[m][metric_key], ddof=1) for m in MODES]
        ax.errorbar(x_positions, means, yerr=stds, fmt='D', color='black', capsize=5, 
                    markersize=8, markerfacecolor='white', markeredgewidth=1.5, zorder=3)
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 0.3, f"{m:.2f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', loc='left', pad=15)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(MODES, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FIG_DIR, filename), dpi=300, bbox_inches='tight')
        plt.close()

    plot_independent_metric('Acc', '(a) Accuracy', 'Accuracy (%)', 'fig8a_Accuracy.png')
    plot_independent_metric('F1', '(b) Macro F1-score', 'Macro F1-score (%)', 'fig8b_MacroF1.png')

    plt.figure(figsize=(8, 4), dpi=300)
    ax_time = plt.gca()
    colors = ['#95a5a6', '#7FB3D5', '#34495e', '#1f4e79']
    times = [results[m]['TotalTime'] for m in MODES]
    bars = ax_time.barh(x_positions, times, height=0.6, color=colors, zorder=3)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ratio = times[i] / times[0]
        ax_time.text(width + 30, bar.get_y() + bar.get_height()/2, f"{width:.1f} s ({ratio:.2f}x)", ha='left', va='center', fontsize=11)
    ax_time.set_title('(c) Recorded computational cost', fontsize=14, fontweight='bold', loc='left', pad=15)
    ax_time.set_yticks(x_positions)
    ax_time.set_yticklabels(MODES, fontsize=11)
    ax_time.set_xlabel('Total experimental runtime for 10 runs (s)', fontsize=12)
    ax_time.invert_yaxis()
    ax_time.grid(axis='x', linestyle='--', alpha=0.5, zorder=0)
    ax_time.spines['top'].set_visible(False)
    ax_time.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIG_DIR, 'fig8c_Cost.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 3 張獨立高畫質圖表已成功輸出至 '{OUTPUT_FIG_DIR}' 資料夾！")