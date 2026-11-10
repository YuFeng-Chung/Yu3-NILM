import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.stats import friedmanchisquare, wilcoxon
import numpy as np
import time
import os
import json
import random
import warnings

try:
    from thop import profile
except ImportError:
    print("錯誤：找不到 thop 套件。請先在終端機執行: pip install thop")
    exit()

warnings.filterwarnings("ignore")

# ==========================================
# 1. 實驗參數設定 (Config)
# ==========================================
DATA_DIR = r'C:\Users\user\plaid_project\dataset'
NUM_CLASSES = 11
BATCH_SIZE = 32
NUM_RUNS = 10
# [已修正] 不再用單一NUM_EPOCHS_PER_RUN套用到所有模型；
# 改為讀取pilot_epoch_search.py產生的best_epochs.json，每個模型各自的固定epoch數。
# 若找不到該檔，退回此預設值並印出警告(不建議正式報告數據時使用預設值)。
DEFAULT_EPOCHS_FALLBACK = 100
BEST_EPOCHS_JSON = "best_epochs.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"啟動 [M1~M5 架構比較] 消融實驗... 使用裝置: {DEVICE}")

# ==========================================
# 2. 模型定義（M1~M5，抽到 models_m1_to_m5.py 統一維護，
#    避免此檔與 pilot_epoch_search.py 各自維護一份、日後漂移不一致）
# ==========================================
from models_m1_to_m5 import M1_Model, M2_Model, M3_Model, M4_Model, M5_Model

# ==========================================
# 4. 訓練與評估函數
#    [已修正] 不再逐epoch用test集評分、挑最佳epoch——
#    epoch數改為train_and_eval()呼叫端傳入的固定值(來自pilot_epoch_search.py的結果)，
#    訓練到指定epoch數後，只在test_loader上評估「一次」，該次結果即為最終成績。
# ==========================================
def train_and_eval(model, train_loader, test_loader, model_name, fixed_epochs):
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_recon = nn.MSELoss()

    for epoch in range(fixed_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)

            if model_name == 'M5':
                p1, p2, recon = outputs
                loss = 0.5*criterion_cls(p1, labels) + 1.0*criterion_cls(p2, labels) + 0.5*criterion_recon(recon, inputs)
            else:
                loss = criterion_cls(outputs, labels)

            loss.backward()
            optimizer.step()

    # 訓練結束，只在此處評估一次；test_loader在此之前完全沒有被用來做任何決策
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)

            if model_name == 'M5':
                final_pred = outputs[1]
            else:
                final_pred = outputs

            _, predicted = torch.max(final_pred.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    mac_pre, mac_rec, mac_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)

    return {'acc': acc, 'mac_pre': mac_pre, 'mac_rec': mac_rec, 'mac_f1': mac_f1}

# ==========================================
# 5. 主程式入口
# ==========================================
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    if not os.path.exists(os.path.join(DATA_DIR, 'train')):
        print(f"錯誤：找不到 {DATA_DIR} 資料夾！請確認路徑是否正確。")
        exit()

    # train_ds: 完整80%訓練集(不是pilot時縮減過的那份)
    # test_ds : 20%held-out測試集，從頭到尾未參與任何epoch數/模型選擇決策
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform)
    test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform)

    models_to_test = {
        'M1': M1_Model, 'M2': M2_Model, 'M3': M3_Model, 'M4': M4_Model, 'M5': M5_Model
    }

    # 讀取pilot_epoch_search.py決定的各模型固定epoch數
    if os.path.exists(BEST_EPOCHS_JSON):
        with open(BEST_EPOCHS_JSON, "r", encoding="utf-8") as f:
            best_epochs = json.load(f)
        fixed_epochs_by_model = {
            name: best_epochs[name]["fixed_epoch"] for name in models_to_test if name in best_epochs
        }
        print(f"✓ 已讀取 {BEST_EPOCHS_JSON}，各模型固定epoch數: {fixed_epochs_by_model}")
    else:
        fixed_epochs_by_model = {}
        print(f"[警告] 找不到 {BEST_EPOCHS_JSON}，請先執行 pilot_epoch_search.py 產生各模型的固定epoch數。"
              f"目前將對所有模型使用預設值 {DEFAULT_EPOCHS_FALLBACK}，此設定不建議用於正式報告數據。")

    final_results = {}

    for name, ModelClass in models_to_test.items():
        model_epochs = fixed_epochs_by_model.get(name, DEFAULT_EPOCHS_FALLBACK)
        print(f"\n[{name}] 準備就緒，開始進行 {NUM_RUNS} 次獨立實驗 (固定epoch數={model_epochs})...")
        
        temp_model = ModelClass().to(DEVICE)
        dummy_input = torch.randn(1, 1, 64, 64).to(DEVICE)
        macs, params = profile(temp_model, inputs=(dummy_input, ), verbose=False)
        flops = macs * 2 
        
        temp_model.eval()
        with torch.no_grad():
            for _ in range(10): _ = temp_model(dummy_input)
            start_l = time.time()
            for _ in range(100): _ = temp_model(dummy_input)
            latency = (time.time() - start_l) / 100 * 1000 
        del temp_model
        
        runs_acc, runs_pre, runs_rec, runs_f1 = [], [], [], []
        
        start_time = time.time()
        for run in range(NUM_RUNS):
            seed = 42 + run
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            generator = torch.Generator()
            generator.manual_seed(seed)
            train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
            test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

            model = ModelClass().to(DEVICE)
            metrics = train_and_eval(model, train_dl, test_dl, name, model_epochs)
            
            runs_acc.append(metrics['acc'] * 100)
            runs_pre.append(metrics['mac_pre'] * 100)
            runs_rec.append(metrics['mac_rec'] * 100)
            runs_f1.append(metrics['mac_f1'] * 100)
            
            print(f"  └ Run {run+1:02d} (Seed {seed}): Acc={metrics['acc']*100:.2f}%, Pre={metrics['mac_pre']*100:.2f}%, Rec={metrics['mac_rec']*100:.2f}%, F1={metrics['mac_f1']*100:.2f}%")
            
        total_time = time.time() - start_time
        avg_train_time = total_time / NUM_RUNS 
        
        final_results[name] = {
            'Acc': np.mean(runs_acc), 'Acc_std': np.std(runs_acc, ddof=1),
            'Pre': np.mean(runs_pre), 'Pre_std': np.std(runs_pre, ddof=1),
            'Rec': np.mean(runs_rec), 'Rec_std': np.std(runs_rec, ddof=1),
            'F1': np.mean(runs_f1),   'F1_std': np.std(runs_f1, ddof=1),
            'Params(M)': params / 1e6, 'FLOPs(G)': flops / 1e9, 
            'Latency(ms)': latency, 'TrainTime(s)': avg_train_time,
            'runs_acc_raw': runs_acc, 'runs_pre_raw': runs_pre,
            'runs_rec_raw': runs_rec, 'runs_f1_raw': runs_f1
        }
        print(f"[{name}] 執行完成! 平均每 Run 耗時: {avg_train_time:.1f} 秒")

    # ==========================================
    # 6. 輸出最終報表與統計檢定 (終端機顯示)
    # ==========================================
    print("\n\n" + "="*145)
    print(" 論文數據表格 (均已換算為百分比 %，格式: Mean ± SD，附加複雜度指標)")
    print("="*145)
    print("| Model | Accuracy (%) | Macro Pre (%) | Macro Rec (%) | Macro F1 (%) | Params (M) | FLOPs (G) | Latency (ms) | Time/Run (s) |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for name in models_to_test.keys():
        res = final_results[name]
        print(f"| **{name}** | {res['Acc']:.2f} ± {res['Acc_std']:.2f} | {res['Pre']:.2f} ± {res['Pre_std']:.2f} | {res['Rec']:.2f} ± {res['Rec_std']:.2f} | {res['F1']:.2f} ± {res['F1_std']:.2f} | {res['Params(M)']:.3f} | {res['FLOPs(G)']:.3f} | {res['Latency(ms)']:.2f} | {res['TrainTime(s)']:.1f} |")
    print("="*145)
    
    # ---------------------------------------------------------
    # 🔥 關鍵修正：印出完整的「表 15」4 指標 Wilcoxon 檢定結果
    # ---------------------------------------------------------
    print("\n" + "="*125)
    print(" 📋 表 15 PLAID 上之 Wilcoxon 符號等級配對檢定結果 (全 4 項指標)")
    print("="*125)
    print("| Comparison | Accuracy (Δ / p-value) | Macro Pre (Δ / p-value) | Macro Rec (Δ / p-value) | Macro F1 (Δ / p-value) |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    
    comparisons_text = [
        ("M2 vs M1", 'M2', 'M1'),
        ("M3 vs M1", 'M3', 'M1'),
        ("M4 vs M2", 'M4', 'M2'),
        ("M4 vs M3", 'M4', 'M3'),
        ("M5 (Yu3) vs M3", 'M5', 'M3')
    ]
    
    metrics_info = [
        ('runs_acc_raw'),
        ('runs_pre_raw'),
        ('runs_rec_raw'),
        ('runs_f1_raw')
    ]

    for desc, m_test, m_base in comparisons_text:
        row_str = f"| **{desc:<15}** "
        for metric_key in metrics_info:
            data_test = final_results[m_test][metric_key]
            data_base = final_results[m_base][metric_key]
            delta = np.mean(data_test) - np.mean(data_base)
            
            try:
                _, p_val = wilcoxon(data_test, data_base)
            except ValueError:
                p_val = 1.0
            
            sign = "+" if delta > 0 else ""
            
            # 加上星星記號，讓您一眼看出顯著性
            if p_val < 0.01: star = "**"
            elif p_val < 0.05: star = "*"
            elif p_val <= 0.10: star = "†"
            else: star = ""
            
            row_str += f"| {sign}{delta:.2f} / p={p_val:.3f}{star} "
        row_str += "|"
        print(row_str)
    print("="*125 + "\n")

    # ==========================================
    # 7. 自動作圖與輸出獨立圖檔 (包含 4 指標熱力圖)
    # ==========================================
    print("\n" + "="*80)
    print(" 🔬 正在自動生成 3 張高畫質獨立圖表...")
    print("="*80)

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        model_names = ['M1', 'M2', 'M3', 'M4', 'M5 (Yu3)']
        keys = ['M1', 'M2', 'M3', 'M4', 'M5']
        
        plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
        sns.set_theme(style="whitegrid", rc={"axes.spines.top": False, "axes.spines.right": False})

        # --- 圖 (a) ---
        plt.figure(figsize=(8, 6))
        means_acc = [final_results[m]['Acc'] for m in keys]
        stds_acc = [final_results[m]['Acc_std'] for m in keys]
        colors = ['#9e9e9e', '#9e9e9e', '#9e9e9e', '#9e9e9e', '#3b82f6']
        
        bars = plt.bar(model_names, means_acc, yerr=stds_acc, capsize=6, color=colors, edgecolor='black', alpha=0.9, linewidth=1.2)
        plt.ylim(75, 90)
        plt.ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
        plt.title('(a) Clean PLAID Baseline Performance (Accuracy)', fontsize=15, fontweight='bold', pad=15)
        
        for bar, mean in zip(bars, means_acc):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.6, f'{mean:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
            
        plt.tight_layout()
        plt.savefig("Exp1_fig_a_accuracy_bar.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 成功生成圖 (a): Exp1_fig_a_accuracy_bar.png")

        # --- 圖 (b) ---
        plt.figure(figsize=(10, 6))
        df_list = []
        for i, m in enumerate(keys):
            for val in final_results[m]['runs_acc_raw']:
                df_list.append({'Model': model_names[i], 'Score': val, 'Metric': 'Accuracy'})
            for val in final_results[m]['runs_f1_raw']:
                df_list.append({'Model': model_names[i], 'Score': val, 'Metric': 'Macro F1'})
        df_box = pd.DataFrame(df_list)
        
        ax = sns.boxplot(data=df_box, x='Model', y='Score', hue='Metric', palette=['#A9CCE3', '#F5B041'], showfliers=False, width=0.6, boxprops=dict(edgecolor="black", linewidth=1.2))
        sns.stripplot(data=df_box, x='Model', y='Score', hue='Metric', dodge=True, color='black', alpha=0.5, jitter=True, size=4, ax=ax)
        
        plt.ylabel('Score (%)', fontsize=13, fontweight='bold')
        plt.xlabel('')
        plt.title('(b) Run-level distribution (n=10): Accuracy vs. Macro F1', fontsize=15, fontweight='bold', pad=15)
        
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles[:2], labels[:2], loc='lower right', frameon=True, edgecolor='black')
        plt.tight_layout()
        plt.savefig("Exp1_fig_b_distribution_box.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 成功生成圖 (b): Exp1_fig_b_distribution_box.png")

        # --- 圖 (c) 包含 4 指標 ---
        comparisons = [
            ("M2 vs M1\n(CBAM effect)", 'M2', 'M1'),
            ("M3 vs M1\n(Feedback effect)", 'M3', 'M1'),
            ("M4 vs M2\n(Feedback | CBAM)", 'M4', 'M2'),
            ("M4 vs M3\n(CBAM | Feedback)", 'M4', 'M3'),
            ("M5 (Yu3) vs M3\n(Key test)", 'M5', 'M3')
        ]
        metrics_display = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1']
        metrics_keys = ['runs_acc_raw', 'runs_pre_raw', 'runs_rec_raw', 'runs_f1_raw']

        plt.figure(figsize=(11, 7))
        heatmap_data = np.zeros((len(comparisons), len(metrics_display)))
        annot_data = np.empty((len(comparisons), len(metrics_display)), dtype=object)
        
        for i, (comp_name, m_test, m_base) in enumerate(comparisons):
            for j, m_key in enumerate(metrics_keys):
                data_test = final_results[m_test][m_key]
                data_base = final_results[m_base][m_key]
                
                delta = np.mean(data_test) - np.mean(data_base)
                heatmap_data[i, j] = delta
                
                try:
                    _, p_val = wilcoxon(data_test, data_base)
                except ValueError:
                    p_val = 1.0
                    
                if p_val < 0.01: star = "**"
                elif p_val < 0.05: star = "*"
                elif p_val <= 0.10: star = "†"
                else: star = "ns"
                
                sign = "+" if delta > 0 else ""
                annot_data[i, j] = f"{sign}{delta:.2f}\n\n{star}"

        sns.heatmap(heatmap_data, annot=annot_data, fmt="", cmap='RdBu_r', center=0, 
                    cbar_kws={'label': 'Δ (pp)'}, annot_kws={"weight": "bold", "size": 13},
                    linewidths=2, linecolor='white')
        
        plt.yticks(ticks=np.arange(len(comparisons))+0.5, labels=[c[0] for c in comparisons], rotation=0, fontsize=12, fontweight='bold')
        plt.xticks(ticks=np.arange(len(metrics_display))+0.5, labels=metrics_display, fontsize=12, fontweight='bold')
        plt.title('(c) Pairwise Δ and significance (Wilcoxon, n=10)', fontsize=15, fontweight='bold', pad=20)
        
        plt.text(0.5, -0.15, '* p<0.05  ** p<0.01  † 0.05<=p<=0.10 (trend)  ns not significant.', 
                 ha='center', va='center', transform=plt.gca().transAxes, fontsize=11, style='italic')
                 
        plt.tight_layout()
        plt.savefig("Exp1_fig_c_significance_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(" 成功生成圖 (c): Exp1_fig_c_significance_heatmap.png")
        print("\n 三張圖表已完美輸出！")
        
    except ImportError:
        print("\n 系統找不到畫圖套件。請執行 `pip install matplotlib seaborn pandas`")