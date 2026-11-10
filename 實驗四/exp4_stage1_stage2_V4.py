"""
Exp4 主程式（修正版 V3）：雜訊與缺失資料穩健性實驗
==================================================
對照手稿 Materials and Methods (3) 與 Experiment 4 段落重寫，關鍵修正：

  1. 資料一律經由 exp4_raw_data.py 的原始波形管線建構（固定訓練集統計量、
     擾動加在波形上），不是在Exp1的PNG圖片上動手腳。
  2. 三個baseline改成from-scratch、1-channel、64×64（models_baselines.py）。
  3. Stage 1『只』用Yu3(M5)，5-fold Group-Stratified CV，在80%訓練集內部
     完成，完全不碰held-out test set；雜訊與缺失資料是『兩條獨立』的
     Stage1/Stage2流程（各自的候選策略、各自的最終選定策略），不要混在一起。
  4. Stage 2的統計設計嚴格照手稿公式：每個模型10次獨立配對訓練(r=1..10)，
     每個擾動條件另外用10個『固定且跨模型一致』的污染種子(s=1..10)，
     每次訓練得到的模型先在10個污染實現上評估、取平均，才是該次訓練在
     該條件下的代表值；最終統計樣本數 n=10（10次獨立訓練），而不是100。

【V3 新增紀錄功能】
  - Stage 1 雜訊篩選：補存 S0~S4 在 Clean、AWGN 0dB、Salt-Pepper 20% 三個條件
    個別的逐 fold Macro-F1，並於結尾輸出 Mean ± SD 供 Table 23 使用。
  - Stage 1 缺失資料篩選：補存 4 種策略在 連續 20%、隨機 20% 個別的逐 fold 
    原始 Macro-F1，並於結尾輸出 Mean ± SD 供表格使用。
"""

import os
import json
import random
import warnings
from collections import Counter

import numpy as np
import torch

# --- 強制關閉 Flash Attention，避免舊顯卡架構報錯 ---
if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
# ------------------------------------------------
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import friedmanchisquare, wilcoxon

from models_m1_to_m5_V2 import M5_Model
from models_baselines_V2 import ResNet18_Model, MobileNetV3_Model, ViTTiny_Model
from exp4_raw_data_V2 import (
    load_plaid_raw_samples, split_by_exp1_record, compute_fixed_norm_stats,
    build_label_map, Exp4WaveformDataset, SEVERITY_GRIDS,
)
from exp4_defense_strategies_V2 import (
    NOISE_STRATEGIES, MISSING_STRATEGIES, FINAL_NOISE_STRATEGY, FINAL_MISSING_STRATEGY,
    get_training_config, get_inference_filter,
)

warnings.filterwarnings("ignore")

BATCH_SIZE = 32
NUM_RUNS = 10          
CORRUPTION_SEEDS = list(range(1000, 1010))  
STAGE1_N_FOLDS = 5
STAGE1_SEED = 42
BEST_EPOCHS_JSON = "best_epochs_exp4.json"
STATS_JSON = "exp4_fixed_norm_stats.json"
LABEL_MAP_JSON = "exp4_label_map.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_MODELS = {
    'Yu3': M5_Model, 'ResNet18': ResNet18_Model,
    'MobileNetV3': MobileNetV3_Model, 'ViT-Tiny': ViTTiny_Model,
}
YU3_FAMILY = {'Yu3', 'M5'}


# ==========================================
# 共用：訓練 / 評估
# ==========================================
def make_loader(raw_samples, stats, label_map, seed, shuffle=True, **ds_kwargs):
    ds = Exp4WaveformDataset(raw_samples, stats, label_map, seed=seed, **ds_kwargs)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, generator=g if shuffle else None)


def train_model(model, name, train_loader, epochs, weight_decay=0.0):
    lr = 0.0005 
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_recon = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            if name in YU3_FAMILY:
                p1, p2, recon = outputs
                loss = 0.5 * criterion_cls(p1, labels) + 1.0 * criterion_cls(p2, labels) \
                       + 0.5 * criterion_recon(recon, inputs)
            else:
                loss = criterion_cls(outputs, labels)
            loss.backward()
            optimizer.step()
    return model


def eval_model(model, name, eval_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in eval_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            final_pred = outputs[1] if name in YU3_FAMILY else outputs
            _, predicted = torch.max(final_pred.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    _, _, mac_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    return acc, mac_f1


def eval_condition_averaged(model, name, test_raw, stats, label_map, corruption_type, severity,
                            inference_filter=None):
    if corruption_type == 'clean':
        dl = make_loader(test_raw, stats, label_map, seed=0, shuffle=False,
                        corruption_type='clean', inference_filter=inference_filter)
        return eval_model(model, name, dl)

    accs, f1s = [], []
    for cseed in CORRUPTION_SEEDS:
        dl = make_loader(test_raw, stats, label_map, seed=cseed, shuffle=False,
                        corruption_type=corruption_type, severity=severity,
                        inference_filter=inference_filter)
        acc, f1 = eval_model(model, name, dl)
        accs.append(acc)
        f1s.append(f1)
    return float(np.mean(accs)), float(np.mean(f1s))


# ==========================================
# Stage 1（雜訊）：S0~S4 五種策略，5-fold CV，只用Yu3
# ==========================================
NOISE_STAGE1_SELECTION_CONDITIONS_METHODS = [('clean', None), ('awgn', 0), ('salt_pepper', 0.20)]

def stage1_noise_selection(train_raw, stats, label_map, fixed_epochs, defense_epochs):
    groups = np.array([s['group'] for s in train_raw])
    labels = np.array([s['label'] for s in train_raw])
    indices = np.arange(len(train_raw))
    sgkf = StratifiedGroupKFold(n_splits=STAGE1_N_FOLDS, shuffle=True, random_state=STAGE1_SEED)

    epochs = fixed_epochs.get('Yu3', fixed_epochs.get('M5', 50))
    
    # 🌟 V3 新增：建立資料結構來儲存 5 折的詳細數字 (供 Table 23 使用)
    detailed_history = {s: {'clean': [], 'awgn_0': [], 'sp_0.2': []} for s in NOISE_STRATEGIES}
    fold_scores_results = {s: [] for s in NOISE_STRATEGIES} 
    fold_winners = []

    for fold_i, (tr_idx, va_idx) in enumerate(sgkf.split(indices, labels, groups)):
        assert len(set(groups[tr_idx]) & set(groups[va_idx])) == 0, f"[Stage1-noise fold{fold_i}] 機台重疊"
        tr_raw = [train_raw[i] for i in tr_idx]
        va_raw = [train_raw[i] for i in va_idx]

        # 原本的程式碼：
        #random.seed(STAGE1_SEED); np.random.seed(STAGE1_SEED); torch.manual_seed(STAGE1_SEED)
        #clean_model = M5_Model().to(DEVICE)
        
        
        # 🌟 修改為：
        random.seed(STAGE1_SEED); np.random.seed(STAGE1_SEED); torch.manual_seed(STAGE1_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(STAGE1_SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        clean_model = M5_Model().to(DEVICE)
        clean_loader = make_loader(tr_raw, stats, label_map, seed=STAGE1_SEED + fold_i, corruption_type='clean')
        train_model(clean_model, 'Yu3', clean_loader, epochs)

        trained_models = {'S0': clean_model, 'S1': clean_model, 'S2': clean_model}
        for strat in ('S3', 'S4'):
            random.seed(STAGE1_SEED); np.random.seed(STAGE1_SEED); torch.manual_seed(STAGE1_SEED)
            # 🌟 補上這段鎖死 GPU 與 cuDNN
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(STAGE1_SEED)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                
            m = M5_Model().to(DEVICE)
            
            aug_fn, wd = get_training_config('noise', strat)
            loader = DataLoader(
                    Exp4WaveformDataset(tr_raw, stats, label_map, dynamic_augment_fn=aug_fn, seed=STAGE1_SEED + fold_i),
                    batch_size=BATCH_SIZE, shuffle=True,
                    generator=torch.Generator().manual_seed(STAGE1_SEED + fold_i),
                )
            
            # 🌟 新增：讀取該策略的專屬 Epoch，找不到預設 150
            dict_key = f"Yu3_{strat}"
            current_epochs = defense_epochs.get(dict_key, 150)
            
            # 🌟 原本：train_model(m, 'Yu3', loader, epochs, weight_decay=wd)
            train_model(m, 'Yu3', loader, current_epochs, weight_decay=wd)
            trained_models[strat] = m

        this_fold_detailed = {}
        this_fold_results = {}
        
        for strat in NOISE_STRATEGIES:
            model = trained_models[strat]
            inference_filter = strat if strat in ('S1', 'S2') else None

            # 🌟 V3 改寫：獨立抓取並儲存這三個條件的分數
            f1_dict = {}
            for ctype, sev in NOISE_STAGE1_SELECTION_CONDITIONS_METHODS:
                dl = make_loader(va_raw, stats, label_map, seed=STAGE1_SEED, shuffle=False,
                                  corruption_type=ctype, severity=sev, inference_filter=inference_filter)
                _, f1 = eval_model(model, 'Yu3', dl)
                key = 'clean' if ctype == 'clean' else f"{ctype}_{sev}"
                f1_dict[key] = f1
            
            # 存入詳細紀錄字典
            detailed_history[strat]['clean'].append(f1_dict['clean'])
            detailed_history[strat]['awgn_0'].append(f1_dict['awgn_0'])
            detailed_history[strat]['sp_0.2'].append(f1_dict['salt_pepper_0.2'])
            this_fold_detailed[strat] = f1_dict
            
            # 保留原本的 2 條件平均邏輯來決定贏家 (0dB AWGN + 20% 椒鹽)
            severe_mean = np.mean([f1_dict['awgn_0'], f1_dict['salt_pepper_0.2']])
            this_fold_results[strat] = severe_mean
            fold_scores_results[strat].append(severe_mean)

        winner = max(this_fold_results, key=this_fold_results.get) 
        fold_winners.append(winner)
        
        print(f"\n  [Stage1-noise fold{fold_i}] 該fold多數決贏家: {winner}")
        print(f"    ↳ 各策略詳細 Macro-F1 (Clean, AWGN_0, SP_20%):")
        for strat, d in this_fold_detailed.items():
            print(f"      {strat}: {{'Clean': {d['clean']*100:.2f}%, 'AWGN_0dB': {d['awgn_0']*100:.2f}%, 'SP_20%': {d['salt_pepper_0.2']*100:.2f}%}}")

    # 彙總輸出 Mean ± SD (重建 Table 23 使用)
    print("\n[Stage1-noise] 各策略 5-fold 詳細 Macro-F1 (Mean ± SD) 供 Table 23 使用:")
    for strat in NOISE_STRATEGIES:
        c_m, c_s = np.mean(detailed_history[strat]['clean'])*100, np.std(detailed_history[strat]['clean'])*100
        a_m, a_s = np.mean(detailed_history[strat]['awgn_0'])*100, np.std(detailed_history[strat]['awgn_0'])*100
        s_m, s_s = np.mean(detailed_history[strat]['sp_0.2'])*100, np.std(detailed_history[strat]['sp_0.2'])*100
        print(f"  {strat}:")
        print(f"    Clean       : {c_m:.2f} ± {c_s:.2f}%")
        print(f"    AWGN 0dB    : {a_m:.2f} ± {a_s:.2f}%")
        print(f"    Salt-Pepper : {s_m:.2f} ± {s_s:.2f}%")

    vote = Counter(fold_winners)
    top = max(vote.values())
    tied = [s for s, c in vote.items() if c == top]
    selected = tied[0] if len(tied) == 1 else max(
        tied, key=lambda s: np.mean(fold_scores_results[s])
    )
    print(f"\n[Stage1-noise] 逐fold多數決結果: {dict(vote)}")
    print(f"[Stage1-noise] 選定策略: {selected}（跟manuscript記錄的S3做對照確認）")
    return selected


# ==========================================
# Stage 1（缺失資料）：4種策略，5-fold CV，只用Yu3，20%遺失率
# ==========================================
def stage1_missing_selection(train_raw, stats, label_map, fixed_epochs, defense_epochs):
    groups = np.array([s['group'] for s in train_raw])
    labels = np.array([s['label'] for s in train_raw])
    indices = np.arange(len(train_raw))
    sgkf = StratifiedGroupKFold(n_splits=STAGE1_N_FOLDS, shuffle=True, random_state=STAGE1_SEED)

    epochs = fixed_epochs.get('Yu3', fixed_epochs.get('M5', 50))
    win_counts = {s: {'consecutive': 0, 'random': 0} for s in MISSING_STRATEGIES if s != 'no_defense'}
    
    # 🌟 V3 新增：儲存各折 2 個條件的原始 Macro-F1 歷史
    missing_history = {s: {'consecutive': [], 'random': []} for s in MISSING_STRATEGIES}

    for fold_i, (tr_idx, va_idx) in enumerate(sgkf.split(indices, labels, groups)):
        assert len(set(groups[tr_idx]) & set(groups[va_idx])) == 0, f"[Stage1-missing fold{fold_i}] 機台重疊"
        tr_raw = [train_raw[i] for i in tr_idx]
        va_raw = [train_raw[i] for i in va_idx]

        fold_f1 = {}
        for strat in MISSING_STRATEGIES:
            # 原本的程式碼：
            #random.seed(STAGE1_SEED); np.random.seed(STAGE1_SEED); torch.manual_seed(STAGE1_SEED)
            #m = M5_Model().to(DEVICE)
            
            # 🌟 修改為：
            random.seed(STAGE1_SEED); np.random.seed(STAGE1_SEED); torch.manual_seed(STAGE1_SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(STAGE1_SEED)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            m = M5_Model().to(DEVICE)
            aug_fn, wd = get_training_config('missing', strat)
            if aug_fn is not None:
                loader = DataLoader(
                    Exp4WaveformDataset(tr_raw, stats, label_map, dynamic_augment_fn=aug_fn, seed=STAGE1_SEED + fold_i),
                    batch_size=BATCH_SIZE, shuffle=True,
                    generator=torch.Generator().manual_seed(STAGE1_SEED + fold_i),
                )
            else:
                loader = make_loader(tr_raw, stats, label_map, seed=STAGE1_SEED + fold_i, corruption_type='clean')
            # 🌟 新增：動態決定 Epoch
            if strat == 'no_defense':
                current_epochs = fixed_epochs.get('Yu3', fixed_epochs.get('M5', 50))
            else:
                dict_key = f"Yu3_{strat}"
                current_epochs = defense_epochs.get(dict_key, 150)
                
            # 🌟 換成 current_epochs
            train_model(m, 'Yu3', loader, current_epochs, weight_decay=wd)

            for missing_type in ('missing_consecutive', 'missing_random'):
                dl = make_loader(va_raw, stats, label_map, seed=STAGE1_SEED, shuffle=False,
                                corruption_type=missing_type, severity=0.20)
                _, f1 = eval_model(m, 'Yu3', dl)
                fold_f1[(strat, missing_type)] = f1
                
            # 存入詳細紀錄字典
            missing_history[strat]['consecutive'].append(fold_f1[(strat, 'missing_consecutive')])
            missing_history[strat]['random'].append(fold_f1[(strat, 'missing_random')])

        for strat in win_counts:
            for missing_type, key in (('consecutive', 'missing_consecutive'), ('random', 'missing_random')):
                if fold_f1[(strat, key)] > fold_f1[('no_defense', key)]:
                    win_counts[strat][missing_type] += 1

        print(f"\n  [Stage1-missing fold{fold_i}] 完成。各策略F1 (連續20%, 隨機20%):")
        for strat in MISSING_STRATEGIES:
            c_val = fold_f1[(strat, 'missing_consecutive')] * 100
            r_val = fold_f1[(strat, 'missing_random')] * 100
            print(f"    {strat}: 連續={c_val:.2f}%, 隨機={r_val:.2f}%")

    # 🌟 V3 新增：彙總輸出 Mean ± SD (供表格使用)
    print("\n[Stage1-missing] 各策略 5-fold 原始 Macro-F1 (Mean ± SD) 供表格使用:")
    for strat in MISSING_STRATEGIES:
        c_m = np.mean(missing_history[strat]['consecutive']) * 100
        c_s = np.std(missing_history[strat]['consecutive']) * 100
        r_m = np.mean(missing_history[strat]['random']) * 100
        r_s = np.std(missing_history[strat]['random']) * 100
        print(f"  {strat}:")
        print(f"    連續遺失 20%: {c_m:.2f} ± {c_s:.2f}%")
        print(f"    隨機遺失 20%: {r_m:.2f} ± {r_s:.2f}%")

    print(f"\n[Stage1-missing] 各策略相對no_defense的逐fold勝場（滿分{STAGE1_N_FOLDS}）: {win_counts}")
    def total_wins(s):
        return win_counts[s]['consecutive'] + win_counts[s]['random']
    def min_wins(s):
        return min(win_counts[s]['consecutive'], win_counts[s]['random'])

    selected = max(win_counts, key=lambda s: (min_wins(s), total_wins(s)))
    print(f"[Stage1-missing] 選定策略: {selected}（跟manuscript記錄的combined做對照確認）")
    return selected


# ==========================================
# Stage 2：跨模型比較（held-out test set，套用Stage1選定策略）
# ==========================================
def stage2_track(track, selected_strategy, train_raw, test_raw, stats, label_map, fixed_epochs, defense_epochs, conditions):
    results = {}
    aug_fn, wd = get_training_config(track, selected_strategy)
    inference_filter = get_inference_filter(track, selected_strategy)

    # 🌟 V3 新增：自動建立儲存權重的專屬資料夾 (對應老師第 3 點)
    save_dir = os.path.join("saved_models_exp4", track)
    os.makedirs(save_dir, exist_ok=True)

    for name, ModelClass in ALL_MODELS.items():
        dict_key = f"{name}_{selected_strategy}"
        epochs = defense_epochs.get(dict_key, 150)
        
        results[name] = {}
        for corruption_type, severity in conditions:
            cond_key = 'clean' if corruption_type == 'clean' else f"{corruption_type}_{severity}"
            results[name][cond_key] = {'acc': [], 'f1': []}

        for run in range(NUM_RUNS):
            seed = 42 + run
            # 原本的程式碼：
            #random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            #if torch.cuda.is_available():
            #    torch.cuda.manual_seed_all(seed)

            # 🌟 修改為：
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            if aug_fn is not None:
                loader = DataLoader(
                    Exp4WaveformDataset(train_raw, stats, label_map, dynamic_augment_fn=aug_fn, seed=seed),
                    batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(seed),
                )
            else:
                loader = make_loader(train_raw, stats, label_map, seed=seed, corruption_type='clean')

            model = ModelClass().to(DEVICE)
            train_model(model, name, loader, epochs, weight_decay=wd)
            
            # 🌟 V3 新增：訓練完畢後，立刻儲存該 run 的模型權重
            weight_path = os.path.join(save_dir, f"{name}_run_{run+1}.pth")
            torch.save(model.state_dict(), weight_path)

            for corruption_type, severity in conditions:
                cond_key = 'clean' if corruption_type == 'clean' else f"{corruption_type}_{severity}"
                acc, f1 = eval_condition_averaged(model, name, test_raw, stats, label_map, corruption_type,
                                                severity, inference_filter=inference_filter)
                results[name][cond_key]['acc'].append(acc * 100)
                results[name][cond_key]['f1'].append(f1 * 100)

            print(f"  [{track}][{name}] run {run+1}/{NUM_RUNS} 完成 (權重已儲存至 {weight_path})")

        for cond_key, vals in results[name].items():
            acc_mean, acc_std = np.mean(vals['acc']), np.std(vals['acc'])
            f1_mean, f1_std = np.mean(vals['f1']), np.std(vals['f1'])
            
            print(f"  [{track}][{name}] {cond_key}: "
                f"Acc={acc_mean:.2f}±{acc_std:.2f}%  Macro-F1={f1_mean:.2f}±{f1_std:.2f}% (n={NUM_RUNS})")
            raw_f1_rounded = [round(x, 2) for x in vals['f1']]
            print(f"    ↳ 單次Run F1原始數據: {raw_f1_rounded}")

    return results


def noise_conditions():
    conds = [('clean', None)]
    for snr in SEVERITY_GRIDS['awgn']:
        conds.append(('awgn', snr))
    for amt in SEVERITY_GRIDS['salt_pepper']:
        conds.append(('salt_pepper', amt))
    return conds


def missing_conditions():
    conds = [('clean', None)]
    for r in SEVERITY_GRIDS['missing_random']:
        conds.append(('missing_random', r))
    for r in SEVERITY_GRIDS['missing_consecutive']:
        conds.append(('missing_consecutive', r))
    return conds


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


def run_statistical_tests(results, model_names, cond_key, metric='f1'):
    data_matrix = [results[m][cond_key][metric] for m in model_names]
    stat, p_friedman = friedmanchisquare(*data_matrix)
    print(f"\n[{cond_key}] Friedman檢定: chi2={stat:.3f}, p={p_friedman:.4f}")
    if p_friedman >= 0.05:
        print("  → 整體無顯著差異，不進行事後兩兩比較")
        return
    pairs, p_raw = [], []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            try:
                _, p = wilcoxon(results[m1][cond_key][metric], results[m2][cond_key][metric])
            except ValueError:
                p = 1.0
            pairs.append((m1, m2))
            p_raw.append(p)
    p_corrected = holm_bonferroni(np.array(p_raw))
    for (m1, m2), p_c in zip(pairs, p_corrected):
        sig = "**" if p_c < 0.01 else ("*" if p_c < 0.05 else "ns")
        print(f"  {m1} vs {m2}: Holm-corrected p={p_c:.4f} {sig}")


# ==========================================
# 主程式
# ==========================================
if __name__ == "__main__":
    print("=== 讀取原始波形、套用Exp1機台切分、載入固定正規化統計量 ===")
    raw_all = load_plaid_raw_samples()
    train_raw, test_raw = split_by_exp1_record(raw_all)

    if os.path.exists(STATS_JSON) and os.path.exists(LABEL_MAP_JSON):
        with open(STATS_JSON, "r", encoding="utf-8") as f:
            stats = json.load(f)
        with open(LABEL_MAP_JSON, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        print(f"✓ 已讀取 {STATS_JSON} / {LABEL_MAP_JSON}（請先執行pilot_epoch_search_exp4.py產生）")
    else:
        print(f"[警告] 找不到 {STATS_JSON}，請先執行 pilot_epoch_search_exp4.py。這裡臨時重算一次。")
        stats = compute_fixed_norm_stats(train_raw)
        label_map = build_label_map(train_raw)

    if os.path.exists(BEST_EPOCHS_JSON):
        with open(BEST_EPOCHS_JSON, "r", encoding="utf-8") as f:
            best_epochs_raw = json.load(f)
        fixed_epochs = {k: v["fixed_epoch"] for k, v in best_epochs_raw.items()}
        fixed_epochs.setdefault('Yu3', fixed_epochs.get('M5', 50))
    else:
        print(f"[警告] 找不到 {BEST_EPOCHS_JSON}，請先執行 pilot_epoch_search_exp4.py。")
        fixed_epochs = {}

    DEFENSE_EPOCHS_JSON = "best_epochs_defense.json"
    if os.path.exists(DEFENSE_EPOCHS_JSON):
        with open(DEFENSE_EPOCHS_JSON, "r", encoding="utf-8") as f:
            defense_epochs_raw = json.load(f)
        # 轉換成 {'Yu3_S3': 180, ...} 的格式
        defense_epochs = {k: v["fixed_epoch"] for k, v in defense_epochs_raw.items()}
    else:
        print(f"[警告] 找不到 {DEFENSE_EPOCHS_JSON}，請先執行 exp4_defense_pilot_search.py。")
        defense_epochs = {}

    # ---- Stage 1：雜訊策略挑選 ----
    print("\n=== Stage 1（雜訊）：S0~S4 五種防禦策略，5-fold CV，只用Yu3 ===")
    # 🌟 原本：selected_noise_strategy = stage1_noise_selection(train_raw, stats, label_map, fixed_epochs)
    selected_noise_strategy = stage1_noise_selection(train_raw, stats, label_map, fixed_epochs, defense_epochs)
    # 若要直接沿用manuscript已定案的S3而不用本次CV重新選，改成：
    # selected_noise_strategy = FINAL_NOISE_STRATEGY

    # ---- Stage 1：缺失資料策略挑選 ----
    print("\n=== Stage 1（缺失資料）：4種策略，5-fold CV，只用Yu3，20%遺失率 ===")
    selected_missing_strategy = stage1_missing_selection(train_raw, stats, label_map, fixed_epochs, defense_epochs)
    # 若要直接沿用manuscript已定案的combined而不用本次CV重新選，改成：
    # selected_missing_strategy = FINAL_MISSING_STRATEGY

    # ---- Stage 2：雜訊跨模型比較 ----
    print(f"\n=== Stage 2（雜訊）：套用策略={selected_noise_strategy}，跨模型比較 ===")
    noise_results = stage2_track('noise', selected_noise_strategy, train_raw, test_raw,
                                stats, label_map, fixed_epochs, defense_epochs, noise_conditions())
    with open("exp4_stage2_noise_results.json", "w", encoding="utf-8") as f:
        json.dump(noise_results, f, ensure_ascii=False, indent=2)

    # ---- Stage 2：缺失資料跨模型比較 ----
    print(f"\n=== Stage 2（缺失資料）：套用策略={selected_missing_strategy}，跨模型比較 ===")
    missing_results = stage2_track('missing', selected_missing_strategy, train_raw, test_raw,
                                    stats, label_map, fixed_epochs, defense_epochs, missing_conditions())
    with open("exp4_stage2_missing_results.json", "w", encoding="utf-8") as f:
        json.dump(missing_results, f, ensure_ascii=False, indent=2)

    # ---- 統計檢定示範（clean baseline，即Table 22/圖9那組比較）----
    print("\n=== 統計檢定示範：clean baseline（四模型）===")
    run_statistical_tests(noise_results, list(ALL_MODELS.keys()), cond_key='clean', metric='f1')