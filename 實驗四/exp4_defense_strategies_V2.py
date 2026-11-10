"""
Exp4 的防禦策略實作：對應手稿 Table 13（雜訊：S0~S4）與
「連續/隨機資料遺失四種處理策略」（未強化基準／Architecture Regularization／
Dynamic Augmentation／Combined）。

⚠️ 手稿正文清楚說明了「屬於哪個階段」(推論時 vs 訓練時)與「概念設計」，
   但沒有寫出動態增強時「雜訊強度怎麼隨機抽樣」的確切數值範圍，也沒有寫
   Architecture Regularization 的weight_decay確切數值。以下標成TODO的
   地方，是本檔案基於手稿描述所做的『合理預設值』，正式報告數據前，
   請跟manuscript或指導教授核對，若有明確數值請直接覆蓋這裡的預設。
"""

import numpy as np
from scipy.ndimage import median_filter as _scipy_median_filter

from exp4_raw_data_V2 import add_awgn_waveform, add_salt_pepper_waveform, \
    add_missing_random_waveform, add_missing_consecutive_waveform, SEVERITY_GRIDS

# ==========================================
# 推論階段濾波（S1、S2）—— 作用在原始波形上，VI影像建構之前
# ==========================================
def median_filter_1d(x, k=3):
    """S1：一維中值濾波，視窗大小k=3。"""
    return _scipy_median_filter(x, size=k, mode='nearest')


def bilateral_filter_1d(x, sigma_spatial=5, sigma_range_ratio=0.1):
    """S2：一維雙邊濾波，sigma_spatial=5、sigma_range改為基於訊號標準差的動態比例。"""
    n = len(x)
    radius = max(1, int(round(3 * sigma_spatial)))
    out = np.empty_like(x, dtype=np.float64)
    
    # 【關鍵修正】根據輸入訊號 x 的真實波動大小，動態計算 sigma_range
    sigma_range = sigma_range_ratio * np.std(x)
    if sigma_range < 1e-6:
        sigma_range = 1e-6  # 避免數值幾乎不變時發生除以零的錯誤
        
    for i in range(n):
        lo, hi = max(0, i - radius), min(n, i + radius + 1)
        idx = np.arange(lo, hi)
        spatial_w = np.exp(-((idx - i) ** 2) / (2 * sigma_spatial ** 2))
        range_w = np.exp(-((x[idx] - x[i]) ** 2) / (2 * sigma_range ** 2))
        w = spatial_w * range_w
        w_sum = w.sum()
        out[i] = (w * x[idx]).sum() / w_sum if w_sum > 1e-12 else x[i]
    return out.astype(x.dtype)


def apply_inference_filter(current, voltage, strategy):
    """S1/S2在推論階段對電壓電流各自獨立做濾波；S0/S3/S4不做任何事（原樣回傳）。"""
    if strategy == 'S1':
        return median_filter_1d(current, k=3), median_filter_1d(voltage, k=3)
    if strategy == 'S2':
        return bilateral_filter_1d(current), bilateral_filter_1d(voltage)
    return current, voltage  # S0 / S3 / S4：推論階段不做額外處理


# ==========================================
# 訓練階段動態增強（S3、S4）—— 每個training sample，以機率p動態加入雜訊
# ==========================================
# TODO：手稿只說「以機率p=0.5動態加入AWGN或脈衝雜訊」，未寫嚴重度怎麼抽樣。
# 這裡預設：加雜訊時，AWGN/Salt-Pepper各50%機率，強度從Table 6的網格中均勻隨機抽一個等級。
NOISE_AUGMENT_CONFIG = {
    'p': 0.5,
    'awgn_choices': SEVERITY_GRIDS['awgn'],
    'sp_choices': SEVERITY_GRIDS['salt_pepper'],
}

# TODO：手稿只說missing-data的Dynamic Augmentation，未寫隨機/連續各佔多少機率。
# 這裡預設：加擾動時，隨機缺失/連續缺失各50%機率，missing ratio從Table 6網格均勻抽樣。
MISSING_AUGMENT_CONFIG = {
    'p': 0.5,
    'ratio_choices': SEVERITY_GRIDS['missing_random'],
}

# TODO：手稿說S4/Combined是「Dynamic Augmentation結合Architecture Regularization
# (Dropout + weight decay)」，但未寫weight_decay確切數值。models_m1_to_m5.py的
# 分類器本來就已經有Dropout(0.5)，這裡的預設是「維持原本0.5不變，只加weight_decay」，
# 如果manuscript其實是把dropout從某個更低的值『提高』到0.5，請告知並修改
# models_m1_to_m5.py / models_baselines.py裡的dropout設定，而不是在這裡硬改。
REGULARIZATION_CONFIG = {
    'weight_decay': 1e-4,
}


def dynamic_noise_augment(current, voltage, stats, rng, config=NOISE_AUGMENT_CONFIG):
    """S3/S4/Combined(noise track)用：訓練時，以機率p動態加AWGN或Salt-Pepper。
    stats用於salt-pepper的固定上下限（跟Stage2評估時用同一組固定統計量，
    避免訓練與測試階段對『雜訊上下限』的定義不一致）。"""
    if rng.random() >= config['p']:
        return current, voltage
    if rng.random() < 0.5:
        snr = rng.choice(config['awgn_choices'])
        return add_awgn_waveform(current, voltage, float(snr), rng)
    else:
        amount = rng.choice(config['sp_choices'])
        return add_salt_pepper_waveform(current, voltage, float(amount), stats, rng)


def dynamic_missing_augment(current, voltage, stats, rng, config=MISSING_AUGMENT_CONFIG):
    """Combined(missing track)用：訓練時，以機率p動態加隨機缺失或連續缺失。"""
    if rng.random() >= config['p']:
        return current, voltage
    ratio = float(rng.choice(config['ratio_choices']))
    if rng.random() < 0.5:
        return add_missing_random_waveform(current, voltage, ratio, rng)
    else:
        return add_missing_consecutive_waveform(current, voltage, ratio, rng)


# ==========================================
# Stage 1 候選策略清單（跟Table 13 / Table 27 一致）
# ==========================================
NOISE_STRATEGIES = ['S0', 'S1', 'S2', 'S3', 'S4']
MISSING_STRATEGIES = ['no_defense', 'regularization', 'augmentation', 'combined']

# 本研究最終選定的策略（見手稿Results 4-B「基於此一多數決考量...採用S3」，
# 以及4-F「本研究選定Combined...」）。Stage2跨模型比較直接套用這兩個，
# 不需要每次都重跑一次Stage1（除非要重現篩選過程本身）。
FINAL_NOISE_STRATEGY = 'S3'          # dynamic noise augmentation（不含額外regularization）
FINAL_MISSING_STRATEGY = 'combined'  # dynamic augmentation + architecture regularization


def get_inference_filter(track, strategy):
    """S1/S2是『推論階段』濾波策略，訓練不變，只有評估時要多做一步濾波。
    Stage2若套用到S1/S2（目前manuscript的最終選擇是S3，不會用到，但保留給
    學生重跑Stage1、選出不同策略時仍然正確）。missing track沒有濾波概念，
    一律回傳None。"""
    if track == 'noise' and strategy in ('S1', 'S2'):
        return strategy
    return None


def get_training_config(track, strategy):
    """回傳(dynamic_augment_fn or None, weight_decay)給訓練迴圈使用。
    track: 'noise' 或 'missing'。"""
    if track == 'noise':
        if strategy in ('S3', 'S4'):
            wd = REGULARIZATION_CONFIG['weight_decay'] if strategy == 'S4' else 0.0
            return dynamic_noise_augment, wd
        return None, 0.0  # S0/S1/S2 訓練階段不做任何事，差異只在推論時的濾波

    if track == 'missing':
        if strategy in ('augmentation', 'combined'):
            wd = REGULARIZATION_CONFIG['weight_decay'] if strategy == 'combined' else 0.0
            return dynamic_missing_augment, wd
        if strategy == 'regularization':
            return None, REGULARIZATION_CONFIG['weight_decay']
        return None, 0.0  # no_defense

    raise ValueError(f"未知的track: {track}")
