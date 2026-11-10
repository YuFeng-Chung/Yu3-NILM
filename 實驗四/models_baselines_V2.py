"""
落地部署基準模型：ResNet18 / MobileNetV3-Small / ViT-Tiny
──────────────────────────────────────────────────────────────
修正說明（相對於第一版models_baselines.py）：
  手稿Table 22明確列出「ViT (64×64)」，且全文沒有任何「pretrained／ImageNet／
  transfer learning」字樣，加上ViT的準確率(58.03±7.31% Macro-F1)與變異度都
  明顯符合「從零開始訓練的ViT在小資料集上表現不穩定」這個已知現象——
  三者合起來確認：Exp4的三個baseline都是『從零訓練』，直接吃1-channel、
  64×64輸入，不是ImageNet pretrained、也沒有resize到224。
  第一版用pretrained+224+channel adapter的寫法是錯的，這一版整個換掉。

  ResNet18的參數量(11.1759M)與MobileNetV3-Small的參數量(1.5288M)，換算回
  torchvision標準架構、把最後分類層從1000類改成11類所省下的參數量，數字
  對得起來——代表這兩個baseline用的就是torchvision裡『架構』對應的標準
  ResNet18/MobileNetV3-Small，只是(a)不載入pretrained權重、(b)輸入層改成
  1-channel而非3-channel、(c)輸出層改成11類。

  ViT-Tiny因為輸入是64×64而非ImageNet標準的224×224，patch大小也要跟著改，
  這裡選用patch_size=8（64/8=8，變成8×8=64個patch，跟patch16@224得到14×14
  patch的『每邊patch數』量級相近），維持vit_tiny_patch16_224的embed_dim/
  depth/heads設定（192/12/3），但透過timm的img_size/patch_size/in_chans
  參數覆蓋，pretrained=False。

需要的套件：
    pip install torchvision timm
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models

try:
    import timm
except ImportError:
    timm = None  # 只有在建立 ViT-Tiny 時才會真的用到

NUM_CLASSES = 11
IMG_SIZE = 64  # 跟Yu3系列(M1~M5)、Exp1完全一致，三個baseline不做resize


class ResNet18_Model(nn.Module):
    """ResNet18，from-scratch（無ImageNet pretrained），輸入改為1-channel 64x64。"""
    def __init__(self):
        super().__init__()
        backbone = tv_models.resnet18(weights=None)
        # 原本 conv1: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.fc = nn.Linear(backbone.fc.in_features, NUM_CLASSES)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


class MobileNetV3_Model(nn.Module):
    """MobileNetV3-Small，from-scratch，輸入改為1-channel 64x64。"""
    def __init__(self):
        super().__init__()
        backbone = tv_models.mobilenet_v3_small(weights=None)
        # features[0] 是 ConvNormActivation，其 [0] 是輸入conv
        old_conv = backbone.features[0][0]
        backbone.features[0][0] = nn.Conv2d(
            1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
            stride=old_conv.stride, padding=old_conv.padding, bias=False
        )
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


class ViTTiny_Model(nn.Module):
    """ViT-Tiny，from-scratch，patch_size改小以配合64x64輸入（見檔頭說明）。"""
    def __init__(self, img_size: int = IMG_SIZE, patch_size: int = 8):
        super().__init__()
        if timm is None:
            raise ImportError(
                "找不到 timm 套件，ViT-Tiny 需要它才能建立。請先執行: pip install timm"
            )
        self.backbone = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=False,
            img_size=img_size,
            patch_size=patch_size,
            in_chans=1,
            num_classes=NUM_CLASSES,
        )

    def forward(self, x):
        return self.backbone(x)


BASELINE_MODELS = {
    'ResNet18': ResNet18_Model,
    'MobileNetV3': MobileNetV3_Model,
    'ViT-Tiny': ViTTiny_Model,
}
