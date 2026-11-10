import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# M1~M5 模型定義（從 plaid_m1_to_m5.py 抽出，供訓練腳本與pilot腳本共用，
# 避免兩份程式碼各自維護一份模型定義而漂移不一致）
# NUM_CLASSES 由呼叫端傳入，預設11（PLAID 2014）
# ==========================================
NUM_CLASSES = 11

# ==========================================
# 2. 核心基礎模組 (CBAM 與 Encoder)
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x_cat))

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class BaseEncoder(nn.Module):
    def __init__(self, in_channels=1):
        super(BaseEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        return x 

# ==========================================
# 3. M1 到 M5 模型定義 
# ==========================================
class M1_Model(nn.Module):
    def __init__(self):
        super(M1_Model, self).__init__()
        self.encoder = BaseEncoder(in_channels=1)
        self.fc = nn.Linear(128 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(256, NUM_CLASSES)
    def forward(self, x):
        z = self.encoder(x)
        z_flat = z.view(z.size(0), -1)
        return self.classifier(self.dropout(F.relu(self.fc(z_flat))))

class M2_Model(nn.Module):
    def __init__(self):
        super(M2_Model, self).__init__()
        self.encoder = BaseEncoder(in_channels=1)
        self.cbam = CBAM(128)
        self.fc = nn.Linear(128 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(256, NUM_CLASSES)
    def forward(self, x):
        z = self.encoder(x)
        z_cbam = self.cbam(z) 
        z_flat = z_cbam.view(z_cbam.size(0), -1)
        return self.classifier(self.dropout(F.relu(self.fc(z_flat))))

class M3_Model(nn.Module):
    def __init__(self):
        super(M3_Model, self).__init__()
        self.encoder = BaseEncoder(in_channels=1)
        self.W_phi = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.fc = nn.Linear(128 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(256, NUM_CLASSES)
    def forward(self, x):
        z1 = self.encoder(x)                
        F_fb = self.W_phi(z1)               
        z2 = z1 + F_fb                      
        z_flat = z2.view(z2.size(0), -1)
        return self.classifier(self.dropout(F.relu(self.fc(z_flat))))

class M4_Model(nn.Module):
    def __init__(self):
        super(M4_Model, self).__init__()
        self.encoder = BaseEncoder(in_channels=1)
        self.cbam = CBAM(128)
        self.W_phi = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.fc = nn.Linear(128 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(256, NUM_CLASSES)
    def forward(self, x):
        z_enc = self.encoder(x)
        z1 = self.cbam(z_enc)               
        F_fb = self.W_phi(z1)               
        z2 = z1 + F_fb                      
        z_flat = z2.view(z2.size(0), -1)
        return self.classifier(self.dropout(F.relu(self.fc(z_flat))))

class M5_Model(nn.Module):
    def __init__(self):
        super(M5_Model, self).__init__()
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
        dummy_r = torch.zeros_like(x)
        feat1 = self.encoder(torch.cat([x, dummy_r], dim=1))
        flat1 = feat1.view(feat1.size(0), -1)
        p1 = self.classifier(self.dropout(F.relu(self.fc(flat1))))
        
        recon = F.relu(self.dec_bn1(self.dec_conv1(feat1)))
        recon = F.relu(self.dec_bn2(self.dec_conv2(recon)))
        x_hat = torch.sigmoid(self.dec_conv3(recon))
        
        r = torch.abs(x - x_hat).detach()
        feat2 = self.encoder(torch.cat([x, r], dim=1))
        flat2 = feat2.view(feat2.size(0), -1)
        p2 = self.classifier(self.dropout(F.relu(self.fc(flat2))))
        return p1, p2, x_hat
