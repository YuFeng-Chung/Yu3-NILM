"""Explicit small-image models for E2; supplied historical definitions untouched."""
import torch
from torch import nn
from torchvision.models import resnet18, mobilenet_v3_small
from timm.models.vision_transformer import VisionTransformer
from run_e5_ablation import Ablation, loss_function

MODEL_NAMES = ['Yu3', 'ResNet18', 'MobileNetV3', 'ViT-Tiny']


def build_model(name, dropout):
    if name == 'Yu3':
        model = Ablation('err_detach')
        model.dropout.p = dropout
        return model
    if name == 'ResNet18':
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(1,64,3,stride=1,padding=1,bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Sequential(nn.Dropout(dropout),nn.Linear(512,11))
        return model
    if name == 'MobileNetV3':
        model = mobilenet_v3_small(weights=None)
        conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(1,conv.out_channels,3,stride=1,padding=1,bias=False)
        # First inverted residual's depthwise convolution is the second reduction.
        changed = 0
        for module in model.features[1].modules():
            if isinstance(module,nn.Conv2d) and module.stride == (2,2):
                module.stride=(1,1); changed+=1
        assert changed==1
        model.classifier[2].p=dropout
        model.classifier[3]=nn.Linear(model.classifier[3].in_features,11)
        return model
    if name == 'ViT-Tiny':
        model=VisionTransformer(img_size=64,patch_size=8,in_chans=1,num_classes=11,
                                embed_dim=192,depth=6,num_heads=3,mlp_ratio=4,qkv_bias=True)
        model.head=nn.Sequential(nn.Dropout(dropout),model.head)
        return model
    raise ValueError(name)


def final_logits(model, x, name):
    out=model(x)
    return out[0][-1] if name=='Yu3' else out


def training_loss(model, x, y, name):
    out=model(x)
    return loss_function(out,x,y,'err_detach') if name=='Yu3' else nn.functional.cross_entropy(out,y)


def architecture_smoke(device="cpu"):
    """Validate output geometry, classifier dropout and finite backward paths."""
    report={}
    x=torch.randn(2,1,64,64,device=device); y=torch.tensor([0,1],device=device)
    for name in MODEL_NAMES:
        model=build_model(name,.5).to(device)
        features=[]
        module=(model.encoder.pool3 if name=='Yu3' else model.layer4 if name=='ResNet18'
                else model.features if name=='MobileNetV3' else model.patch_embed)
        handle=module.register_forward_hook(lambda m,a,b: features.append(list(b.shape)))
        loss=training_loss(model,x,y,name); loss.backward(); handle.remove()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        model.eval()
        with torch.no_grad(): assert final_logits(model,x,name).shape==(2,11)
        if name!='ViT-Tiny': assert features[0][-2:]==[8,8],(name,features)
        else: assert features[0]==[2,64,192] and len(model.blocks)==6
        report[name]={'feature_shapes':features,'parameters':sum(p.numel() for p in model.parameters()),
                      'finite_forward_backward':True,'head_dropout':.5}
    return report
