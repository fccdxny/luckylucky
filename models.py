import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from resnet import wide_resnet50_2
from de_resnet import de_wide_resnet50_2

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=16):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity + identity * a_w * a_h

def get_model(device, image_size=256):
    """ 初始化完整的教师-学生架构 """
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device).eval()
    for param in encoder.parameters(): param.requires_grad = False
    
    # 冻结 BN 的主干参数
    for name, param in bn.named_parameters():
        if any(key in name for key in ["conv", "bn", "layer", "resnet"]):
            param.requires_grad = False
    bn = bn.to(device)

    decoder = de_wide_resnet50_2(pretrained=False).to(device)

    # 动态探测通道数并初始化 R-CAR
    with torch.no_grad():
        dummy_feats = bn(encoder(torch.zeros(1, 3, image_size, image_size).to(device)))
        channels = [dummy_feats.shape[1]] if isinstance(dummy_feats, torch.Tensor) else [f.shape[1] for f in dummy_feats]
    ca_list = nn.ModuleList([CoordAtt(ch, ch) for ch in channels]).to(device)

    return encoder, bn, decoder, ca_list

def cal_anomaly_map(fs_list, ft_list, out_size=256):
    """ 计算异常图：支持多层特征聚合 """
    if isinstance(fs_list, torch.Tensor): fs_list = [fs_list]
    if isinstance(ft_list, torch.Tensor): ft_list = [ft_list]
    
    anomaly_map = np.zeros([out_size, out_size])
    # 按特征图大小降序排列
    fs_sorted = sorted(fs_list, key=lambda x: x.numel(), reverse=True)
    ft_sorted = sorted(ft_list, key=lambda x: x.numel(), reverse=True)

    for i in range(min(len(fs_sorted), len(ft_sorted))):
        fs, ft = fs_sorted[i], ft_sorted[i]
        
        
        if fs.shape != ft.shape:
            # 1. 对齐 H 和 W
            if fs.shape[2:] != ft.shape[2:]:
                ft = F.interpolate(ft, size=fs.shape[2:], mode='bilinear', align_corners=False)
            
            # 2. 对齐通道数 (解决 256 vs 512 的报错)
            if fs.shape[1] != ft.shape[1]:
                min_c = min(fs.shape[1], ft.shape[1])
                fs = fs[:, :min_c, :, :]
                ft = ft[:, :min_c, :, :]

        # 计算余弦距离
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = F.interpolate(a_map.unsqueeze(1), size=out_size, mode='bilinear', align_corners=True)
        anomaly_map += a_map[0, 0, :, :].cpu().detach().numpy()
        
    return anomaly_map
