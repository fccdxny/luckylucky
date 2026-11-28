import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from torch.nn import functional as F
import numpy as np
import random
import argparse
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# 导入你的内部模块
from dataset import get_data_transforms, MVTecDataset,VisADataset
from models import get_model  # 从 model.py 导入统一的模型实例化函数
from test import evaluation  # 从 test.py 导入评估函数

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def contrastive_loss_function(clean_feats, student_feats, mask, anomaly_feats=None):
    cos_sim = torch.nn.CosineSimilarity(dim=1)
    loss = 0
    if isinstance(clean_feats, torch.Tensor): clean_feats = [clean_feats]
    if isinstance(student_feats, torch.Tensor): student_feats = [student_feats]
    if anomaly_feats is not None and isinstance(anomaly_feats, torch.Tensor): anomaly_feats = [anomaly_feats]
    
    a_sorted = sorted(clean_feats, key=lambda x: x.numel(), reverse=True)
    b_sorted = sorted(student_feats, key=lambda x: x.numel(), reverse=True)
    if anomaly_feats is not None:
        anom_sorted = sorted(anomaly_feats, key=lambda x: x.numel(), reverse=True)
    
    for i in range(min(len(a_sorted), len(b_sorted))):
        feat_clean = a_sorted[i]
        feat_student = b_sorted[i]
        if feat_clean.shape != feat_student.shape:
            if feat_clean.shape[2:] != feat_student.shape[2:]:
                feat_student = F.interpolate(feat_student, size=feat_clean.shape[2:], mode='bilinear', align_corners=False)
            if feat_clean.shape[1] != feat_student.shape[1]:
                min_c = min(feat_clean.shape[1], feat_student.shape[1])
                feat_clean = feat_clean[:, :min_c, :, :]
                feat_student = feat_student[:, :min_c, :, :]
                
        feat_clean = F.normalize(feat_clean, dim=1)
        feat_student = F.normalize(feat_student, dim=1)        
        
        dist_to_clean = 1 - cos_sim(feat_clean, feat_student)
        resized_mask = F.interpolate(mask, size=dist_to_clean.shape[1:], mode='bilinear', align_corners=False).squeeze(1)
        
        base_loss = torch.mean(dist_to_clean ** 2)
        weight = torch.exp(dist_to_clean.detach()) 
        focused_loss = torch.mean((dist_to_clean ** 2) * weight)
        
        repulsive_loss = 0
        if anomaly_feats is not None:
            feat_anom = anom_sorted[i]
            feat_anom = F.normalize(feat_anom[:, :feat_clean.shape[1], :, :], dim=1) 
            dist_to_anom = 1 - cos_sim(feat_student, feat_anom)
            margin = 0.5
            repulsive_penalty = torch.clamp(margin - dist_to_anom, min=0.0) 
            repulsive_loss = torch.mean(repulsive_penalty * resized_mask * 2.0)
        
        loss += (base_loss + focused_loss + repulsive_loss)
    return loss


def generate_pseudo_anomaly_batch(imgs, device, anomaly_prob=0.2):
    B, C, H, W = imgs.shape
    augmented_imgs = imgs.clone()
    masks = torch.zeros((B, 1, H, W), device=device)
    for i in range(B):
        if random.random() > anomaly_prob: continue
        noise = torch.rand((1, 1, H, W), device=device)
        mask = F.avg_pool2d(noise, kernel_size=7, stride=1, padding=3)
        mask = (mask > 0.5).float()
        mask = F.avg_pool2d(mask, kernel_size=7, stride=1, padding=3)
        if B > 1:
            strategy = random.choice(['shuffle', 'intensity'])
            if strategy == 'shuffle':
                j = random.randint(0, B - 1)
                while j == i: j = random.randint(0, B - 1)
                anomaly_source = imgs[j].clone()
            else:
                anomaly_source = torch.clamp(imgs[i] * random.uniform(0.7, 1.3), 0.0, 1.0)
        else:
            anomaly_source = torch.clamp(imgs[i] * random.uniform(0.7, 1.3), 0.0, 1.0)
        alpha = random.uniform(0.3, 0.7)
        augmented_imgs[i] = imgs[i] * (1 - mask * alpha) + anomaly_source * (mask * alpha)
        masks[i] = mask
    return augmented_imgs, masks

# ==========================================
# 训练主程序
# ==========================================
def train(_class_, data_path):
    print(f"========== Training Class: {_class_} ==========")
    epochs = 200
    learning_rate = 0.001
    batch_size = 16
    image_size = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    train_path = os.path.join(data_path, _class_, 'train')
    if not os.path.exists(train_path): raise FileNotFoundError(f"数据集路径不存在: {train_path}")

    train_data = ImageFolder(root=train_path, transform=data_transform)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    test_path = os.path.join(data_path, _class_)
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    # 1. 统一实例化模型
    encoder, bn, decoder, ca_list = get_model(device, image_size)

    # 2. 【优化器配置】只把 requires_grad=True 的参数传进去
    trainable_params = (
        list(filter(lambda p: p.requires_grad, bn.parameters())) +
        list(decoder.parameters()) +
        list(ca_list.parameters())
    )
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    trainable_m = sum(p.numel() for p in trainable_params) / 1e6
    print(f"Total Trainable Parameters in Optimizer: {trainable_m:.3f} M")

    best_mean_score = 0.0
    os.makedirs('./checkpoints4/', exist_ok=True)
    best_ckp_path = f'./checkpoints4/ca_rd4ad_{_class_}_best.pth'

    for epoch in range(epochs):
        bn.train()
        decoder.train()
        ca_list.train()
        loss_list = []

        for img, _ in train_dataloader:
            img = img.to(device)
           
            augmented_img, mask = generate_pseudo_anomaly_batch(img, device, anomaly_prob=0.2)
            
            with torch.no_grad():
                clean_inputs = encoder(img) 
                aug_inputs = encoder(augmented_img)
            
            # 前向传播
            be_feats = bn(aug_inputs)
            if isinstance(be_feats, torch.Tensor):
                fused_feats = ca_list[0](be_feats)
            else:
                fused_feats = [ca_list[i](be_feats[i]) for i in range(len(be_feats))]
            
            outputs = decoder(fused_feats)
            loss = contrastive_loss_function(clean_inputs, outputs, mask, anomaly_feats=aug_inputs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {np.mean(loss_list):.4f}')
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            auroc_px, auroc_sp, aupro_px = evaluation(
                encoder, bn, decoder, ca_list, test_dataloader, device, _class_=_class_
            )
            current_mean = (auroc_px + auroc_sp + aupro_px) / 3
            print(f'\n--- Epoch [{epoch + 1}] Evaluation ---')
            print(f'PX-AUROC: {auroc_px:.3f} | SP-AUROC: {auroc_sp:.3f} | PX-AUPRO: {aupro_px:.3f}')

            if current_mean > best_mean_score:
                best_mean_score = current_mean
                save_dict = {
                    'epoch': epoch + 1,
                    'bn': bn.state_dict(),
                    'decoder': decoder.state_dict(),
                    'ca_list': ca_list.state_dict()
                }
                torch.save(save_dict, best_ckp_path)
                print(f'>>> Saved Best Weights: {best_mean_score:.4f}')
    
    return best_mean_score

if __name__ == '__main__':
    setup_seed(111)
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_name', type=str, default='transistor')
    parser.add_argument('--data_path', type=str, required=True)
    args = parser.parse_args()

    item_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                 'transistor', 'metal_nut', 'screw','toothbrush', 'zipper', 'tile', 'wood']
                 
    if args.class_name == 'all':
        for i in item_list: train(i, args.data_path)
    else:
        train(args.class_name, args.data_path)
