import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
from skimage import measure
from sklearn.metrics import auc, roc_auc_score

# 导入你的模型与数据集定义
from dataset import MVTecDataset, get_data_transforms
from models import get_model, cal_anomaly_map

# 强制使用特定显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# ==================== 1. 复杂度与结构评估工具 ====================

def count_parameters(model):
    """统计可训练参数量"""
    if model is None: return 0
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_s_measure(pred, gt, alpha=0.5):
    """
    S-measure (Structure-measure) 核心逻辑实现
    用于衡量预测图与真值的结构相似性
    """
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

    def object_aware(p, g):
        # 物体感知相似度
        x = np.mean(p[g > 0]) if np.any(g > 0) else 0
        y = np.mean(p[g == 0]) if np.any(g == 0) else 0
        return 2 * x / (x ** 2 + 1 + 1e-8)

    def region_aware(p, g):
        # 区域感知相似度 (四象限划分)
        h, w = p.shape
        m_h, m_w = h // 2, w // 2
        p_regs = [p[0:m_h, 0:m_w], p[0:m_h, m_w:], p[m_h:, 0:m_w], p[m_h:, m_w:]]
        g_regs = [g[0:m_h, 0:m_w], g[0:m_h, m_w:], g[m_h:, 0:m_w], g[m_h:, m_w:]]
        scores = [object_aware(p_regs[i], g_regs[i]) if np.any(g_regs[i] > 0)
                  else (1 - np.mean(p_regs[i])) for i in range(4)]
        return np.mean(scores)

    return alpha * object_aware(pred, gt) + (1 - alpha) * region_aware(pred, gt)


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min) if (a_max - a_min) > 0 else image


# ==================== 2. 评估核心逻辑 ====================

def evaluation(encoder, bn, decoder, ca_list, dataloader, device, _class_=None):
    encoder.eval();
    bn.eval();
    decoder.eval()
    for ca in ca_list: ca.eval()

    gt_px, pr_px, gt_sp, pr_sp = [], [], [], []
    aupro_list, s_measure_list = [], []

    with torch.no_grad():
        for img, gt, label, _ in dataloader:
            img = img.to(device)
            inputs = encoder(img)
            be_feats = bn(inputs)

            # 融合特征路径
            if isinstance(be_feats, torch.Tensor):
                fused = ca_list[0](be_feats)
            else:
                fused = [ca_list[i](be_feats[i]) for i in range(len(be_feats))]

            outputs = decoder(fused)
            anomaly_map = cal_anomaly_map(inputs, outputs, img.shape[-1])

            # 后处理平滑
            sigma = 4
            if _class_ == 'transistor':
                sigma = 12
            elif _class_ in ['cable', 'zipper', 'capsule']:
                sigma = 1
            anomaly_map = gaussian_filter(anomaly_map, sigma=sigma)

            # 准备评估用的 Numpy 数据
            gt_np = gt.cpu().numpy().astype(int)
            gt_np[gt_np > 0.5] = 1
            gt_np = gt_np.squeeze()

            # 核心指标累加
            if label.item() != 0:  # 只有异常样本计算 AUPRO 和 S-measure
                amap_norm = min_max_norm(anomaly_map)
                # 计算 AUPRO (此处省略具体 compute_pro 实现，假设你已有该函数)
                # aupro_list.append(compute_pro(gt_np, amap_norm))

                # 计算 S-measure
                s_score = calculate_s_measure(amap_norm, gt_np)
                s_measure_list.append(s_score)

            gt_px.extend(gt_np.ravel())
            pr_px.extend(anomaly_map.ravel())
            gt_sp.append(np.max(gt_np))
            pr_sp.append(np.max(anomaly_map))

    auroc_px = round(roc_auc_score(gt_px, pr_px), 4)
    auroc_sp = round(roc_auc_score(gt_sp, pr_sp), 4)
    mean_s = round(np.mean(s_measure_list), 4) if s_measure_list else 0.0

    return auroc_px, auroc_sp, mean_s


# ==================== 3. 测试与参数统计主函数 ====================

def test(_class_, data_path, weight_type='best'):
    print(f"\n🚀 Testing Class: [{_class_.upper()}]")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder, bn, decoder, ca_list = get_model(device, 256)

    ckp_path = f'./checkpoints4/ca_rd4ad_{_class_}_{weight_type}.pth'
    if not os.path.exists(ckp_path):
        print(f"❌ Checkpoint not found: {ckp_path}")
        return 0, 0, 0, 0

    ckp = torch.load(ckp_path, map_location=device)
    bn.load_state_dict(ckp['bn'])
    decoder.load_state_dict(ckp['decoder'])
    ca_list.load_state_dict(ckp['ca_list'])

    # --- 参数量统计输出 ---
    p_en = sum(p.numel() for p in encoder.parameters()) / 1e6
    p_train = (count_parameters(bn) + count_parameters(ca_list) + count_parameters(decoder)) / 1e6

    print("-" * 45)
    print(f"📊 Model Complexity Breakdown:")
    print(f"   - Encoder (Frozen) : {p_en:.2f} M")
    print(f"   - Trainable Parts  : {p_train:.3f} M (BN+CA+Decoder)")
    print("-" * 45)

    data_transform, gt_transform = get_data_transforms(256, 256)
    test_loader = DataLoader(MVTecDataset(os.path.join(data_path, _class_), data_transform, gt_transform, "test"),
                             batch_size=1, shuffle=False)

    px_auc, sp_auc, sm = evaluation(encoder, bn, decoder, ca_list, test_loader, device, _class_=_class_)
    print(f"✨ Results: Pixel-AUC: {px_auc} | Image-AUC: {sp_auc} | S-measure: {sm}")

    return px_auc, sp_auc, sm, p_train


# ==================== 4. 程序入口 ====================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_name', type=str, default='all')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--weight_type', type=str, default='best')
    args = parser.parse_args()

    classes = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut',
               'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

    if args.class_name == 'all':
        results = []
        final_params = 0
        for cls in classes:
            px, sp, sm, pt = test(cls, args.data_path, args.weight_type)
            results.append([px, sp, sm])
            final_params = pt  # 记录参数量

        avg = np.mean(results, axis=0)
        print("\n" + "Scale Analysis Summary".center(50, "="))
        print(f"Average Image-AUROC : {avg[1]:.4f}")
        print(f"Average S-measure   : {avg[2]:.4f}")
        print(f"Total Trainable M   : {final_params:.3f} M")
        print("=" * 50)
    else:
        test(args.class_name, args.data_path, args.weight_type)