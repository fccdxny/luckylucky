from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import os
import torch
import glob
import numpy as np
import random
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

# ==========================================
#  轻量增强（统一弱增强，避免破坏异常）
# ==========================================

class WeakAugmentation:
    def __init__(self, p=0.5, class_name='default'):
        self.p = p
        self.class_name = class_name
        
        # 1. 严格禁止【水平翻转】的类别 (只有具有严格非对称结构的才禁)
        # 移除了 cable, capsule (它们可以水平翻转)
        self.no_hflip_classes = ['transistor', 'toothbrush', 'pcb1', 'pcb2', 'pcb3', 'pcb4']
        
        # 增加 VisA 中不能垂直翻转的类别（如 pcb 系列和蜡烛）
        self.no_vflip_classes = ['transistor', 'toothbrush', 'bottle', 'candle', 'pcb1', 'pcb2', 'pcb3', 'pcb4']

    def __call__(self, img):
        
        # a. 水平翻转 (左右)
        if self.class_name not in self.no_hflip_classes:
            if random.random() < self.p:
                img = TF.hflip(img)
                
        # b. 垂直翻转 (上下) 
        if self.class_name not in self.no_vflip_classes:
            if random.random() < self.p:
                img = TF.vflip(img)

        # ==========================================
        # 🚀 针对 Cable、Zipper 等高频几何类的专属微调
        # ==========================================
        if self.class_name in ['cable', 'zipper']:
            # 允许轻微的平移和旋转，但绝对不要加 Blur！
            if random.random() < self.p:
                angle = random.uniform(-10, 10) 
                translate_x = random.uniform(-0.03, 0.03) * img.size[0]
                translate_y = random.uniform(-0.03, 0.03) * img.size[1]
                img = TF.affine(img, angle=angle, translate=(translate_x, translate_y), scale=1.0, shear=0)

            # 亮度对比度扰动可以保留，增强光照鲁棒性
            if random.random() < self.p:
                img = TF.adjust_brightness(img, random.uniform(0.8, 1.2)) 
                img = TF.adjust_contrast(img, random.uniform(0.8, 1.2))

        # ==========================================
        # 其他类别的常规增强
        # ==========================================
        else:
            if random.random() < self.p:
                brightness = random.uniform(0.85, 1.15)
                contrast = random.uniform(0.85, 1.15)
                img = TF.adjust_brightness(img, brightness)
                img = TF.adjust_contrast(img, contrast)

        return img


# ==========================================
#  获取数据预处理
#  核心：无裁剪 + 224输入 + CLIP兼容
# ==========================================
def get_data_transforms(size=224, class_name='default', is_train=True):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    #  训练增强（很关键：必须弱！）
    if is_train:
        aug = WeakAugmentation()
    else:
        aug = None

    transform_list = [
        transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
    ]

    if aug is not None:
        transform_list.append(aug)

    transform_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]

    data_transform = transforms.Compose(transform_list)

    #  GT（绝对不能模糊）
    gt_transform = transforms.Compose([
        transforms.Resize((size, size), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    return data_transform, gt_transform


# ==========================================
#  移除随机增强（测试用）
# ==========================================
def remove_augmentation(transform):
    clean = []
    for t in transform.transforms:
        if not isinstance(t, WeakAugmentation):
            clean.append(t)
    return transforms.Compose(clean)


# ==========================================
# MVTec + VisA Dataset
# ==========================================
class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase, class_name='default'):
        self.phase = phase
        self.transform = transform
        self.gt_transform = gt_transform

        if phase == 'test':
            self.transform = remove_augmentation(transform)

        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')

        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()

    def load_dataset(self):
        img_paths = []
        gt_paths = []
        labels = []
        types = []

        defect_types = [
            d for d in os.listdir(self.img_path)
            if os.path.isdir(os.path.join(self.img_path, d))
        ]

        for defect in defect_types:

            imgs = []
            for ext in ['*.png', '*.jpg', '*.JPG', '*.jpeg']:
                imgs.extend(glob.glob(os.path.join(self.img_path, defect, ext)))

            imgs.sort()

            if defect == 'good':
                img_paths.extend(imgs)
                gt_paths.extend([0] * len(imgs))
                labels.extend([0] * len(imgs))
                types.extend(['good'] * len(imgs))

            else:
                for img_path in imgs:
                    name = os.path.splitext(os.path.basename(img_path))[0]

                    gt1 = os.path.join(self.gt_path, defect, name + "_mask.png")
                    gt2 = os.path.join(self.gt_path, defect, name + ".png")

                    if os.path.exists(gt1):
                        gt_path = gt1
                    elif os.path.exists(gt2):
                        gt_path = gt2
                    else:
                        gt_path = 0

                    img_paths.append(img_path)
                    gt_paths.append(gt_path)
                    labels.append(1)
                    types.append(defect)

        return img_paths, gt_paths, labels, types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        gt_path = self.gt_paths[idx]
        label = self.labels[idx]
        img_type = self.types[idx]

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        if gt_path == 0:
            gt = torch.zeros([1, img.shape[1], img.shape[2]])
        else:
            gt = Image.open(gt_path).convert('L')
            gt = self.gt_transform(gt)

        return img, gt, label, img_type
class VisADataset(Dataset):
    def __init__(self, root, transform=None, gt_transform=None, phase="train"):
        """
        root: 类别根目录，例如 '.../VisA_MVTec/candle'
        """
        self.root = root
        self.transform = transform
        self.gt_transform = gt_transform
        self.phase = phase
        
        # 路径定义
        self.img_path = os.path.join(root, phase)
        self.gt_path = os.path.join(root, 'ground_truth')
        
        # 加载数据
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()
        
        # 调试信息：确保你能看到加载了多少图
        print(f"--- VisA [{phase}] 数据集加载成功 ---")
        print(f"数据路径: {self.img_path}")
        print(f"图片数量: {len(self.img_paths)} | 掩码数量: {len([x for x in self.gt_paths if x != 0])}")
        if len(self.img_paths) == 0:
            print(" 警告：未发现图片，请检查路径！")

    def load_dataset(self):
        img_paths, gt_paths, labels, types = [], [], [], []
        
        # 如果路径不存在直接返回空
        if not os.path.exists(self.img_path):
            return img_paths, gt_paths, labels, types

        # 获取子文件夹 (good, bad)
        defect_types = sorted(os.listdir(self.img_path))
        
        for defect in defect_types:
            # 搜索所有图片格式
            folder_path = os.path.join(self.img_path, defect)
            imgs = []
            for ext in ['*.png', '*.jpg', '*.JPG', '*.jpeg']:
                imgs.extend(glob.glob(os.path.join(folder_path, ext)))
            imgs.sort()

            if defect == 'good':
                # 正常样本
                img_paths.extend(imgs)
                gt_paths.extend([0] * len(imgs))
                labels.extend([0] * len(imgs))
                types.extend(['good'] * len(imgs))
            else:
                # 异常样本 (bad 文件夹)
                for img_path in imgs:
                    name = os.path.splitext(os.path.basename(img_path))[0]
                    
                    #  核心：精准匹配你的 _mask.png
                    gt_file = os.path.join(self.gt_path, defect, name + "_mask.png")
                    
                    # 备选：如果有些类没有 _mask 后缀
                    if not os.path.exists(gt_file):
                        gt_file = os.path.join(self.gt_path, defect, name + ".png")

                    if os.path.exists(gt_file):
                        img_paths.append(img_path)
                        gt_paths.append(gt_file)
                        labels.append(1)
                        types.append(defect)
                    else:
                        # 只有在测试阶段找不到 mask 才报警
                        if self.phase == 'test':
                            print(f"跳过异常样本 (找不到掩码): {img_path}")

        return img_paths, gt_paths, labels, types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_p, gt_p, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        
        # 读取原图
        img = Image.open(img_p).convert('RGB')
        if self.transform:
            img = self.transform(img)

        # 处理掩码
        if gt_p == 0:
            # 正常样本返回全黑掩码
            if torch.is_tensor(img):
                gt = torch.zeros([1, img.shape[-2], img.shape[-1]])
            else:
                gt = 0 # 也可以根据需要返回全黑 Image
        else:
            # 异常样本读取 mask 文件
            gt = Image.open(gt_p).convert('L')
            if self.gt_transform:
                gt = self.gt_transform(gt)
            # 确保 mask 是 0 和 1 的二值图
            gt = (torch.tensor(np.array(gt)) > 127).float().unsqueeze(0) if not torch.is_tensor(gt) else (gt > 0).float()

        return img, gt, label, img_type
