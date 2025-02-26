# -*- coding: utf-8 -*-
"""
DRIVE数据集医学图像超分辨率重建（改进版SRCNN模型）
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
import torch.nn.functional as F


# 配置参数
class Config:
    # 数据参数
    data_dir = "./data/"  # 数据集路径
    hr_size = 256  # 高分辨率图像尺寸
    scale_factor = 2  # 超分辨率缩放因子
    # 训练参数
    batch_size = 16
    epochs = 300
    lr = 1e-3
    # 模型参数
    in_channels = 1  # 输入单通道
    out_channels = 1  # 输出单通道
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 随机种子
    seed = 42
    torch.manual_seed(seed)


# 数据预处理类
class DRIVEDataset(Dataset):
    def __init__(self, img_paths, mode='train'):
        self.img_paths = img_paths
        self.mode = mode
        self.hr_transform = self._define_hr_transforms()
        self.lr_transform = self._define_lr_transforms()

    def __len__(self):
        return len(self.img_paths)

    def _define_hr_transforms(self):
        # 单通道转换
        if self.mode == 'train':
            return transforms.Compose([
                transforms.RandomCrop(Config.hr_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])])
        else:
            return transforms.Compose([
                transforms.CenterCrop(Config.hr_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

    def _define_lr_transforms(self):
        return transforms.Compose([
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __getitem__(self, idx):
        # 加载单通道图像
        hr_img = Image.open(self.img_paths[idx]).convert('L')  # 转为灰度

        # 应用变换
        hr_img = self.hr_transform(hr_img)

        # 生成LR图像
        lr_img = F.interpolate(
            hr_img.unsqueeze(0),
            scale_factor=1 / Config.scale_factor,
            mode='bicubic',
            align_corners=False
        ).squeeze(0)

        lr_img = self.lr_transform(lr_img)

        return lr_img, hr_img


# 增强型SRCNN模型
class EnhancedSRCNN(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        # 特征提取
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(1, 128, 9, padding=4),
            nn.PReLU(),
            nn.Conv2d(128, 64, 5, padding=2),
            nn.PReLU()
        )
        # 上采样
        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.PReLU()
        )
        # 重建
        self.reconstruction = nn.Sequential(
            nn.Conv2d(64, 1, 5, padding=2),
            nn.Tanh()  # 新增激活函数
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.upscale(x)
        x = self.reconstruction(x)
        return x


# 计算评价指标（单通道）
def compute_metrics(pred, target):
    # 反归一化并确保范围
    pred_np = (pred.detach().cpu().numpy() * 0.5 + 0.5).clip(0, 1)
    target_np = (target.detach().cpu().numpy() * 0.5 + 0.5).clip(0, 1)

    psnr_list = []
    ssim_list = []

    for p, t in zip(pred_np, target_np):
        p = p.squeeze()  # 移除通道维度 [1, H, W] → [H, W]
        t = t.squeeze()

        # 计算PSNR
        psnr = peak_signal_noise_ratio(t, p, data_range=1.0)

        # 计算SSIM（关键参数修正）
        ssim = structural_similarity(
            t, p,
            win_size=7,
            data_range=1.0,
            gaussian_weights=True,
            sigma=1.5
        )

        psnr_list.append(psnr)
        ssim_list.append(ssim)

    return np.mean(psnr_list), np.mean(ssim_list), 0  # 暂时忽略RMSE


# 训练函数（增加早停机制）
def train(model, train_loader, val_loader, config):
    model = model.to(config.device)
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_psnr = 0
    patience = 20
    counter = 0

    train_loss = []
    val_metrics = {'psnr': [], 'ssim': [], 'rmse': []}

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        for lr, hr in train_loader:
            lr = lr.to(config.device)
            hr = hr.to(config.device)

            optimizer.zero_grad()
            sr = model(lr)
            loss = criterion(sr, hr)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_loss.append(avg_loss)
        scheduler.step()

        # 验证
        model.eval()
        val_psnr, val_ssim, val_rmse = 0, 0, 0
        with torch.no_grad():
            for lr, hr in val_loader:
                lr = lr.to(config.device)
                hr = hr.to(config.device)
                sr = model(lr)
                psnr, ssim, rmse = compute_metrics(sr, hr)
                val_psnr += psnr
                val_ssim += ssim
                val_rmse += rmse

        val_psnr /= len(val_loader)
        val_ssim /= len(val_loader)
        val_rmse /= len(val_loader)
        val_metrics['psnr'].append(val_psnr)
        val_metrics['ssim'].append(val_ssim)
        val_metrics['rmse'].append(val_rmse)

        # 早停机制
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), "enhanced_srcnn_best.pth")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        print(f"Epoch [{epoch + 1}/{config.epochs}]")
        print(f"Train Loss: {avg_loss:.4f}")
        print(f"Val PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f} | RMSE: {val_rmse:.4f}\n")

    torch.save(model.state_dict(), "enhanced_srcnn_final.pth")
    return train_loss, val_metrics


# 其余函数保持相似，注意调整可视化部分的通道处理
# 可视化结果
def visualize_results(model, test_loader, config, save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i, (lr, hr) in enumerate(test_loader):
            if i >= 5:
                break
            lr = lr.to(config.device)
            sr = model(lr).cpu().squeeze().numpy()
            lr = lr.cpu().squeeze().numpy()
            hr = hr.cpu().squeeze().numpy()

            # 反归一化并转换为uint8
            sr = ((sr * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
            lr = ((lr * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
            hr = ((hr * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

            # 使用最近邻插值放大LR图像
            lr_large = cv2.resize(lr, (hr.shape[1], hr.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

            # 保存对比图
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            axes[0].imshow(lr, cmap='gray')
            axes[0].set_title("LR (Input)")
            axes[1].imshow(lr_large, cmap='gray')
            axes[1].set_title("LR (Upscaled)")
            axes[2].imshow(sr, cmap='gray')
            axes[2].set_title("SR (Output)")
            axes[3].imshow(hr, cmap='gray')
            axes[3].set_title("HR (Ground Truth)")
            plt.savefig(f"{save_dir}/comparison_{i}.png")
            plt.close()

if __name__ == "__main__":
    config = Config()

    # 准备数据集
    train_dir = os.path.join(config.data_dir, "training")
    test_dir = os.path.join(config.data_dir, "test")

    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".tif")]
    val_files = train_files[-5:]  # 取最后5张作为验证集
    train_files = train_files[:-5]
    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".tif")]

    train_set = DRIVEDataset(train_files, mode='train')
    val_set = DRIVEDataset(val_files, mode='val')
    test_set = DRIVEDataset(test_files, mode='test')

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size)
    test_loader = DataLoader(test_set, batch_size=1)

    # 初始化模型
    model = EnhancedSRCNN(scale_factor=Config.scale_factor)
    torch.save(model.state_dict(), "enhanced_srcnn_best.pth")

    # 训练模型
    train_loss, val_metrics = train(model, train_loader, val_loader, config)

    # 测试最佳模型
    model.load_state_dict(
        torch.load("enhanced_srcnn_best.pth",
                   map_location=config.device,
                   weights_only=True)  # 添加安全参数
    )
    # 正确遍历测试集
    model.eval()
    test_psnr, test_ssim, test_rmse = [], [], []
    with torch.no_grad():
        for lr, hr in test_loader:
            lr = lr.to(config.device)
            hr = hr.to(config.device)
            sr = model(lr)
            psnr, ssim, rmse = compute_metrics(sr, hr)
            test_psnr.append(psnr)
            test_ssim.append(ssim)
            test_rmse.append(rmse)

    print(
        f"Test Results - PSNR: {np.mean(test_psnr):.2f} dB | SSIM: {np.mean(test_ssim):.4f} | RMSE: {np.mean(test_rmse):.4f}")

    # 可视化训练过程
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.plot(train_loss)
    plt.title("Training Loss")
    plt.subplot(132)
    plt.plot(val_metrics['psnr'])
    plt.title("Validation PSNR")
    plt.subplot(133)
    plt.plot(val_metrics['ssim'])
    plt.title("Validation SSIM")
    plt.savefig("training_metrics.png")

    # 生成结果对比图
    visualize_results(model, test_loader, config)