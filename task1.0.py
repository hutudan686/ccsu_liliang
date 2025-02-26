# -*- coding: utf-8 -*-
"""
DRIVE数据集医学图像超分辨率重建（SRCNN模型）
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
    batch_size = 8
    epochs = 200
    lr = 1e-3
    # 模型参数
    in_channels = 3  # 输入RGB三通道
    out_channels = 3  # 输出RGB三通道
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
        # HR图像变换（输出张量）
        if self.mode == 'train':
            return transforms.Compose([
                transforms.RandomCrop(Config.hr_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            return transforms.Compose([
                transforms.CenterCrop(Config.hr_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def _define_lr_transforms(self):
        # LR图像变换（仅标准化，输入已经是张量）
        return transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __getitem__(self, idx):
        # 1. 加载原始HR图像（PIL格式）
        hr_img = Image.open(self.img_paths[idx]).convert('RGB')

        # 2. 对HR图像应用裁剪和增强（返回张量）
        hr_img = self.hr_transform(hr_img)  # 输出形状 [C, H, W]

        # 3. 生成LR图像（直接在张量上操作）
        lr_img = F.interpolate(
            hr_img.unsqueeze(0),
            scale_factor=1 / Config.scale_factor,
            mode='bicubic',
            align_corners=False
        ).squeeze(0)

        # 4. 对LR张量应用标准化（移除ToTensor）
        lr_img = self.lr_transform(lr_img)

        return lr_img, hr_img

# SRCNN模型（适配RGB三通道）
class SRCNN(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.upscale = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=False),
            nn.Conv2d(3, 3, 3, padding=1)  # 添加卷积平滑插值结果
        )
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 9, padding=4),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 5, padding=2)
        )

    def forward(self, x):
        x = self.upscale(x)  # [B,3,128,128] → [B,3,256,256]
        return self.main(x)


# 计算评价指标（适配RGB三通道）
def compute_metrics(pred, target):
    pred_np = pred.detach().cpu().numpy().transpose(0, 2, 3, 1)  # (B, C, H, W) → (B, H, W, C)
    target_np = target.detach().cpu().numpy().transpose(0, 2, 3, 1)

    psnr_list = [peak_signal_noise_ratio(t, p, data_range=1.0)
                 for p, t in zip(pred_np, target_np)]
    # 修复SSIM计算参数
    ssim_list = [structural_similarity(
        t, p,
        win_size=7,                  # 显式设置窗口大小
        channel_axis=2,              # 替换multichannel=True
        data_range=1.0
    ) for p, t in zip(pred_np, target_np)]
    rmse_list = [np.sqrt(np.mean((t - p)**2))
                 for p, t in zip(pred_np, target_np)]

    return np.mean(psnr_list), np.mean(ssim_list), np.mean(rmse_list)



# 训练函数
def train(model, train_loader, val_loader, config):
    model = model.to(config.device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    best_psnr = 0
    train_loss = []
    val_metrics = {'psnr': [], 'ssim': [], 'rmse': []}

    for epoch in range(config.epochs):
        # 训练阶段
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

        # 验证阶段
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
        scheduler.step(val_rmse)

        # 保存最佳模型
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), "srcnn_drive_best.pth")

        # 打印日志
        print(f"Epoch [{epoch + 1}/{config.epochs}]")
        print(f"Train Loss: {avg_loss:.4f}")
        print(f"Val PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f} | RMSE: {val_rmse:.4f}\n")

    # 保存最终模型
    torch.save(model.state_dict(), "srcnn_drive_final.pth")
    return train_loss, val_metrics


# 可视化结果
def visualize_results(model, test_loader, config, save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i, (lr, hr) in enumerate(test_loader):
            if i >= 5:  # 只可视化前5个样本
                break
            lr = lr.to(config.device)
            sr = model(lr).cpu().squeeze().numpy().transpose(1, 2, 0)
            lr = lr.cpu().squeeze().numpy().transpose(1, 2, 0)
            hr = hr.cpu().squeeze().numpy().transpose(1, 2, 0)

            # 反归一化到[0,1]
            sr = (sr * 0.5 + 0.5).clip(0, 1)
            lr = (lr * 0.5 + 0.5).clip(0, 1)
            hr = (hr * 0.5 + 0.5).clip(0, 1)

            # 保存对比图
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(lr)
            axes[0].set_title("Low Resolution")
            axes[1].imshow(sr)
            axes[1].set_title("Super Resolution")
            axes[2].imshow(hr)
            axes[2].set_title("High Resolution")
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
    model = SRCNN()

    # 训练模型
    train_loss, val_metrics = train(model, train_loader, val_loader, config)

    # 测试最佳模型
    model.load_state_dict(
        torch.load("srcnn_drive_best.pth", map_location=config.device, weights_only=True)
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