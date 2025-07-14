import matplotlib.pyplot as plt
import re
import os
import numpy as np

def plot_training_metrics_from_log(log_file_path):
    """
    从训练日志文件中读取 Epoch, PSNR, SSIM, G_Loss, D_Loss，并绘制变化曲线。
    适用于每 Epoch 记录所有指标的日志格式。

    Args:
        log_file_path (str): 训练日志文件的完整路径。
    """
    epochs = []
    psnr_values = []
    ssim_values = []
    g_losses = []
    d_losses = []

    if not os.path.exists(log_file_path):
        print(f"错误：文件未找到 -> {log_file_path}")
        print("请检查文件路径是否正确。")
        return

    print(f"正在读取日志文件：{log_file_path}")

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 使用正则表达式匹配 Epoch, PSNR, SSIM, G_Loss, D_Loss 值
                match = re.search(
                    r'Epoch (\d+): PSNR=([\d.]+), SSIM=([\d.]+), G_Loss=([\d.]+), D_Loss=([\d.]+)', line
                )
                if match:
                    epoch = int(match.group(1))
                    psnr = float(match.group(2))
                    ssim = float(match.group(3))
                    g_loss = float(match.group(4))
                    d_loss = float(match.group(5))

                    epochs.append(epoch)
                    psnr_values.append(psnr)
                    ssim_values.append(ssim)
                    g_losses.append(g_loss)
                    d_losses.append(d_loss)

        if not epochs:
            print("未在日志文件中找到匹配的指标数据（Epoch, PSNR, SSIM, G_Loss, D_Loss）。")
            print("请确认日志文件的格式是否为 'Epoch X: PSNR=..., SSIM=..., G_Loss=..., D_Loss=...'.")
            return

        # --- 绘制损失曲线 ---
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, g_losses, marker='o', linestyle='-', color='orange', label='Generator Loss')
        plt.plot(epochs, d_losses, marker='o', linestyle='-', color='purple', label='Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

        # --- 绘制评估指标曲线 ---
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, psnr_values, marker='o', linestyle='-', color='b', label='PSNR')
        plt.plot(epochs, ssim_values, marker='o', linestyle='-', color='r', label='SSIM')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.title('Validation Metrics Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()


    except Exception as e:
        print(f"读取或处理日志文件时发生错误：{e}")
        print("请检查日志文件格式是否符合预期。")

# 指定你的日志文件路径
log_file_path = r'C:\Users\91144\Desktop\dongtaimohu\training_output\training_log.txt' # 请确保这个路径是正确的

# 运行绘图函数
plot_training_metrics_from_log(log_file_path)
