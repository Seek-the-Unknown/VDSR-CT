# import cv2
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from model import VDSR

# # 设置中文字体支持（防止中文乱码）
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False


# def load_image(path):
#     image = cv2.imread(path)
#     if image is None:
#         raise FileNotFoundError(f"无法读取图片: {path}")
#     return image


# def downsample_image(image, scale):
#     """将高清图下采样得到低分辨率图像（LR）"""
#     h, w = image.shape[:2]
#     lr = cv2.resize(image, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
#     return lr


# def sr_with_vdsr(lr_bgr, hr_size, model, device):
#     """对低分辨率图像进行超分重建"""
#     # resize 回原始尺寸，作为模型输入
#     resized = cv2.resize(lr_bgr, hr_size, interpolation=cv2.INTER_CUBIC)
#     ycrcb = cv2.cvtColor(resized, cv2.COLOR_BGR2YCrCb)
#     y = ycrcb[..., 0].astype(np.float32) / 255.0

#     input_tensor = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).to(device)

#     with torch.no_grad():
#         out = model(input_tensor)
#         sr_y = out.squeeze().cpu().numpy()

#     sr_y = np.clip(sr_y, 0.0, 1.0) * 255.0
#     sr_y = sr_y.astype(np.uint8)

#     cb = cv2.resize(ycrcb[..., 1], sr_y.shape[::-1], interpolation=cv2.INTER_CUBIC)
#     cr = cv2.resize(ycrcb[..., 2], sr_y.shape[::-1], interpolation=cv2.INTER_CUBIC)

#     sr_ycrcb = cv2.merge([sr_y, cb.astype(np.uint8), cr.astype(np.uint8)])
#     sr_bgr = cv2.cvtColor(sr_ycrcb, cv2.COLOR_YCrCb2BGR)

#     return sr_bgr


# def visualize_results(lr, sr, hr, save_path=None):
#     """可视化：低分辨率、超分图、高清原图"""
#     lr_rgb = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
#     sr_rgb = cv2.cvtColor(sr, cv2.COLOR_BGR2RGB)
#     hr_rgb = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)

#     plt.figure(figsize=(15, 5))
#     titles = ['低分辨率图（LR）', 'VDSR 超分结果（SR）', '高清原图（HR）']
#     images = [lr_rgb, sr_rgb, hr_rgb]

#     for i in range(3):
#         plt.subplot(1, 3, i + 1)
#         plt.imshow(images[i])
#         plt.title(titles[i])
#         plt.axis('off')

#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path)
#         print(f"可视化图已保存至: {save_path}")
#     plt.show()


# def main():
#     # 路径配置
#     model_path = "E:/VDSR-PyTorch-master/results/vdsr_baseline/best.pth.tar"
#     input_path = "E:/VDSR-PyTorch-master/data/test/hr/LIDC-IDRI-0002_3000522.000000-NA-04919_0008.png"
#     save_path = "E:/VDSR-PyTorch-master/output_image/sdfsfsfdspng"
#     scale = 3

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     # 加载模型
#     model = VDSR().to(device)
#     checkpoint = torch.load(model_path, map_location=device)
#     model.load_state_dict(checkpoint["state_dict"])
#     model.eval()

#     # 加载原图 → 下采样得到 LR → 超分得到 SR
#     hr = load_image(input_path)
#     lr = downsample_image(hr, scale)
#     sr = sr_with_vdsr(lr, (hr.shape[1], hr.shape[0]), model, device)

#     # 可视化对比
#     visualize_results(lr, sr, hr, save_path=save_path)


# if __name__ == '__main__':
#     main()


# import cv2
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from model import VDSR
# import math # 【新增】导入 math 库用于计算

# # 设置中文字体支持（防止中文乱码）
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False


# def load_image(path):
#     # 【修改】以灰度模式读取图片，因为我们的模型处理的是单通道CT图
#     image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     if image is None:
#         raise FileNotFoundError(f"无法读取图片: {path}")
#     return image


# def downsample_image(image, scale):
#     """将高清图下采样得到低分辨率图像（LR）"""
#     h, w = image.shape[:2]
#     lr = cv2.resize(image, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
#     return lr

# # 【新增】一个专门计算 PSNR 的函数
# def calculate_psnr(img1, img2):
#     """计算两张灰度图之间的PSNR值"""
#     # 确保图像是浮点型，以便进行精确计算
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     mse = np.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return float('inf')
#     return 20 * math.log10(255.0 / math.sqrt(mse))


# def sr_with_vdsr(lr_gray, hr_size, model, device):
#     """对低分辨率图像进行超分重建，并返回重建图和PSNR值"""
#     # 1. 预处理：准备模型输入
#     # 将低分辨率图先用传统方法放大回目标尺寸，得到一张模糊的图
#     bicubic_resized = cv2.resize(lr_gray, hr_size, interpolation=cv2.INTER_CUBIC)
    
#     # 将图像归一化到 [0, 1] 范围
#     y = bicubic_resized.astype(np.float32) / 255.0

#     # 转换为PyTorch Tensor
#     input_tensor = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).to(device)

#     # 2. 模型推理
#     with torch.no_grad():
#         out = model(input_tensor)
#         sr_y_tensor = out.clamp(0.0, 1.0) # 将输出限制在[0,1]
#         sr_y = sr_y_tensor.squeeze().cpu().numpy()

#     # 3. 后处理：将结果转回图像格式
#     sr_gray = (sr_y * 255.0).astype(np.uint8)

#     return sr_gray, bicubic_resized


# def visualize_results(lr, sr, hr, psnr_bicubic, psnr_vdsr, save_path=None): # 【修改】增加psnr参数
#     """可视化：低分辨率、超分图、高清原图，并显示PSNR"""
#     # 为了可视化，将单通道灰度图转换为三通道的RGB图
#     lr_rgb = cv2.cvtColor(lr, cv2.COLOR_GRAY2RGB)
#     sr_rgb = cv2.cvtColor(sr, cv2.COLOR_GRAY2RGB)
#     hr_rgb = cv2.cvtColor(hr, cv2.COLOR_GRAY2RGB)
    
#     # 为了对比，需要将小的LR图也放大到同样尺寸
#     resized_lr_rgb = cv2.resize(lr_rgb, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_NEAREST)

#     plt.figure(figsize=(18, 6))
    
#     # 【修改】更新标题和图像列表
#     titles = [
#         f'低分辨率图 (LR)\n(放大后)', 
#         f'Bicubic 插值结果\nPSNR: {psnr_bicubic:.2f} dB', 
#         f'VDSR 超分结果 (SR)\nPSNR: {psnr_vdsr:.2f} dB', 
#         '高清原图 (HR)'
#     ]
#     # 我们需要一个Bicubic插值的结果图来进行对比
#     bicubic_resized_for_show = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)
#     bicubic_resized_for_show_rgb = cv2.cvtColor(bicubic_resized_for_show, cv2.COLOR_GRAY2RGB)
    
#     images = [resized_lr_rgb, bicubic_resized_for_show_rgb, sr_rgb, hr_rgb]

#     for i in range(4):
#         plt.subplot(1, 4, i + 1)
#         plt.imshow(images[i])
#         plt.title(titles[i], fontsize=14)
#         plt.axis('off')

#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=300) # 提高保存图片的清晰度
#         print(f"可视化图已保存至: {save_path}")
#     plt.show()


# def main():
#     # 路径配置
#     model_path = "E:/VDSR-PyTorch-master/results/vdsr_ct_lidc_final/best.pth.tar"
#     input_path = "E:/VDSR-PyTorch-master/data/CT_Test/hr/LIDC-IDRI-0002_3000522.000000-NA-04919_0009.npy"
#     save_path = "E:/VDSR-PyTorch-master/output_image/sr_result_with_psnr.png" # 【修改】文件名更有意义
#     scale = 3

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     # 加载模型
#     model = VDSR().to(device)
#     # 【修改】增加 weights_only=True 参数，消除FutureWarning
#     checkpoint = torch.load(model_path, map_location=device, weights_only=True)
#     model.load_state_dict(checkpoint["state_dict"])
#     model.eval()

#     # 【修改】整个核心流程重构，以适应灰度图和PSNR计算
#     # 1. 加载高清原图 (HR)，直接以灰度加载
#     hr = load_image(input_path)
    
#     # 2. 对高清图进行下采样，得到低清图 (LR)
#     lr = downsample_image(hr, scale)
    
#     # 3. 使用模型进行超分，得到SR图，同时也得到用于对比的Bicubic插值图
#     sr, bicubic_resized = sr_with_vdsr(lr, (hr.shape[1], hr.shape[0]), model, device)

#     # 4. 计算PSNR值
#     # 计算Bicubic插值结果与原图的PSNR
#     psnr_bicubic = calculate_psnr(bicubic_resized, hr)
#     # 计算VDSR超分结果与原图的PSNR
#     psnr_vdsr = calculate_psnr(sr, hr)
    
#     print(f"Bicubic PSNR: {psnr_bicubic:.2f} dB")
#     print(f"VDSR PSNR: {psnr_vdsr:.2f} dB")

#     # 5. 可视化对比
#     visualize_results(lr, sr, hr, psnr_bicubic, psnr_vdsr, save_path=save_path)


# if __name__ == '__main__':
#     main()

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import VDSR
import math
import os
import config

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def calculate_psnr(img1, img2):
    """在[0, 1]范围的浮点数图像上计算PSNR，与训练时保持一致。"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))

def main():
    # --- 配置 ---
    # 【重要】请确保这里的 exp_name 与您成功训练时 config.py 中的一致
    exp_name = "VDSR_CT_FINAL_TRAIN" 
    model_path = f"results/{exp_name}/best.pth.tar"
    
    # 【重要】指定包含'hr'和'lr'子文件夹的测试集根目录
    test_data_dir = "data/CT_Test"
    # 指定您想可视化的文件名
    image_filename = "LIDC-IDRI-0002_3000522.000000-NA-04919_0036.npy"
    
    save_path = f"output_image/FINAL_RESULT_WITH_LR_{image_filename.replace('.npy', '.png')}"

    # --- 设备和模型加载 ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    print("正在加载模型...")
    model = VDSR().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"模型加载完成: {model_path}")

    # === 1. 数据加载与准备（严格遵循训练验证流程） ===
    hr_path = os.path.join(test_data_dir, "hr", image_filename)
    lr_path = os.path.join(test_data_dir, "lr", image_filename)
    
    print(f"正在加载 HR 图像: {hr_path}")
    print(f"正在加载 LR 图像: {lr_path}")

    # 直接加载预处理好的、[0,1]范围的 HR 和 LR .npy 文件
    hr_norm = np.load(hr_path)
    # bicubic_input_norm 实际上就是训练时所用的LR图像
    bicubic_input_norm = np.load(lr_path) 
    
    # 为了可视化，我们需要一个单独的、从HR动态生成的LR图像
    h_hr, w_hr = hr_norm.shape
    lr_downsampled = cv2.resize(hr_norm, (w_hr // config.upscale_factor, h_hr // config.upscale_factor), interpolation=cv2.INTER_CUBIC)


    # === 2. 模型推理 ===
    print("正在进行VDSR超分辨率重建...")
    input_tensor = torch.from_numpy(bicubic_input_norm).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float)
    
    with torch.no_grad():
        predicted_residual = model(input_tensor)
        sr_tensor = input_tensor + predicted_residual
    print("超分完成。")

    # 将所有图像转为numpy，确保它们都在[0,1]范围
    sr_norm = np.clip(sr_tensor.squeeze().cpu().numpy(), 0, 1)
    
    # === 3. 计算PSNR（在[0,1]浮点数上计算，与训练日志对齐） ===
    print("正在计算 PSNR...")
    psnr_bicubic = calculate_psnr(bicubic_input_norm, hr_norm)
    psnr_vdsr = calculate_psnr(sr_norm, hr_norm)
    
    print(f"Bicubic PSNR: {psnr_bicubic:.2f} dB")
    print(f"VDSR PSNR: {psnr_vdsr:.2f} dB")

    # === 4. 可视化 ===
    # 将[0,1]的浮点图转换为用于显示的[0,255]整数图
    hr_display = (hr_norm * 255.0).astype(np.uint8)
    # 将下采样的LR图放大以供显示
    lr_display_resized = cv2.resize((lr_downsampled * 255.0).astype(np.uint8), (w_hr, h_hr), interpolation=cv2.INTER_NEAREST)
    bicubic_display = (bicubic_input_norm * 255.0).astype(np.uint8)
    sr_display = (sr_norm * 255.0).astype(np.uint8)
    
    print("正在生成对比可视化图...")
    plt.figure(figsize=(20, 5))
    titles = [
        f'低分辨率图 (LR)\n(放大后)',
        f'Bicubic 插值结果\nPSNR: {psnr_bicubic:.2f} dB', 
        f'VDSR 超分结果 (SR)\nPSNR: {psnr_vdsr:.2f} dB', 
        '高清原图 (HR)'
    ]
    images = [lr_display_resized, bicubic_display, sr_display, hr_display]

    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(images[i], cmap='gray', vmin=0, vmax=255)
        plt.title(titles[i], fontsize=14)
        plt.axis('off')

    plt.tight_layout(pad=0.5)
    if not os.path.exists('output_image'):
        os.makedirs('output_image')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"可视化图已保存至: {save_path}")
    plt.show()


if __name__ == '__main__':
    main()