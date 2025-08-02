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

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 这个函数现在只用于【可视化】，不再参与模型计算 ---
def convert_to_displayable_uint8(image_data):
    """将任意范围的图像数据（HU值或[0,1]浮点数）转换为可供显示和评估的uint8图像。"""
    # 如果输入是浮点数[0,1]，先转回[0,255]
    if image_data.dtype == np.float32 or image_data.dtype == np.float64:
        image_data = np.clip(image_data, 0.0, 1.0) * 255.0
        
    # 如果输入是HU值，使用自动窗位
    else:
        lower_bound = np.percentile(image_data, 1.0)
        upper_bound = np.percentile(image_data, 99.0)
        if upper_bound == lower_bound:
            return np.zeros_like(image_data, dtype=np.uint8)
        image_data = np.clip(image_data, lower_bound, upper_bound)
        image_data = (image_data - lower_bound) / (upper_bound - lower_bound) * 255.0
        
    return image_data.astype(np.uint8)


def calculate_psnr(img1_uint8, img2_uint8):
    """计算两张uint8图像之间的PSNR值。"""
    img1 = img1_uint8.astype(np.float64)
    img2 = img2_uint8.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def main():
    # --- 配置 ---
    model_path = "E:/VDSR-PyTorch-master/results/vdsr_ct_lidc_final/best.pth.tar"
    input_path = "E:/VDSR-PyTorch-master/data/CT_Test/hr/LIDC-IDRI-0002_3000522.000000-NA-04919_0009.npy"
    save_path = "E:/VDSR-PyTorch-master/output_image/sr_result_CORRECT.png"
    scale = 3

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    # --- 加载模型 ---
    print("正在加载模型...")
    model = VDSR().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print("模型加载完成。")

    # === 1. 数据加载与准备（严格遵循训练流程） ===
    print(f"正在从 {input_path} 加载并处理图像...")
    
    # a) 加载原始 .npy 数据 (HU值)
    hr_raw_hu = np.load(input_path)
    if hr_raw_hu.ndim != 2: hr_raw_hu = hr_raw_hu[0]
        
    # b) 【核心】直接将HU值线性归一化到[0, 1]浮点数，这才是送入模型的数据
    min_val, max_val = hr_raw_hu.min(), hr_raw_hu.max()
    hr_norm = (hr_raw_hu - min_val) / (max_val - min_val)
    hr_norm = hr_norm.astype(np.float32)

    # c) 在归一化的浮点数上进行下采样和Bicubic插值
    h, w = hr_norm.shape
    lr_norm = cv2.resize(hr_norm, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
    bicubic_norm = cv2.resize(lr_norm, (w, h), interpolation=cv2.INTER_CUBIC)

    # === 2. 模型推理 ===
    print("正在进行VDSR超分辨率重建...")
    input_tensor = torch.from_numpy(bicubic_norm).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        residual = model(input_tensor)
        # 在归一化的浮点数上加上残差
        sr_norm_tensor = input_tensor + residual
    print("超分完成。")

    # 将输出转回numpy浮点数
    sr_norm = sr_norm_tensor.squeeze().cpu().numpy()

    # === 3. 准备可视化和评估 ===
    # 现在才将所有图像（原图、Bicubic、SR结果）转换为可供显示的 uint8 格式
    hr_display = convert_to_displayable_uint8(hr_raw_hu)
    bicubic_display = convert_to_displayable_uint8(bicubic_norm)
    sr_display = convert_to_displayable_uint8(sr_norm)
    
    # 低分辨率图也需要可视化
    lr_display = convert_to_displayable_uint8(lr_norm)
    resized_lr_display = cv2.resize(lr_display, (w, h), interpolation=cv2.INTER_NEAREST)


    # === 4. 计算PSNR ===
    # 在转换后的 uint8 图像上计算PSNR，这符合常规评估标准
    print("正在计算 PSNR...")
    psnr_bicubic = calculate_psnr(bicubic_display, hr_display)
    psnr_vdsr = calculate_psnr(sr_display, hr_display)
    
    print(f"Bicubic PSNR: {psnr_bicubic:.2f} dB")
    print(f"VDSR PSNR: {psnr_vdsr:.2f} dB") # 这次的结果应该会超过Bicubic，接近29dB

    # === 5. 可视化 ===
    print("正在生成对比可视化图...")
    plt.figure(figsize=(20, 5))
    titles = [
        f'低分辨率图 (LR)\n(放大后)', 
        f'Bicubic 插值结果\nPSNR: {psnr_bicubic:.2f} dB', 
        f'VDSR 超分结果 (SR)\nPSNR: {psnr_vdsr:.2f} dB', 
        '高清原图 (HR)'
    ]
    images = [resized_lr_display, bicubic_display, sr_display, hr_display]

    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i], fontsize=14)
        plt.axis('off')

    plt.tight_layout(pad=0.5)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化图已保存至: {save_path}")
    plt.show()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n程序运行中发生未知错误: {e}")