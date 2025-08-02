# import os
# import pydicom
# import numpy as np
# import cv2
# from tqdm import tqdm
# import SimpleITK as sitk

# # --- 1. 参数配置 ---
# # !!! 请根据您的实际情况修改以下路径 !!!
# DICOM_ROOT_PATH = "D:/NBIAdata/manifest-1753969228236/LIDC-IDRI"  # 您下载并解压后的 LIDC-IDRI 数据集根目录
# OUTPUT_HR_PATH = "E:/LIDC-for-VDSR/HR"             # 用于存放高清图 (HR) 的输出目录
# OUTPUT_LR_PATH = "E:/LIDC-for-VDSR/LR"             # 用于存放低清图 (LR) 的输出目录

# # CT 窗宽窗位设置 (用于将HU值转为可视化灰度图)
# WINDOW_LEVEL = 40   # 窗位 (Window Level)
# WINDOW_WIDTH = 400  # 窗宽 (Window Width)

# # 低分辨率图像模拟参数
# DOWNSAMPLE_FACTOR = 4  # 定义厚层是薄层的多少倍。例如，用4个1.25mm的切片模拟1个5mm的切片

# # --- 2. 创建输出目录 ---
# os.makedirs(OUTPUT_HR_PATH, exist_ok=True)
# os.makedirs(OUTPUT_LR_PATH, exist_ok=True)


# def get_series_paths(patient_dir):
#     """获取一个病人文件夹下所有的扫描序列文件夹路径"""
#     series_paths = []
#     for root, dirs, files in os.walk(patient_dir):
#         # 如果一个文件夹内包含.dcm文件，就认为它是一个序列
#         if any(f.endswith('.dcm') for f in files):
#             series_paths.append(root)
#     return series_paths


# def get_slice_thickness(series_dir):
#     """读取一个序列的层厚信息"""
#     try:
#         # 读取序列中的第一个dcm文件来获取元数据
#         first_dcm_file = [f for f in os.listdir(series_dir) if f.endswith('.dcm')][0]
#         dcm_path = os.path.join(series_dir, first_dcm_file)
#         dcm_info = pydicom.dcmread(dcm_path)
#         return float(dcm_info.SliceThickness)
#     except Exception as e:
#         # print(f"警告：无法读取层厚信息: {series_dir}, 错误: {e}")
#         return float('inf') # 返回一个极大值，使其在排序中排到最后


# def hu_to_grayscale(volume):
#     """将CT的HU值通过窗宽窗位转换为8位的灰度图像"""
#     min_hu = WINDOW_LEVEL - WINDOW_WIDTH // 2
#     max_hu = WINDOW_LEVEL + WINDOW_WIDTH // 2
    
#     # 将HU值裁剪到窗宽范围内
#     volume = np.clip(volume, min_hu, max_hu)
    
#     # 线性变换到 0-255 范围
#     volume = (volume - min_hu) / (max_hu - min_hu) * 255.0
    
#     # 转换为 uint8 类型
#     return volume.astype(np.uint8)


# def process_patient(patient_id, patient_dir):
#     """处理单个病人的数据"""
#     series_dirs = get_series_paths(patient_dir)
#     if not series_dirs:
#         return 0

#     # 找到该病人所有扫描序列中，层厚最薄的那个
#     series_thickness = [(s_dir, get_slice_thickness(s_dir)) for s_dir in series_dirs]
#     # 按层厚从小到大排序
#     series_thickness.sort(key=lambda x: x[1])
    
#     # 选择最薄的那个序列作为我们的高清数据源
#     thin_slice_dir, thickness = series_thickness[0]
    
#     # 如果最薄的层厚都大于2.0mm，则认为该数据质量不高，跳过
#     if thickness > 2.0:
#         print(f"跳过病人 {patient_id}，最薄层厚为 {thickness}mm，质量不高。")
#         return 0

#     print(f"处理病人 {patient_id}，使用层厚为 {thickness}mm 的序列...")
    
#     # --- 使用 SimpleITK 读取整个 3D 序列 ---
#     reader = sitk.ImageSeriesReader()
#     dicom_names = reader.GetGDCMSeriesFileNames(thin_slice_dir)
#     reader.SetFileNames(dicom_names)
#     image_sitk = reader.Execute()
    
#     # 将 SimpleITK image 转换为 numpy array
#     volume_hu = sitk.GetArrayFromImage(image_sitk) # 格式为 (层数, 高, 宽)
    
#     # 将HU值转换为可视化灰度图
#     volume_gray = hu_to_grayscale(volume_hu)
    
#     num_slices = volume_gray.shape[0]
#     saved_count = 0

#     # --- 遍历所有切片，生成 HR 和 LR 对 ---
#     for i in range(num_slices - DOWNSAMPLE_FACTOR):
#         # 1. 定义 HR 图像
#         # 我们选择合成窗口的中间切片作为HR图像
#         hr_slice_index = i + DOWNSAMPLE_FACTOR // 2
#         hr_image = volume_gray[hr_slice_index]
        
#         # 2. 模拟 LR 图像
#         # 取 DOWNSAMPLE_FACTOR 个连续的薄层切片
#         slices_for_lr = volume_gray[i : i + DOWNSAMPLE_FACTOR]
#         # 计算平均值来模拟厚层扫描
#         lr_image = np.mean(slices_for_lr, axis=0).astype(np.uint8)
        
#         # 3. 保存图像
#         # 用 病人ID-序列号-切片索引 的方式命名，保证唯一性
#         base_filename = f"{patient_id}_{os.path.basename(thin_slice_dir)}_{i:04d}.png"
#         cv2.imwrite(os.path.join(OUTPUT_HR_PATH, base_filename), hr_image)
#         cv2.imwrite(os.path.join(OUTPUT_LR_PATH, base_filename), lr_image)
        
#         saved_count += 1
        
#     return saved_count

# # --- 3. 主函数 ---
# if __name__ == "__main__":
#     patient_ids = [d for d in os.listdir(DICOM_ROOT_PATH) if os.path.isdir(os.path.join(DICOM_ROOT_PATH, d))]
    
#     total_saved = 0
    
#     # 使用 tqdm 创建进度条
#     with tqdm(total=len(patient_ids), desc="处理病人数据") as pbar:
#         for patient_id in patient_ids:
#             patient_dir = os.path.join(DICOM_ROOT_PATH, patient_id)
#             count = process_patient(patient_id, patient_dir)
#             total_saved += count
#             pbar.set_postfix_str(f"已保存 {total_saved} 张图像对")
#             pbar.update(1)
            
#     print(f"\n处理完成！总共生成了 {total_saved} 对 HR/LR 图像。")
#     print(f"HR 图像保存在: {OUTPUT_HR_PATH}")
#     print(f"LR 图像保存在: {OUTPUT_LR_PATH}")

# dcm_converter_to_npy.py
import os
import pydicom
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk

# --- 1. 参数配置 (请务必修改) ---
DICOM_ROOT_PATH = "D:/NBIAdata/manifest-1753969228236/LIDC-IDRI"  # 您下载的数据集根目录
OUTPUT_DIR = "data/LIDC_NPY"                       # 所有 .npy 文件将保存在这里
WINDOW_LEVEL = 40   # 窗位 (Window Level for soft tissue)
WINDOW_WIDTH = 400  # 窗宽 (Window Width for soft tissue)
DOWNSAMPLE_FACTOR = 4  # 用 4 个薄层模拟 1 个厚层

# --- 2. 创建输出目录 ---
HR_PATH = os.path.join(OUTPUT_DIR, "hr")
LR_PATH = os.path.join(OUTPUT_DIR, "lr")
os.makedirs(HR_PATH, exist_ok=True)
os.makedirs(LR_PATH, exist_ok=True)

# (get_series_paths 和 get_slice_thickness 函数与之前相同，这里省略)
def get_series_paths(patient_dir):
    series_paths = []
    for root, dirs, files in os.walk(patient_dir):
        if any(f.endswith('.dcm') for f in files):
            if root not in series_paths:
                series_paths.append(root)
    return series_paths

def get_slice_thickness(series_dir):
    try:
        first_dcm_file = [f for f in os.listdir(series_dir) if f.endswith('.dcm')][0]
        dcm_path = os.path.join(series_dir, first_dcm_file)
        dcm_info = pydicom.dcmread(dcm_path)
        return float(dcm_info.SliceThickness)
    except Exception:
        return float('inf')

def preprocess_hu(volume_hu):
    """将HU值裁剪并归一化到 [0, 1] 的浮点数范围"""
    min_hu = WINDOW_LEVEL - WINDOW_WIDTH // 2
    max_hu = WINDOW_LEVEL + WINDOW_WIDTH // 2
    volume_hu = np.clip(volume_hu, min_hu, max_hu)
    # 线性归一化到 [0.0, 1.0]
    normalized_volume = (volume_hu - min_hu) / (max_hu - min_hu)
    return normalized_volume.astype(np.float32)

def process_patient(patient_id, patient_dir):
    series_dirs = get_series_paths(patient_dir)
    if not series_dirs: return 0
    
    series_thickness = [(s_dir, get_slice_thickness(s_dir)) for s_dir in series_dirs]
    series_thickness.sort(key=lambda x: x[1])
    
    thin_slice_dir, thickness = series_thickness[0]
    if thickness > 2.0: return 0

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(thin_slice_dir)
    if not dicom_names: return 0
    reader.SetFileNames(dicom_names)
    image_sitk = reader.Execute()
    
    volume_hu = sitk.GetArrayFromImage(image_sitk)
    volume_float = preprocess_hu(volume_hu) # 使用新的高精度处理函数
    
    num_slices = volume_float.shape[0]
    saved_count = 0

    for i in range(num_slices - DOWNSAMPLE_FACTOR):
        hr_slice_index = i + DOWNSAMPLE_FACTOR // 2
        hr_image = volume_float[hr_slice_index]
        
        slices_for_lr = volume_float[i : i + DOWNSAMPLE_FACTOR]
        lr_image = np.mean(slices_for_lr, axis=0).astype(np.float32)
        
        base_filename = f"{patient_id}_{os.path.basename(thin_slice_dir)}_{i:04d}"
        np.save(os.path.join(HR_PATH, base_filename + ".npy"), hr_image)
        np.save(os.path.join(LR_PATH, base_filename + ".npy"), lr_image)
        saved_count += 1
        
    return saved_count

if __name__ == "__main__":
    patient_ids = [d for d in os.listdir(DICOM_ROOT_PATH) if os.path.isdir(os.path.join(DICOM_ROOT_PATH, d))]
    total_saved = 0
    with tqdm(total=len(patient_ids), desc="Converting DICOM to .npy") as pbar:
        for patient_id in patient_ids:
            patient_dir = os.path.join(DICOM_ROOT_PATH, patient_id)
            count = process_patient(patient_id, patient_dir)
            total_saved += count
            pbar.set_postfix_str(f"Saved {total_saved} pairs")
            pbar.update(1)
    print(f"\nConversion complete! Total {total_saved} HR/LR pairs saved to '{OUTPUT_DIR}'.")