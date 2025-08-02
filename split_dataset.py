# split_dataset.py
import os
import random
import shutil
from tqdm import tqdm

# --- 1. 参数配置 (请根据您的实际情况修改) ---
TOTAL_NPY_DIR = "data/LIDC_NPY"  # .npy文件总目录

# 输出目录
TRAIN_DIR = "data/CT_Train"
VALID_DIR = "data/CT_Valid"
TEST_DIR  = "data/CT_Test"

# 分割比例
VALID_RATIO = 0.1  # 10% 作为验证集
TEST_RATIO  = 0.1  # 10% 作为测试集
# 剩余的 80% 将作为训练集

# --- 2. 主逻辑 ---
def main():
    hr_source_dir = os.path.join(TOTAL_NPY_DIR, "hr")
    lr_source_dir = os.path.join(TOTAL_NPY_DIR, "lr")

    if not os.path.exists(hr_source_dir):
        print(f"错误：源文件夹不存在 -> {hr_source_dir}")
        return

    # 创建所有目标文件夹
    for path in [TRAIN_DIR, VALID_DIR, TEST_DIR]:
        os.makedirs(os.path.join(path, "hr"), exist_ok=True)
        os.makedirs(os.path.join(path, "lr"), exist_ok=True)

    all_files = os.listdir(hr_source_dir)
    random.shuffle(all_files)

    # 计算分割点
    num_files = len(all_files)
    num_valid = int(num_files * VALID_RATIO)
    num_test = int(num_files * TEST_RATIO)

    valid_files = all_files[:num_valid]
    test_files = all_files[num_valid : num_valid + num_test]
    train_files = all_files[num_valid + num_test:]

    print(f"总计: {num_files} | 训练集: {len(train_files)} | 验证集: {len(valid_files)} | 测试集: {len(test_files)}")

    # 定义一个辅助函数来复制文件
    def copy_files(file_list, dest_dir, desc):
        for filename in tqdm(file_list, desc=desc):
            shutil.copy(os.path.join(hr_source_dir, filename), os.path.join(dest_dir, "hr", filename))
            shutil.copy(os.path.join(lr_source_dir, filename), os.path.join(dest_dir, "lr", filename))

    # 执行复制
    copy_files(train_files, TRAIN_DIR, "Copying to Train")
    copy_files(valid_files, VALID_DIR, "Copying to Valid")
    copy_files(test_files, TEST_DIR, "Copying to Test")

    print("\n数据集分割完成！")

if __name__ == "__main__":
    main()