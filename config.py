# # Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# # Licensed under the Apache License, Version 2.0 (the "License");
# #   you may not use this file except in compliance with the License.
# #   You may obtain a copy of the License at
# #
# #       http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ==============================================================================
# """Realize the parameter configuration function of dataset, model, training and verification code."""
# import random

# import numpy as np
# import torch
# from torch.backends import cudnn

# # Random seed to maintain reproducible results
# random.seed(0)
# torch.manual_seed(0)
# np.random.seed(0)
# # Use GPU for training by default
# device = torch.device("cuda", 0)
# # Turning on when the image size does not change during training can speed up training
# cudnn.benchmark = True
# # Image magnification factor
# upscale_factor = 2
# # Current configuration parameter method
# mode = "train"
# # Experiment name, easy to save weights and log files
# exp_name = "vdsr_baseline"

# if mode == "train":
#     # Dataset
#     train_image_dir = f"data"
#     valid_image_dir = f"data"
#     test_image_dir = f"data/Set5"

#     image_size = 41
#     batch_size = 16
#     num_workers = 4

#     # Incremental training and migration training
#     start_epoch = 0
#     resume = ""

#     # Total num epochs
#     epochs = 80

#     # SGD optimizer parameter
#     model_lr = 0.1
#     model_momentum = 0.9
#     model_weight_decay = 1e-4
#     model_nesterov = False

#     # StepLR scheduler parameter
#     lr_scheduler_step_size = epochs // 4
#     lr_scheduler_gamma = 0.1

#     # gradient clipping constant
#     clip_gradient = 0.01

#     print_frequency = 200

# if mode == "valid":
#     # Test data address
#     sr_dir = f"results/test/{exp_name}"
#     hr_dir = f"data/Set5/GTmod12"

#     model_path = f"results/{exp_name}/best.pth.tar"



# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Realize the parameter configuration function of dataset, model, training and verification code."""
# import random

# import numpy as np
# import torch
# from torch.backends import cudnn

# # Random seed to maintain reproducible results
# random.seed(0)
# torch.manual_seed(0)
# np.random.seed(0)
# # Use GPU for training by default
# device = torch.device("cuda", 0)
# # Turning on when the image size does not change during training can speed up training
# cudnn.benchmark = True
# # Image magnification factor
# upscale_factor = 2
# # #######################################################################################
# # ##############  这里是关键修改：将模式从 "train" 改为 "valid" ###################
# # #######################################################################################
# mode = "train"
# # Experiment name, easy to save weights and log files
# exp_name = "vdsr_baseline"


# if mode == "train":
#     # Dataset
#     train_image_dir = f"data/train"
#     valid_image_dir = f"data/valid"
#     test_image_dir = f"data/test/hr"

#     image_size = 41
#     batch_size = 16
#     num_workers = 4

#     # Incremental training and migration training
#     start_epoch = 0
#     resume = ""

#     # Total num epochs
#     epochs = 100

#     # SGD optimizer parameter
#     model_lr = 0.1
#     model_momentum = 0.9
#     model_weight_decay = 1e-4
#     model_nesterov = False

#     # StepLR scheduler parameter
#     lr_scheduler_step_size = epochs // 4
#     lr_scheduler_gamma = 0.1

#     # gradient clipping constant
#     clip_gradient = 0.01

#     print_frequency = 200

# if mode == "valid":
#     # #######################################################################################
#     # ##############  这里是关键修改：填写您自己的模型和数据路径 #####################
#     # #######################################################################################
    
#     # 测试数据地址 (请确保Set5文件夹里直接是图片)
#     lr_dir = f"data/test/lr"
#     hr_dir = f"data/test/hr" # 验证时也需要原始高清图做对比

#     # !! 请务必将下面这行中的路径，修改为您自己训练好的 best.pth 的实际路径 !!
#     # 例如 "results/2025-07-10_22-45-00/best.pth"
#     model_path ="E:/VDSR-PyTorch-master/results/vdsr_baseline/best.pth.tar"
#     sr_dir = f"results/test/{exp_name}"

## **第三步：配置文件 (`config.py`)**

# 这个文件需要被修改以指向新的数据集路径和使用更合适的训练超参数。

# **操作**：用以下全部代码**覆盖**您项目中的 `config.py` 文件。

# config.py
import random
import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

# Use GPU for training by default
device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True

# Image magnification factor
# 您可以根据需要将其改为 2, 3, 或 4
upscale_factor = 4 

# Experiment name, easy to save weights and log files
exp_name = "VDSR_CT_FINAL_TRAIN" # 使用一个全新的实验名称，避免与旧的混淆

# Current configuration parameter method
# 当您要训练时，请确保这里是 "train"
# 当您要用 validate.py 或 vdsr_visualize.py 测试时，再将其改为 "valid"
mode = "valid" 

# ==============================================================================
#                              训练模式配置
# ==============================================================================
if mode == "train":
    # --- 数据集路径 ---
    train_image_dir = "data/CT_Train"
    valid_image_dir = "data/CT_Valid"
    test_image_dir  = "data/CT_Test" 

    # --- 训练参数 ---
    image_size = 41   # 训练时裁剪的图像块大小
    batch_size = 64
    num_workers = 4   # 根据您的CPU核心数调整

    # --- 恢复训练设置（保持默认即可） ---
    start_epoch = 0
    resume = ""

    # --- 训练周期 ---
    # 【重要】为了让模型充分学习，建议至少训练 80 轮
    epochs = 200 

    # --- 优化器 (Adam) ---
    # 这是适合Adam的稳定学习率
    model_lr = 1e-3 
    model_betas = (0.9, 0.999)

    # --- 学习率调度器 ---
    # 每 20 个 epoch，学习率衰减为原来的 0.5 倍
    lr_scheduler_step_size = 20
    lr_scheduler_gamma = 0.5 

    # --- 梯度裁剪 ---
    # 这是适合Adam的固定值梯度裁剪
    clip_gradient = 0.1 

    # --- 日志打印频率 ---
    print_frequency = 100

# ==============================================================================
#                              验证/测试模式配置
# ==============================================================================
if mode == "valid":
    # 验证模式，用于评估最终的 best.pth.tar 模型
    sr_dir = f"results/test/{exp_name}"
    # hr_dir 和 model_path 应该指向您成功训练后的结果
    hr_dir = "data/CT_Test/hr" 
    model_path = f"results/{exp_name}/best.pth.tar"