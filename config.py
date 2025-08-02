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
upscale_factor = 4
# Current configuration parameter method
mode = "valid"
# Experiment name, easy to save weights and log files
exp_name = "vdsr_ct_lidc_final" # 最终版实验名称

if mode == "train":
    # Dataset
    # 【最终正确配置】训练、验证、测试集均指向同源的CT数据
    train_image_dir = "data/CT_Train"
    valid_image_dir = "data/CT_Valid"
    # 注意：TestImageDataset 需要的是直接包含HR图像的路径
    test_image_dir = "data/CT_Test" 

    image_size = 41  # 训练时裁剪的图像块大小
    batch_size = 64
    num_workers = 4

    start_epoch = 0
    resume = ""

    epochs = 100 # 建议至少50-80轮

    # Optimizer (Adam)
    model_lr = 1e-4 # 使用一个较小的学习率
    model_betas = (0.9, 0.999)

    # Scheduler
    lr_scheduler_step_size = 20
    lr_scheduler_gamma = 0.5 # 可以温和一些，每次衰减一半

    # gradient clipping constant
    clip_gradient = 0.1 # 可以适当放宽

    print_frequency = 100

if mode == "valid":
    # 验证模式，用于评估最终的 best.pth.tar 模型
    sr_dir = f"results/test/{exp_name}"
    hr_dir = "data/CT_Test/hr" # 使用我们独立的CT测试集进行最终评估
    model_path = f"results/{exp_name}/best.pth.tar"