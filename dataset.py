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
# """Realize the function of dataset preparation."""
# import gc
# import os
# import queue
# import threading

# import cv2
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm

# import imgproc

# __all__ = [
#     "TrainValidImageDataset", "TestImageDataset",
#     "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
# ]


# class TrainValidImageDataset(Dataset):
#     """Customize the data set loading function and prepare low/high resolution image data in advance.

#     Args:
#         image_dir (str): Train/Valid dataset address.
#         image_size (int): High resolution image size.
#         mode (str): Data set loading method, the training data set is for data enhancement, and the verification data set is not for data enhancement.
#     """

#     def __init__(self, image_dir: str, image_size: int, mode: str) -> None:
#         super(TrainValidImageDataset, self).__init__()
#         # Get all image file names in folder
#         self.lr_image_file_names = [os.path.join(image_dir, "lr", image_file_name) for image_file_name in os.listdir(os.path.join(image_dir, "lr"))]
#         self.hr_image_file_names = [os.path.join(image_dir, "hr", image_file_name) for image_file_name in os.listdir(os.path.join(image_dir, "hr"))]
#         # Specify the high-resolution image size, with equal length and width
#         self.image_size = image_size
#         # Load training dataset or test dataset
#         self.mode = mode

#         # Contains low-resolution and high-resolution image Tensor data
#         self.lr_datasets = []
#         self.hr_datasets = []

#         # preload images into memory
#         self.read_image_to_memory()

#     def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor]:
#         # Read a batch of image data
#         lr_y_image = self.lr_datasets[batch_index]
#         hr_y_image = self.hr_datasets[batch_index]

#         if self.mode == "Train":
#             # Data augment
#             lr_y_image, hr_y_image = imgproc.random_crop(lr_y_image, hr_y_image, self.image_size)
#             lr_y_image, hr_y_image = imgproc.random_rotate(lr_y_image, hr_y_image, angles=[0, 90, 180, 270])
#             lr_y_image, hr_y_image = imgproc.random_horizontally_flip(lr_y_image, hr_y_image, p=0.5)
#             lr_y_image, hr_y_image = imgproc.random_vertically_flip(lr_y_image, hr_y_image, p=0.5)
#         elif self.mode == "Valid":
#             lr_y_image, hr_y_image = imgproc.center_crop(lr_y_image, hr_y_image, self.image_size)
#         else:
#             raise ValueError("Unsupported data processing model, please use `Train` or `Valid`.")

#         # Convert image data into Tensor stream format (PyTorch).
#         # Note: The range of input and output is between [0, 1]
#         lr_y_tensor = imgproc.image2tensor(lr_y_image, range_norm=False, half=False)
#         hr_y_tensor = imgproc.image2tensor(hr_y_image, range_norm=False, half=False)

#         return {"lr": lr_y_tensor, "hr": hr_y_tensor}

#     def __len__(self) -> int:
#         return len(self.lr_image_file_names)

#     def read_image_to_memory(self) -> None:
#         lr_progress_bar = tqdm(self.lr_image_file_names,
#                                total=len(self.lr_image_file_names),
#                                unit="image",
#                                desc=f"Read lr dataset into memory")

#         for lr_image_file_name in lr_progress_bar:
#             # Disabling garbage collection after for loop helps speed things up
#             # gc.disable()

#             # lr_image = cv2.imread(lr_image_file_name, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
#             # # Only extract the image data of the Y channel
#             # lr_y_image = imgproc.bgr2ycbcr(lr_image, use_y_channel=True)
#             # self.lr_datasets.append(lr_y_image)

#             # # After executing append, you need to turn on garbage collection again
#             # gc.enable()

# #单通道读取图片
#             gc.disable()
#             # 直接以灰度模式读取图片
#             lr_image = cv2.imread(lr_image_file_name, cv2.IMREAD_GRAYSCALE)
#             # 检查是否读取成功
#             if lr_image is None:
#                 print(f"警告: 无法读取LR图片 {lr_image_file_name}，跳过。")
#                 continue # 跳过这张有问题的图片
#             # 因为已经是单通道灰度图（Y通道），直接归一化即可
#             lr_y_image = lr_image.astype(np.float32) / 255.
#             self.lr_datasets.append(lr_y_image)
#             gc.enable()
#         hr_progress_bar = tqdm(self.hr_image_file_names,
#                                total=len(self.hr_image_file_names),
#                                unit="image",
#                                desc=f"Read hr dataset into memory")

#         for hr_image_file_name in hr_progress_bar:
#             # Disabling garbage collection after for loop helps speed things up
#             # gc.disable()

#             # hr_image = cv2.imread(hr_image_file_name, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
#             # # Only extract the image data of the Y channel
#             # hr_y_image = imgproc.bgr2ycbcr(hr_image, use_y_channel=True)
#             # self.hr_datasets.append(hr_y_image)

#             # # After executing append, you need to turn on garbage collection again
#             # gc.enable()
#             gc.disable()
#             # 直接以灰度模式读取图片
#             hr_image = cv2.imread(hr_image_file_name, cv2.IMREAD_GRAYSCALE)
#             # 检查是否读取成功
#             if hr_image is None:
#                 print(f"警告: 无法读取HR图片 {hr_image_file_name}，跳过。")
#                 continue # 跳过这张有问题的图片
#             # 因为已经是单通道灰度图（Y通道），直接归一化即可
#             hr_y_image = hr_image.astype(np.float32) / 255.
#             self.hr_datasets.append(hr_y_image)
#             gc.enable()


# class TestImageDataset(Dataset):
#     """Define Test dataset loading methods.

#     Args:
#         test_image_dir (str): Test dataset address for high resolution image dir.
#         upscale_factor (int): Image up scale factor.
#     """

#     def __init__(self, test_image_dir: str, upscale_factor: int) -> None:
#         super(TestImageDataset, self).__init__()
#         # Get all image file names in folder
#         self.image_file_names = [os.path.join(test_image_dir, x) for x in os.listdir(test_image_dir)]
#         # How many times the high-resolution image is the low-resolution image
#         self.upscale_factor = upscale_factor

#         # Contains low-resolution and high-resolution image Tensor data
#         self.lr_datasets = []
#         self.hr_datasets = []

#         # preload images into memory
#         self.read_image_to_memory()

#     def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor]:
#         # Read a batch of image data
#         lr_y_tensor = self.lr_datasets[batch_index]
#         hr_y_tensor = self.hr_datasets[batch_index]

#         return {"lr": lr_y_tensor, "hr": hr_y_tensor}

#     def __len__(self) -> int:
#         return len(self.image_file_names)

#     def read_image_to_memory(self) -> None:
#         progress_bar = tqdm(self.image_file_names,
#                             total=len(self.image_file_names),
#                             unit="image",
#                             desc=f"Read test dataset into memory")

#         # for image_file_name in progress_bar:
#         #     # Disabling garbage collection after for loop helps speed things up
#         #     gc.disable()

#         #     # Read a batch of image data
#         #     # =============================== 这是唯一的修改之处 ===============================
#         #     # 读取图片
#         #     hr_image = cv2.imread(image_file_name, cv2.IMREAD_UNCHANGED)
            
#         #     # 检查并处理4通道图片（例如带透明度的PNG），将其转换为3通道的BGR图片
#         #     if hr_image.shape[2] == 4:
#         #         hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGRA2BGR)
            
#         #     # 转换为浮点数并归一化到 [0, 1] 范围
#         #     hr_image = hr_image.astype(np.float32) / 255.
#         #     # ==============================================================================

#         #     # Use high-resolution image to make low-resolution image
#         #     lr_image = imgproc.imresize(hr_image, 1 / self.upscale_factor)
#         #     lr_image = imgproc.imresize(lr_image, self.upscale_factor)

#         #     # Only extract the image data of the Y channel
#         #     lr_y_image = imgproc.bgr2ycbcr(lr_image, use_y_channel=True)
#         #     hr_y_image = imgproc.bgr2ycbcr(hr_image, use_y_channel=True)

#         #     # Convert image data into Tensor stream format (PyTorch).
#         #     # Note: The range of input and output is between [0, 1]
#         #     lr_y_tensor = imgproc.image2tensor(lr_y_image, range_norm=False, half=False)
#         #     hr_y_tensor = imgproc.image2tensor(hr_y_image, range_norm=False, half=False)

#         #     self.lr_datasets.append(lr_y_tensor)
#         #     self.hr_datasets.append(hr_y_tensor)

#         #     # After executing append, you need to turn on garbage collection again
#             # gc.enable()
# # --- 这是 TestImageDataset 中修改后的代码 ---

#      # --- 这是 TestImageDataset 中修改后的代码 ---

#         for image_file_name in progress_bar:
#             # ...

#             # Read a batch of image data
#             # 以灰度模式读取高清图片
#             hr_image = cv2.imread(image_file_name, cv2.IMREAD_GRAYSCALE)
            
#             # 检查图像是否成功读取
#             if hr_image is None:
#                 print(f"警告：无法读取测试图片 {image_file_name}，已跳过。")
#                 continue

#             # 将图像转换为浮点数并归一化到 [0, 1] 范围
#             hr_image_normalized = hr_image.astype(np.float32) / 255.

#             # 由于已经是单通道灰度图，hr_y_image 就是它本身
#             hr_y_image = hr_image_normalized

#             # 使用高清灰度图制作低分辨率灰度图
#             lr_y_image = imgproc.imresize(hr_y_image, 1 / self.upscale_factor)
#             lr_y_image = imgproc.imresize(lr_y_image, self.upscale_factor)

#             # Convert image data into Tensor stream format (PyTorch).
#             # Note: The range of input and output is between [0, 1]
#             lr_y_tensor = imgproc.image2tensor(lr_y_image, range_norm=False, half=False)
#             hr_y_tensor = imgproc.image2tensor(hr_y_image, range_norm=False, half=False)

#             self.lr_datasets.append(lr_y_tensor)
#             self.hr_datasets.append(hr_y_tensor)

#             # After executing append, you need to turn on garbage collection again
#             gc.enable()

# class PrefetchGenerator(threading.Thread):
#     """A fast data prefetch generator.

#     Args:
#         generator: Data generator.
#         num_data_prefetch_queue (int): How many early data load queues.
#     """

#     def __init__(self, generator, num_data_prefetch_queue: int) -> None:
#         threading.Thread.__init__(self)
#         self.queue = queue.Queue(num_data_prefetch_queue)
#         self.generator = generator
#         self.daemon = True
#         self.start()

#     def run(self) -> None:
#         for item in self.generator:
#             self.queue.put(item)
#         self.queue.put(None)

#     def __next__(self):
#         next_item = self.queue.get()
#         if next_item is None:
#             raise StopIteration
#         return next_item

#     def __iter__(self):
#         return self


# class PrefetchDataLoader(DataLoader):
#     """A fast data prefetch dataloader.

#     Args:
#         num_data_prefetch_queue (int): How many early data load queues.
#         kwargs (dict): Other extended parameters.
#     """

#     def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
#         self.num_data_prefetch_queue = num_data_prefetch_queue
#         super(PrefetchDataLoader, self).__init__(**kwargs)

#     def __iter__(self):
#         return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


# class CPUPrefetcher:
#     """Use the CPU side to accelerate data reading.

#     Args:
#         dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
#     """

#     def __init__(self, dataloader) -> None:
#         self.original_dataloader = dataloader
#         self.data = iter(dataloader)

#     def next(self):
#         try:
#             return next(self.data)
#         except StopIteration:
#             return None

#     def reset(self):
#         self.data = iter(self.original_dataloader)

#     def __len__(self) -> int:
#         return len(self.original_dataloader)


# class CUDAPrefetcher:
#     """Use the CUDA side to accelerate data reading.

#     Args:
#         dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
#         device (torch.device): Specify running device.
#     """

#     def __init__(self, dataloader, device: torch.device):
#         self.batch_data = None
#         self.original_dataloader = dataloader
#         self.device = device

#         self.data = iter(dataloader)
#         self.stream = torch.cuda.Stream()
#         self.preload()

#     def preload(self):
#         try:
#             self.batch_data = next(self.data)
#         except StopIteration:
#             self.batch_data = None
#             return None

#         with torch.cuda.stream(self.stream):
#             for k, v in self.batch_data.items():
#                 if torch.is_tensor(v):
#                     self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

#     def next(self):
#         torch.cuda.current_stream().wait_stream(self.stream)
#         batch_data = self.batch_data
#         self.preload()
#         return batch_data

#     def reset(self):
#         self.data = iter(self.original_dataloader)
#         self.preload()

#     def __len__(self) -> int:
#         return len(self.original_dataloader)


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
"""Realize the function of dataset preparation."""
# import gc
# import os
# import queue
# import threading

# import cv2
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm

# import imgproc

# __all__ = [
#     "TrainValidImageDataset", "TestImageDataset",
#     "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
# ]


# class TrainValidImageDataset(Dataset):
#     """
#     Load training and validation datasets.
#     This version loads images ON-THE-FLY to save memory.
#     """
#     def __init__(self, image_dir: str, image_size: int, mode: str) -> None:
#         super(TrainValidImageDataset, self).__init__()
#         # Get all image file names in folder
#         self.lr_image_file_names = [os.path.join(image_dir, "lr", f) for f in os.listdir(os.path.join(image_dir, "lr"))]
#         self.hr_image_file_names = [os.path.join(image_dir, "hr", f) for f in os.listdir(os.path.join(image_dir, "hr"))]
#         self.image_size = image_size
#         self.mode = mode

#     def __getitem__(self, batch_index: int) -> dict:
#         # Load images in real-time
#         lr_image_path = self.lr_image_file_names[batch_index]
#         hr_image_path = self.hr_image_file_names[batch_index]

#         # Load as grayscale since CT images are single-channel
#         lr_image = cv2.imread(lr_image_path, cv2.IMREAD_GRAYSCALE)
#         hr_image = cv2.imread(hr_image_path, cv2.IMREAD_GRAYSCALE)

#         # Check if images are loaded correctly
#         if lr_image is None:
#             raise IOError(f"Failed to load LR image: {lr_image_path}")
#         if hr_image is None:
#             raise IOError(f"Failed to load HR image: {hr_image_path}")

#         # Normalize images to [0, 1] range
#         lr_y_image = lr_image.astype(np.float32) / 255.0
#         hr_y_image = hr_image.astype(np.float32) / 255.0

#         if self.mode == "Train":
#             # Data augmentation
#             lr_y_image, hr_y_image = imgproc.random_crop(lr_y_image, hr_y_image, self.image_size)
#             lr_y_image, hr_y_image = imgproc.random_rotate(lr_y_image, hr_y_image, angles=[0, 90, 180, 270])
#             lr_y_image, hr_y_image = imgproc.random_horizontally_flip(lr_y_image, hr_y_image, p=0.5)
#             lr_y_image, hr_y_image = imgproc.random_vertically_flip(lr_y_image, hr_y_image, p=0.5)
#         elif self.mode == "Valid":
#             # For validation, usually no random augmentation is needed. If images are already cropped, this can be passed.
#             lr_y_image, hr_y_image = imgproc.center_crop(lr_y_image, hr_y_image, self.image_size)
#         else:
#             raise ValueError("Unsupported mode, please use `Train` or `Valid`.")

#         # Convert to tensor
#         lr_y_tensor = imgproc.image2tensor(lr_y_image, range_norm=False, half=False)
#         hr_y_tensor = imgproc.image2tensor(hr_y_image, range_norm=False, half=False)

#         return {"lr": lr_y_tensor, "hr": hr_y_tensor}

#     def __len__(self) -> int:
#         return len(self.hr_image_file_names)


# class TestImageDataset(Dataset):
#     """
#     Load test dataset.
#     This version loads images ON-THE-FLY to save memory.
#     """
#     def __init__(self, test_hr_image_dir: str, upscale_factor: int) -> None:
#         super(TestImageDataset, self).__init__()
#         # Get all image file names in folder
#         self.hr_image_file_names = [os.path.join(test_hr_image_dir, x) for x in os.listdir(test_hr_image_dir)]
#         self.upscale_factor = upscale_factor

#     def __getitem__(self, batch_index: int) -> dict:
#         hr_image_path = self.hr_image_file_names[batch_index]

#         # Load as grayscale
#         hr_image = cv2.imread(hr_image_path, cv2.IMREAD_GRAYSCALE)

#         if hr_image is None:
#             raise IOError(f"Failed to load test image: {hr_image_path}")

#         # Normalize to [0, 1] range
#         hr_y_image = hr_image.astype(np.float32) / 255.0

#         # Generate LR image from HR image using imresize
#         lr_y_image = imgproc.imresize(hr_y_image, 1 / self.upscale_factor)
#         lr_y_image = imgproc.imresize(lr_y_image, self.upscale_factor)

#         # Convert to tensor
#         lr_y_tensor = imgproc.image2tensor(lr_y_image, range_norm=False, half=False)
#         hr_y_tensor = imgproc.image2tensor(hr_y_image, range_norm=False, half=False)

#         return {"lr": lr_y_tensor, "hr": hr_y_tensor}

#     def __len__(self) -> int:
#         return len(self.hr_image_file_names)


# class PrefetchGenerator(threading.Thread):
#     """A fast data prefetch generator."""
#     def __init__(self, generator, num_data_prefetch_queue: int) -> None:
#         threading.Thread.__init__(self)
#         self.queue = queue.Queue(num_data_prefetch_queue)
#         self.generator = generator
#         self.daemon = True
#         self.start()

#     def run(self) -> None:
#         for item in self.generator:
#             if item is not None:  # Add a check to avoid putting None into queue
#                 self.queue.put(item)
#         self.queue.put(None)

#     def __next__(self):
#         next_item = self.queue.get()
#         if next_item is None:
#             raise StopIteration
#         return next_item

#     def __iter__(self):
#         return self


# class PrefetchDataLoader(DataLoader):
#     """A fast data prefetch dataloader."""
#     def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
#         # Define a collate_fn to filter out None values from the batch
#         def collate_fn_filter_none(batch):
#             batch = list(filter(lambda x: x is not None, batch))
#             if len(batch) == 0:
#                 return None
#             return torch.utils.data.dataloader.default_collate(batch)

#         # Add the custom collate_fn to kwargs
#         kwargs['collate_fn'] = collate_fn_filter_none
#         self.num_data_prefetch_queue = num_data_prefetch_queue
#         super(PrefetchDataLoader, self).__init__(**kwargs)

#     def __iter__(self):
#         return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


# class CPUPrefetcher:
#     """Use the CPU side to accelerate data reading."""
#     def __init__(self, dataloader) -> None:
#         self.original_dataloader = dataloader
#         self.data = iter(dataloader)

#     def next(self):
#         try:
#             return next(self.data)
#         except StopIteration:
#             return None

#     def reset(self):
#         self.data = iter(self.original_dataloader)

#     def __len__(self) -> int:
#         return len(self.original_dataloader)


# class CUDAPrefetcher:
#     """Use the CUDA side to accelerate data reading."""
#     def __init__(self, dataloader, device: torch.device):
#         self.batch_data = None
#         self.original_dataloader = dataloader
#         self.device = device

#         self.data = iter(dataloader)
#         self.stream = torch.cuda.Stream()
#         self.preload()

#     def preload(self):
#         try:
#             self.batch_data = next(self.data)
#         except StopIteration:
#             self.batch_data = None
#             return None
        
#         # Add a check for None batch from collate_fn
#         if self.batch_data is None:
#             return self.preload() # Try to load the next batch

#         with torch.cuda.stream(self.stream):
#             for k, v in self.batch_data.items():
#                 if torch.is_tensor(v):
#                     self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

#     def next(self):
#         torch.cuda.current_stream().wait_stream(self.stream)
#         batch_data = self.batch_data
#         self.preload()
#         return batch_data

#     def reset(self):
#         self.data = iter(self.original_dataloader)
#         self.preload()

#     def __len__(self) -> int:
#         return len(self.original_dataloader)


# dataset.py

# dataset.py

import os
import threading
import queue
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import imgproc

__all__ = [
    "TrainValidImageDataset", "TestImageDataset",
    "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
]

class TrainValidImageDataset(Dataset):
    """
    Loads training and validation datasets ON-THE-FLY from .npy files
    for memory efficiency and high precision.
    """
    def __init__(self, image_dir: str, image_size: int, mode: str) -> None:
        super(TrainValidImageDataset, self).__init__()
        self.image_dir = image_dir
        self.lr_dir = os.path.join(image_dir, "lr")
        self.hr_dir = os.path.join(image_dir, "hr")
        
        if not os.path.exists(self.hr_dir) or not os.path.exists(self.lr_dir):
            raise FileNotFoundError(f"HR or LR directory not found in '{image_dir}'")
            
        self.image_file_names = sorted([f for f in os.listdir(self.hr_dir) if f.endswith('.npy')])
        self.image_size = image_size
        self.mode = mode

    def __getitem__(self, index: int) -> dict:
        filename = self.image_file_names[index]
        lr_path = os.path.join(self.lr_dir, filename)
        hr_path = os.path.join(self.hr_dir, filename)

        try:
            lr_y_image = np.load(lr_path)
            hr_y_image = np.load(hr_path)
        except Exception as e:
            print(f"Error loading .npy file: {hr_path}. Skipping. Error: {e}")
            return None

        if self.mode == "Train":
            lr_y_image, hr_y_image = imgproc.random_crop(lr_y_image, hr_y_image, self.image_size)
            lr_y_image, hr_y_image = imgproc.random_rotate(lr_y_image, hr_y_image, angles=[0, 90, 180, 270])
            lr_y_image, hr_y_image = imgproc.random_horizontally_flip(lr_y_image, hr_y_image, p=0.5)
            lr_y_image, hr_y_image = imgproc.random_vertically_flip(lr_y_image, hr_y_image, p=0.5)
        
        lr_y_tensor = imgproc.image2tensor(lr_y_image, range_norm=False, half=False)
        hr_y_tensor = imgproc.image2tensor(hr_y_image, range_norm=False, half=False)

        return {"lr": lr_y_tensor, "hr": hr_y_tensor}

    def __len__(self) -> int:
        return len(self.image_file_names)


class TestImageDataset(Dataset):
    """
    【最终修正版】
    Loads test HR and LR data ON-THE-FLY from .npy files.
    This ensures consistency with the training and validation datasets.
    """
    def __init__(self, test_dir: str, upscale_factor: int) -> None:
        super(TestImageDataset, self).__init__()
        # test_dir 应该指向包含 'hr' 和 'lr' 子文件夹的目录，例如 "data/CT_Test"
        self.hr_dir = os.path.join(test_dir, 'hr')
        self.lr_dir = os.path.join(test_dir, 'lr')
        
        if not os.path.exists(self.hr_dir) or not os.path.exists(self.lr_dir):
            raise FileNotFoundError(f"HR or LR directory not found in test directory '{test_dir}'")

        self.image_file_names = sorted([f for f in os.listdir(self.hr_dir) if f.endswith('.npy')])
        self.upscale_factor = upscale_factor # upscale_factor 暂时保留，尽管在.npy模式下不直接使用

    def __getitem__(self, index: int) -> dict:
        filename = self.image_file_names[index]

     

        hr_image_path = os.path.join(self.hr_dir, filename)
        lr_image_path = os.path.join(self.lr_dir, filename)
        

        try:
            # 直接从 .npy 文件加载 HR 和 LR
            hr_image = np.load(hr_image_path)
            lr_image = np.load(lr_image_path)
            if index < 5: # 只打印前5个样本的信息就足够了
                print(f"Sample {index}: HR min={np.min(hr_image)}, HR max={np.max(hr_image)}")
                print(f"Sample {index}: LR min={np.min(lr_image)}, LR max={np.max(lr_image)}")
        except Exception as e:
            print(f"Error loading .npy test file: {hr_image_path}. Skipping. Error: {e}")
            return None

        # 将数据转换为 Tensor
        lr_tensor = imgproc.image2tensor(lr_image, range_norm=False, half=False)
        hr_tensor = imgproc.image2tensor(hr_image, range_norm=False, half=False)

      
        return {"lr": lr_tensor, "hr": hr_tensor}

    def __len__(self) -> int:
        return len(self.image_file_names)

# --- Helper classes for fast data loading (Unchanged) ---
# ... (后面的 PrefetchGenerator, CUDAPrefetcher 等类保持原样) ...
class PrefetchGenerator(threading.Thread):
    def __init__(self, generator, num_data_prefetch_queue: int) -> None:
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_data_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self

class PrefetchDataLoader(DataLoader):
    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
        def collate_fn_filter_none(batch):
            batch = list(filter(lambda x: x is not None, batch))
            if len(batch) == 0:
                return None
            return torch.utils.data.dataloader.default_collate(batch)
        
        if 'collate_fn' not in kwargs:
             kwargs['collate_fn'] = collate_fn_filter_none

        self.num_data_prefetch_queue = num_data_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)

class CPUPrefetcher:
    def __init__(self, dataloader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            batch = next(self.data)
            if batch is None:
                return self.next()
            return batch
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)

class CUDAPrefetcher:
    def __init__(self, dataloader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device
        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return

        if self.batch_data is None:
            # Recursively call preload to fetch the next valid batch
            return self.preload()

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
