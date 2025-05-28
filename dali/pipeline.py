import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def, types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.pytorch import LastBatchPolicy

import os
import cv2
import time
from tqdm import tqdm

def create_file_list_txt(image_dir, output_txt="file_list.txt"):
    with open(f"{image_dir}{output_txt}", "w") as f:
        for filename in sorted(os.listdir(image_dir)):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                f.write(f"{filename} 0\n")  # 0 là label placeholder
                
if not os.path.exists("ship/file_list.txt"):
    create_file_list_txt(image_dir="ship/", output_txt="file_list.txt")
    print("Successfully create file_list.txt")


def loss_func(pred, y):
    pass


def model(x):
    pass


def backward(loss, model):
    pass

import numpy as np

def denormalize(img_tensor):
    """
    img_tensor: numpy array (C, H, W), normalized
    returns: numpy array (H, W, C), uint8
    """
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    img = img_tensor * std + mean        # de-normalize
    img = img * 255.0                    # scale to [0..255]
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))   # C,H,W -> H,W,C for cv2

    return img


@pipeline_def(batch_size=16, num_threads=4, device_id=0)
def dali_pipeline(file_list_path):
    jpegs, labels = fn.readers.file(file_list=file_list_path, random_shuffle=True, name="Reader")
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    
    images = fn.rotate(images, angle=fn.random.uniform(range=(-5, 5)), fill_value=0)
    
    # Brightness Contrast
    images = fn.brightness_contrast(
        images,
        brightness=fn.random.uniform(range=(0.8, 1.2)),
        contrast=fn.random.uniform(range=(0.8, 1.2))
    )
    
    # Resize
    images = fn.resize(images, resize_x=768, resize_y=768)
    
    # Normalization
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        crop=(768, 768),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    )
    
    return images, labels

# data_dir = "/home/lenovo/Documents/workspace/3D-SegClass-Medical/ship"
file_list_path = "ship/file_list.txt"


start_time = time.time()

data_iterator = DALIGenericIterator(
    [dali_pipeline(file_list_path=file_list_path)],
    ['data', 'label'],
    last_batch_policy=LastBatchPolicy.PARTIAL,
    reader_name='Reader'
)

for i, data in tqdm(enumerate(data_iterator)):
    x, y = data[0]['data'], data[0]['label']
    
    if x.shape != (16, 3, 768, 768):
        print("Last batch", x.shape)
    
    # test = x.cpu().numpy()
    # print(f"Batch {i} - shape: {test.shape}")
    # for j, data in enumerate(test):
    #     print(f"Max: {data.max()}")
    #     print(f"Min: {data.min()}")
    #     data_denor = denormalize(data)
    #     print(f"Denormalize max: {data_denor.max()}")
    #     print(f"Denormalize min: {data_denor.min()}")
    #     cv2.imwrite(f"output/test_{j}.png", cv2.cvtColor(data_denor, cv2.COLOR_RGB2BGR))
    #     print()
            
    
    pred = model(x)
    loss = loss_func(pred, y)
    backward(loss, model)


end_time = time.time()

total_time = end_time - start_time

print("Number of batch", i)
print(f"Total time: {total_time:.4f} seconds") 
# Total time: 22.3596 seconds 48.13it/s
