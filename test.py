import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pickle
from PIL import Image
import imageio
import pydicom
import SimpleITK as sitk

def load_npy(file_path: str):
    return np.load(file_path)

def load_npz(file_path: str):
    data = np.load(file_path)
    return data

def load_pickle(file_path: str):
    with open(file_path, "rb") as f:
        data = pickle.load(f) 
    return data

def load_nib(file_path: str):
    nii_data = nib.load(file_path)

    image_data = nii_data.get_fdata()
    return image_data

def save_nib(file_path: str, volume):
    ct_image = sitk.GetImageFromArray(volume)
    sitk.WriteImage(ct_image, file_path)

def load_dcom(file_path: str):
    return pydicom.dcmread(file_path)

def scale_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val - min_val == 0:
        return np.zeros_like(image, dtype=np.uint8)
    
    scaled_image = (image - min_val) / (max_val - min_val) * 255
    
    return scaled_image.astype(np.uint8)

def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)

    if max_val - min_val > 0:
        image = (image - min_val) / (max_val - min_val)

    return image

def lung_window_clip(array: np.ndarray, min_px: int = -1200, max_px: int = 800):
    clipped_array = array.copy()
    clipped_array[clipped_array > max_px] = max_px
    clipped_array[clipped_array < min_px] = min_px
    return clipped_array

def plot_images(array_HU, array_Lung, title1="HU", title2="Lung"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(array_HU, cmap='gray')
    axes[0].set_title(title1)
    axes[0].axis("off")  # Hide axis

    axes[1].imshow(array_Lung, cmap='gray')
    axes[1].set_title(title2)
    axes[1].axis("off")  # Hide axis

def numpy_to_gif(array, folder_path="/data/hpc/dqm/3D-SegClass-Medical/assets", folder_name: str = "image", case_name: str = "3", duration=100):
    folder_path = folder_path + "/" + folder_name
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exists_ok=True)
        
    save_path_x = f"{folder_path}/{case_name}_x.gif"
    # save_path_y = f"{folder_path}/{case_name}_y.gif"
    # save_path_z = f"{folder_path}/{case_name}_z.gif"
    print(save_path_x)
    frames_x = []
    # frames_y = []
    # frames_z = []
    
    for i in range(array.shape[0]):
        img_x = Image.fromarray(scale_image(array[i, :, :])) 
        frames_x.append(img_x)

    frames_x[0].save(save_path_x, save_all=True, append_images=frames_x[1:], duration=duration, loop=0)

def lung_window_clip_volume(pixel_array: np.ndarray, min_px: int = -1200, max_px: int = 800):
    clipped_array = []
    for arr in pixel_array:
        clipped = arr.copy()
        clipped[clipped > max_px] = max_px
        clipped[clipped < min_px] = min_px
        clipped_array.append(clipped)
        
    return np.stack(clipped_array, axis = 0)


case_name = "0029"
image_path = f"/data/hpc/dqm/3D-SegClass-Medical/data/Image/LIDC-IDRI-{case_name}/{case_name}_NI001.npy"
lung_path = f"/data/hpc/dqm/3D-SegClass-Medical/data/Lung_Segmentation/LIDC-IDRI-{case_name}/{case_name}_NI001.npy"
mask_path = f"/data/hpc/dqm/3D-SegClass-Medical/data/Mask/LIDC-IDRI-{case_name}/{case_name}_MA001.npy"

image = load_npy(image_path)
lung = load_npy(lung_path)
mask = load_npy(mask_path) * 255

print("Image, Lung, Mask Shape:", image.shape, lung.shape, mask.shape)
print("Image, Lung, Mask Max:", image.max(), lung.max(), mask.max())
print("Image, Lung, Mask Min:", image.min(), lung.min(), mask.min())

# Convert 3D volume to LIDC Gif
numpy_to_gif(image, case_name=f"{case_name}_image", folder_name="test")
numpy_to_gif(lung, case_name=f"{case_name}_lung", folder_name="test")
numpy_to_gif(mask, case_name=f"{case_name}_mask", folder_name="test")