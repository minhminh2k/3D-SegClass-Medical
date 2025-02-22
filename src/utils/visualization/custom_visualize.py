import torch
import matplotlib
import numpy as np
import nibabel as nb
from tqdm import tqdm
from matplotlib import gridspec
import matplotlib.pyplot as plt
from typing import Any
from io import BytesIO
from matplotlib.lines import Line2D
from torchvision.utils import draw_segmentation_masks
import cv2
import scipy.ndimage as ndimage
import matplotlib.colors as mcolors

def custom_visualization(
    image: Any,
    label: Any,
    pred: Any,
    num_classes: int,
    frame_nums: list[int],
    class_names: list[str],
    colors: list[str],
    fig_size: tuple[int],
    vis_pred: bool = True,
):
    # Custom frame_nums
    # ...
    return render_and_save_gridspec(
        image=image,
        label=label,
        pred=pred,
        frame_nums=frame_nums,
        num_classes=num_classes,
        class_names=class_names,
        colors=colors,
        img_normalize=True,
        transparency=0.65,
        fig_size=fig_size,
        vis_pred=vis_pred
    )
    
def gif_visualization(
    image: Any,
    label: Any,
    num_classes: int,
    colors: list[str],
    transparency: float = 0.4,
):
    visualize_volume = []
    
    for image_data, mask in zip(image, label):
        overlayed_img = overlay(
            image=image_data,
            seg_mask=mask,
            num_classes=num_classes,
            colors=colors,
            normalize=True,
            transparency=transparency,
        )
        visualize_volume.append(overlayed_img)
    
    # return np.stack(visualize_volume, axis=0)
    return visualize_volume

def render_and_save_gridspec(
    image: Any,
    label: Any,
    pred: Any,
    frame_nums: list,
    num_classes: int,
    class_names: list,
    colors: list,
    img_normalize: bool,
    transparency: float = 1.0,
    fig_size: tuple[int] = [2, 2],
    vis_pred: bool = True,
):
    # plot setting
    rows = fig_size[0]
    cols = fig_size[1]
    fig = plt.figure(figsize=(cols, rows))
    gs = gridspec.GridSpec(rows, cols, wspace=0.01, hspace=0.01)
    
    if vis_pred is False:
        pred_masks = [label]
    else:
        pred_masks = [label, pred]
    # rendering a fig_size x fig_size grid plot.
    for row, frame_ind in enumerate(frame_nums):
        for col, pred_mask in enumerate(pred_masks):
            seg_mask_volume = pred_mask
            axis = plt.subplot(gs[row, col])
            mask = seg_mask_volume[:, :, frame_ind]
            image_data = image[:, :, frame_ind]
            
            overlayed_img = overlay(
                image=image_data,
                seg_mask=mask,
                num_classes=num_classes,
                colors=colors,
                normalize=img_normalize,
                transparency=transparency,
            )
            axis.imshow(overlayed_img, cmap="bone")
            axis.axis("off")
            if row == 0 and col == 0:
                axis.set_title("Ground Truth", fontsize=3, pad=2)
            elif row == 0 and col == 1:
                axis.set_title("Predict", fontsize=3, pad=2)
            # elif row == 0 and col == 2:
            #     axis.set_title("nnFormer", fontsize=5, pad=2)
            # elif row == 0 and col == 3:
            #     axis.set_title("Unetr", fontsize=5, pad=2)
    # generate the legend
    legend_ax = plt.subplot(gs[fig_size[0] - 1, fig_size[1] - 1])  # pick the lower center subplot
    legend_artists = list()
    for cls_name, color in zip(class_names, colors):
        artist = Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=color,
            markersize=6,
            label=cls_name,
        )
        legend_artists.append(artist)
    # Create a legend outside of the plot
    legend_ax.legend(
        handles=legend_artists,
        loc="lower center",
        ncol=len(class_names),
        fontsize=2.5,
        bbox_to_anchor=(0.5, -0.2), # -1, -0.33
        frameon=False,
    )
    buffer = BytesIO()
    plt.savefig(
        buffer,
        bbox_inches="tight",
        dpi=1200,
        format="png",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    
    buffer.seek(0)

    return buffer
    # buffer.close()
    # prevent using excessive memory

def overlay(
    image: np.ndarray,
    seg_mask: np.ndarray,
    num_classes: int,
    colors: list,
    normalize: bool = False,
    transparency: float = 0.5,
):
    
    colors = [mcolors.hex2color(c) for c in colors]
    colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]
    # image
    img = image
    if normalize:
        eps = 0.000001
        img = (img - img.min()) / (img.max() - img.min() + eps) * 255.0
    img = img.astype(np.uint8)
    img = torch.tensor(img)
    img = torch.stack((img, img, img), dim=0)
    # mask
    msk = np.rint(seg_mask)  # Round an array to the given number of decimals
    msk = torch.tensor(msk).long()
    msk = torch.nn.functional.one_hot(msk, num_classes)
    msk = torch.moveaxis(msk, 2, 0).bool()
    msk = msk[1:, :, :]
    # blending
    out = draw_segmentation_masks(img, msk, alpha=transparency, colors=colors)
    out = torch.moveaxis(out, 0, 2).numpy()
    return out


def mask_overlay(
    image, mask, 
    color_map: dict= {
        1.0: [0, 255, 0],  # Green
        2.0: [255, 0, 0],  # Red
        4.0: [255, 255, 0],  # Yellow
    }, 
    alpha=0.5):
    """
    Overlay a mask with multiple colors on top of the grayscale image.
    
    Args:
        image (np.ndarray): Grayscale image (H x W or H x W x 3).
        mask (np.ndarray): Mask with pixel values representing different classes (H x W).
        color_map (dict): Dictionary mapping mask values to RGB colors, e.g., {1: (0, 255, 0), 2: (255, 255, 0)}.
    
    Returns:
        np.ndarray: Image with the mask overlaid.
    """
    # Ensure image is 3-channel (convert grayscale to RGB if needed)
    if image.dtype != np.uint8:  # Đảm bảo ảnh là uint8
        image = (image * 255).astype(np.uint8)
    
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Create a copy of the original image
    overlay_image = image.copy()
    
    for mask_value, color in color_map.items():
        # Create a binary mask for the current mask value
        binary_mask = (mask == mask_value).astype(np.uint8)
        
        # Create a color mask for the current mask value
        color_mask = np.dstack([
            binary_mask * color[0],  # Red channel
            binary_mask * color[1],  # Green channel
            binary_mask * color[2],  # Blue channel
        ]).astype(np.uint8)
        
        # Blend the color mask with the original image
        weighted_sum = cv2.addWeighted(color_mask, alpha, overlay_image, 1-alpha, 0.0)
        
        # Apply the mask only on the relevant regions
        overlay_image[binary_mask > 0] = weighted_sum[binary_mask > 0]
    
    return overlay_image

def scale_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val - min_val == 0:
        return np.zeros_like(image, dtype=np.uint8)
    
    scaled_image = (image - min_val) / (max_val - min_val) * 255
    
    return scaled_image.astype(np.uint8)

def resample_3d(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
    return img_resampled
