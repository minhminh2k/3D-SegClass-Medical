import os
import torch
import glob
import yaml
import matplotlib
import numpy as np
import nibabel as nb
from tqdm import tqdm
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torchvision.utils import draw_segmentation_masks


def viz_acdc(config: dict):
    # generate a set of frame indicies to sample from
    case_names = config["acdc"]["case_names"]
    frame_nums = config["acdc"][
        "frame_nums"
    ]  # each case has different number of total frames
    iterables = zip(case_names, frame_nums)
    for case_name, frame_num in tqdm(
        iterables, total=len(case_names), desc="rendering ACDC plots"
    ):
        frame_nums = np.random.choice(
            range(int(frame_num)),
            size=4,
            replace=False,
            p=None,
        )
        # frame_nums = np.sort(frame_nums)
        render_and_save_gridspec(
            case_name=case_name,
            frame_nums=frame_nums,
            data_folder=config["acdc"]["data_folder"],
            pred_folders=config["acdc"]["pred_folders"],
            num_classes=config["acdc"]["num_classes"],
            class_names=config["acdc"]["class_names"],
            colors=config["acdc"]["colors"],
            fig_save_dir=config["acdc"]["fig_save_dir"],
            img_normalize=True,
            transparency=0.65,
        )


def viz_synapse(config: dict):
    # generate a set of frame indicies to sample from
    case_names = config["synapse"]["case_names"]
    frame_nums = config["synapse"]["frame_nums"]
    for case_name, frame_num in tqdm(
        zip(case_names, frame_nums),
        total=len(case_names),
        desc="rendering SYNAPSE plots",
    ):
        frame_num = int(frame_num)

        frame_nums = np.random.choice(
            range(frame_num // 2, frame_num, 5),
            size=4,
            replace=False,
            p=None,
        )
        # frame_nums = np.sort(frame_nums)
        render_and_save_gridspec(
            case_name=case_name,
            frame_nums=frame_nums,
            data_folder=config["synapse"]["data_folder"],
            pred_folders=config["synapse"]["pred_folders"],
            num_classes=config["synapse"]["num_classes"],
            class_names=config["synapse"]["class_names"],
            colors=config["synapse"]["colors"],
            fig_save_dir=config["synapse"]["fig_save_dir"],
            img_normalize=True,
            transparency=0.65,
        )


def viz_brats(config: dict):
    # generate a set of frame indicies to sample from
    case_names = config["brats"]["case_names"]
    frame_n = int(config["brats"]["frame_nums"])
    for case_name in tqdm(case_names, desc="rendering BRATS plots"):
        frame_nums = np.random.choice(
            # brain pops up from %45 to %75 of the volume
            range(30, 90, 8),
            size=4,
            replace=False,
            p=None,
        )
        render_and_save_gridspec(
            case_name=case_name,
            frame_nums=frame_nums,
            data_folder=config["brats"]["data_folder"],
            pred_folders=config["brats"]["pred_folders"],
            num_classes=config["brats"]["num_classes"],
            class_names=config["brats"]["class_names"],
            colors=config["brats"]["colors"],
            fig_save_dir=config["brats"]["fig_save_dir"],
            img_normalize=True,
            transparency=0.65,
        )


def overlay(
    image: np.ndarray,
    seg_mask: np.ndarray,
    num_classes: int,
    colors: list,
    normalize: bool = False,
    transparency: float = 0.5,
):
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


def render_and_save_gridspec(
    data_folder: str,
    pred_folders: str,
    frame_nums: list,
    case_name: str,
    num_classes: int,
    class_names: list,
    colors: list,
    fig_save_dir: str,
    img_normalize: bool,
    transparency: float = 1.0,
):
    assert os.path.exists(data_folder), f"{data_folder} does not exist"
    assert os.path.exists(fig_save_dir), f"{fig_save_dir} does not exist"
    # directory to save the figure
    save_fig_fp = os.path.join(fig_save_dir, case_name.split(".nii.gz")[0] + ".png")
    # reading the orignal image
    data_fp = os.path.join(data_folder, case_name)
    data_volume = nb.load(data_fp).get_fdata()
    # plot setting
    rows = 4
    cols = 4
    fig = plt.figure(figsize=(cols, rows))
    gs = gridspec.GridSpec(rows, cols, wspace=0.01, hspace=0.01)
    # rendering a 4x4 gird plot.
    # column1: ground truth
    # column2: segformer3d
    # column3: nnFormer
    # column5: Unetr
    for row, frame_ind in enumerate(frame_nums):
        for col, pred_folder in enumerate(pred_folders):
            fp = os.path.join(pred_folder, case_name)
            seg_mask_volume = nb.load(fp).get_fdata()
            axis = plt.subplot(gs[row, col])
            mask = seg_mask_volume[:, :, frame_ind]
            image = data_volume[:, :, frame_ind]
            overlayed_img = overlay(
                image=image,
                seg_mask=mask,
                num_classes=num_classes,
                colors=colors,
                normalize=img_normalize,
                transparency=transparency,
            )
            axis.imshow(overlayed_img, cmap="bone")
            axis.axis("off")
            if row == 0 and col == 0:
                axis.set_title("Ground Truth", fontsize=5, pad=2)
            elif row == 0 and col == 1:
                axis.set_title("SegFormer3D", fontsize=5, pad=2)
            elif row == 0 and col == 2:
                axis.set_title("nnFormer", fontsize=5, pad=2)
            elif row == 0 and col == 3:
                axis.set_title("Unetr", fontsize=5, pad=2)
    # generate the legend
    legend_ax = plt.subplot(gs[3, 3])  # pick the lower center subplot
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
        fontsize=4.5,
        bbox_to_anchor=(-1, -0.33),
        frameon=False,
    )
    plt.savefig(
        save_fig_fp,
        bbox_inches="tight",
        dpi=1200,
        format="png",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    # prevent using excessive memory
    plt.close()


if __name__ == "__main__":
    # set random seed to change the plotting frame!
    np.random.seed(42)
    with open("viz_meta.yaml") as f:
        config = yaml.safe_load(f)
    ########################### ACDC ###########################
    viz_acdc(config=config)
    ########################### SYNAPSE ###########################
    viz_synapse(config=config)
    ########################### BRATS ###########################
    viz_brats(config=config)
