import os
import numpy as np
import nibabel as nib


def convert_npy_to_nifti(input_dir, output_dir, spacing=(1.0, 1.0, 1.0), is_label=False):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".npy"):
            npy_path = os.path.join(input_dir, filename)
            data = np.load(npy_path)

            # Chuyển kiểu dữ liệu phù hợp
            if is_label:
                data = data.astype(np.uint8)
            else:
                data = data.astype(np.float32)
            
            affine = np.array([
                [0. , 0. , 1., 0.],
                [0. , -1. , 0., 0.],
                [1. , 0. , 0., 0.],
                [0. , 0. , 0., 1. ]]
            )
            
            nifti_img = nib.Nifti1Image(data, affine)

            nifti_img.set_data_dtype(data.dtype)

            out_path = os.path.join(output_dir, filename.replace(".npy", ".nii.gz"))
            nib.save(nifti_img, out_path)
            print(f"✅ Saved: {out_path}")
            
# Chuyển ảnh
convert_npy_to_nifti("./Resample", "./LIDC", spacing=(1.0, 1.0, 1.0), is_label=False)

