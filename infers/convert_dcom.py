import pyplastimatch as pypla
from pyplastimatch.utils.install import install_precompiled_binaries

install_precompiled_binaries()


# convert one of the NIFTI images to DICOM: name: <patient1>, output folder: <dicom_output>

def convert_dcom(case: str):
    convert_args_ct = {
        "input": f"data/LIDC/{case}.nii.gz",
        "patient-id": f"patient{case[-2:]}",
        "output-dicom": f"data/LIDC_Dcom/{case}",
    }
    pypla.convert(verbose=True, **convert_args_ct)

# Command: sudo /home/lenovo/anaconda3/envs/mis/bin/python infers/convert_dcom.py

if __name__ == "__main__":
    case = "LIDC-IDRI-0003"
    convert_dcom(case)
    