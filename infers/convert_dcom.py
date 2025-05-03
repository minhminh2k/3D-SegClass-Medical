import pyplastimatch as pypla
from pyplastimatch.utils.install import install_precompiled_binaries

install_precompiled_binaries()

# convert one of the NIFTI images to DICOM: name: <patient1>, output folder: <dicom_output>
convert_args_ct = {
    "input": "data/LIDC/LIDC-IDRI-0001.nii.gz",
    "patient-id": "patient1",
    "output-dicom": "data/LIDC_dcom",
}
pypla.convert(verbose=True, **convert_args_ct)
