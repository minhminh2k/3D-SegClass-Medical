import os
import kaggle

kaggle.api.dataset_download_files('minhdngquang/lung-seg', path='data', unzip=False, quiet=False)