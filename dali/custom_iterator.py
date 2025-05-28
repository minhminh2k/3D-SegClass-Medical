import os
import glob
import time
from tqdm import tqdm
import pandas as pd

import nvidia.dali.ops as ops


from random import shuffle
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali import pipeline_def
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.pytorch import LastBatchPolicy

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

class DALICustomIterator(DALIGenericIterator):
    def __init__(self, pipelines, output_map, size, auto_reset=False, fill_last_batch=True, dynamic_shape=False, last_batch_padded=False):
        super(DALICustomIterator, self).__init__(pipelines, output_map, size, auto_reset, fill_last_batch, dynamic_shape, last_batch_padded)

    def __len__(self):
        return int(self._size / self.batch_size) + 1

    def __next__(self):
        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return batch
        feed = super().__next__()
        data = feed[0]['data']
        return data
    
    
class Custom_DALI_Iterator(DALIGenericIterator):
    def __init__(self, pipelines, batch_size, files, last_batch_policy,  last_batch_padded, auto_reset=True):
        super().__init__(pipelines=pipelines, last_batch_policy=last_batch_policy,  last_batch_padded =
        last_batch_padded, auto_reset=auto_reset, output_map=['images', 'labels'])
        self.files = files
        self.batch_size = batch_size
        self.data_set_len = len(self.files)
        self.n = self.data_set_len

    def __iter__(self):
        self.i = 0

        shuffle(self.files)
        return self

    def __len__(self):
        return self.data_set_len

    def __next__(self):

        if self.i >= self.n:
            self.__iter__()
            raise StopIteration
        else:
            out = super().__next__()
            images = out[0]['images']
            labels = out[0]['labels']

            q = (self.n - self.i) // self.batch_size
            mod = (self.n - self.i) % self.batch_size
            if q>0:
                self.i = self.i + self.batch_size
            else:
                self.i = self.i + mod

            return (images, labels)

    next = __next__

def DALIDataLoader():
    pipes = dali_pipeline()
    pipes.build()
    
    dali_iter = DALICustomIterator(pipes, ['data'], pipes.epoch_size("Reader"), auto_reset=True)
    return dali_iter
data_loader = DALIDataLoader()


start_time = time.time()
for image in tqdm(data_loader):
    # Already on GPU
    pass

dali_time = time.time() - start_time


# GPU Direct Storage: for .npy file 
# Note: the device="gpu" is the GPU Direct Storage. 
# This is only compatible when you use with readers.numpy. This enables a direct data path between storage and GPU memory.
@pipeline_def
def custom_pipeline(files, root_dir1, root_dir2):

    numpy_data1 = fn.readers.numpy(device='gpu', file_root=root_dir1, files =files, name="my_reader")
    numpy_data1 = fn.reshape(numpy_data1, rel_shape=[1, 1, -1], layout="HWC")
    numpy_data1 = fn.flip(numpy_data1, vertical=1, horizontal=1)

    numpy_data2 = fn.readers.numpy(device='gpu', file_root=root_dir2, files =files)
    numpy_data2 = fn.reshape(numpy_data2, rel_shape=[1, 1, -1], layout="HWC")
    numpy_data2 = fn.flip(numpy_data2, vertical=1, horizontal=1)

    return (numpy_data1, numpy_data1)