import nvidia.dali as dali
from nvidia.dali.plugin.triton import autoserialize
import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def, types
from nvidia.dali.pipeline import Pipeline
 
@dali.pipeline_def(batch_size=2, num_threads=2, device_id=0)
def pipe():
    images = dali.fn.external_source(device="cpu", name="DALI_INPUT_0")
    images = dali.fn.decoders.image(images, device="mixed")
    images = dali.fn.resize(images, resize_x=224, resize_y=224)
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
    return images

pipe = pipe()
pipe.serialize(filename="triton/model_repository/dali/1/model.dali")