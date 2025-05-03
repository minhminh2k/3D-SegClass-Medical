
import tritonclient.http as httpclient
from PIL import Image
from torchvision import transforms
from tritonclient.utils import triton_to_np_dtype
import numpy as np
import torch

transformed_img = torch.rand(1, 1, 128, 128, 128).numpy().astype(np.float32)

# Setting up client
client = httpclient.InferenceServerClient(url="localhost:8000")

inputs = httpclient.InferInput("input_volume", transformed_img.shape, datatype="FP32")
inputs.set_data_from_numpy(transformed_img)

outputs = httpclient.InferRequestedOutput(
    "SEGMENTATION_OUTPUT"
)

# Querying the server
results = client.infer(model_name="unetr", inputs=[inputs], outputs=[outputs])
inference_output = results.as_numpy("SEGMENTATION_OUTPUT")

print(inference_output.shape)