import cv2
import torch
import numpy as np
import tritonclient.http as httpclient

# Read image
def load_image(img_path: str):
    return np.expand_dims(np.fromfile(img_path, dtype='uint8'), axis=0)

def postprocess_mask(mask):
    mask = torch.sigmoid(torch.from_numpy(mask.copy()))
    mask = (mask >= 0.5).cpu().numpy().astype(np.uint8)
    return mask

# Setting up GRPC Client
client = httpclient.InferenceServerClient(url="0.0.0.0:8000")

# Inputs
image_input = load_image('ship/00a3ab3cc.jpg')
inputs = httpclient.InferInput("IMAGE", image_input.shape, datatype="UINT8")
inputs.set_data_from_numpy(image_input)

# Outputs
outputs = []
outputs.append(httpclient.InferRequestedOutput("CLASSIFICATION"))
outputs.append(httpclient.InferRequestedOutput("SEGMENTATION"))

# Querying the server
results = client.infer(model_name="ensemble_model", inputs=[inputs], outputs=outputs)

# RESNET MODEL
inference_output_resnet = results.as_numpy('CLASSIFICATION')
print("Classification:", torch.sigmoid(torch.from_numpy(inference_output_resnet.copy())))    

# UNET MODEL
inference_output_unet = results.as_numpy('SEGMENTATION').squeeze(0)
inference_output_unet = postprocess_mask(inference_output_unet)
inference_output_unet = (inference_output_unet * 255).transpose(1, 2, 0)
cv2.imwrite("output.png", inference_output_unet)
