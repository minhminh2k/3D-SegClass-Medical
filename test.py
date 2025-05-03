import torch
from monai.inferers import sliding_window_inference


ckpt_path = "test.pt"

test = torch.jit.load(ckpt_path).to("cuda")

test.eval()

with torch.no_grad():
    a = torch.rand(1, 1, 256, 160, 160).to("cuda")
    
    outputs = sliding_window_inference(
        inputs=a,
        roi_size=[128, 128, 128],
        sw_batch_size=1,
        predictor=test,
        overlap=0.25,
    )

    print(outputs.shape)
