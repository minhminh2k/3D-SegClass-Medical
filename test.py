import torch
import time

from monai.inferers import sliding_window_inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.jit.load("checkpoints/unetr.pt").to(device)

model.eval()

start_x = time.time()
with torch.no_grad():
    a = torch.rand(1, 1, 420, 420, 420).to(device)
    
    output = sliding_window_inference(
        inputs=a,
        roi_size=[128, 128, 128],
        sw_batch_size=1,
        predictor=model,
        overlap=0.5
    )
    
    print(output.shape)
    
end_x = time.time()

print("Time", end_x - start_x)