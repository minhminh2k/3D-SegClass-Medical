import torch

from monai.losses import AsymmetricUnifiedFocalLoss
from monai.data.meta_tensor import MetaTensor
import torch

# Tạo tensor ones và wrap thành MetaTensor
meta_tensor = MetaTensor(torch.ones((1, 1, 32, 32, 32), dtype=torch.float32))

pred = torch.ones((1,1,32,32, 32), dtype=torch.float32).to("cuda")
pred = pred * 5
grnd = MetaTensor(torch.zeros((1,1,32,32, 32), dtype=torch.int64).to("cuda"))

print("Max pred:", torch.max(pred))

print("Max ground:", torch.max(grnd))
fl = AsymmetricUnifiedFocalLoss(to_onehot_y=False, num_classes=2).to("cuda")

if torch.max(grnd) != 1:
    print("1111")
print(fl(pred, grnd))