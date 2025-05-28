import torch
import bentoml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.jit.load("model.pt")

saved_model = bentoml.torchscript.save_model(
    "unet34-torchscript",
    model,
    signatures={
        "__call__": {
            "batchable": True,
            "batch_dim": 0,
        }
    },
)

print(f"Model saved: {saved_model}")

