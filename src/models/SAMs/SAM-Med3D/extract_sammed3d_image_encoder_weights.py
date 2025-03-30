import torch
from .build_sam3D import sam_model_registry3D

if __name__ == "__main__":
    sam_checkpoint = "/data/hpc/dqm/3D-SegClass-Medical/checkpoints/sammed3d/sam_med3d_turbo.pth"
    model_type = "vit_b_ori"
    device = "cuda"
    sam = sam_model_registry3D[model_type](checkpoint=None)
    # sam.to(device=device)
    # sam.train()

    model_state = torch.load(sam_checkpoint)

    sam.load_state_dict(model_state["model_state_dict"])

    torch.save(sam.image_encoder.state_dict(),'/data/hpc/dqm/3D-SegClass-Medical/checkpoints/sammed3d/sam_med3d_turbo_image_encoder.pth')
    print(sam.image_encoder)