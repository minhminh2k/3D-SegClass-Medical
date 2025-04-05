import torch
from .build_sam3D import sam_model_registry3D

if __name__ == "__main__":
    sam_checkpoint = "/home/duong.quang.minh/project/3D-SegClass-Medical/checkpoints/sammed3d/sam_med3d_turbo.pth"
    model_type = "vit_b_ori"
    device = "cuda"
    sam = sam_model_registry3D[model_type](checkpoint=None)
    # sam.to(device=device)
    # sam.train()

    # model_state = torch.load(sam_checkpoint, weights_only=False)

    # sam.load_state_dict(model_state["model_state_dict"])

    # torch.save(sam.image_encoder.state_dict(),'/home/duong.quang.minh/project/3D-SegClass-Medical/checkpoints/sammed3d/sam_med3d_turbo_image_encoder.pth')

    input_test = torch.rand((1, 1, 64, 192, 192), dtype=torch.float)
    x, hidden_states_out = sam.image_encoder(input_test)
    print(x.shape)

    for i in hidden_states_out:
        print(i.shape)