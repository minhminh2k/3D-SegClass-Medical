import wandb

# Initialize the Weights & Biases API
api = wandb.Api()

# Get the run
run_id = "88znpzxh"
run = api.run(f"minhqd9112003/3d-segmentation/{run_id}")

# Location of file you want to upload
ckpt_path = "/home/duong.quang.minh/project/3D-SegClass-Medical/checkpoints/unetr.ckpt"

# Upload
run.upload_file(ckpt_path)
print(f"Successfully upload {ckpt_path}")