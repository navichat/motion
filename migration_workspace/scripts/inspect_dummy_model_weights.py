import torch

model_path = 'migration_workspace/models/dummy_deephase_model.pth'
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

# Assuming the checkpoint is directly the state_dict
state_dict = checkpoint

print("State dict keys and shapes:")
for k, v in state_dict.items():
    print(f"Key: {k}, Shape: {v.shape}")
