import torch
import numpy as np
import os

model_path = 'migration_workspace/models/dummy_deephase_model.pth'
output_dir = 'migration_workspace/weights/deephase/'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
state_dict = checkpoint # Assuming it's directly the state_dict

print(f"Saving weights to {output_dir}")
for k, v in state_dict.items():
    file_name = k.replace('.', '_') + '.npy' # Replace dots with underscores for valid filenames
    output_path = os.path.join(output_dir, file_name)
    np.save(output_path, v.numpy())
    print(f"Saved {k} to {output_path}")

print("All weights saved as .npy files.")
