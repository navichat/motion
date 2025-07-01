# migration_workspace/scripts/create_dummy_deephase_model.py
import torch
import torch.nn as nn
import os

# Dummy Skeleton class (same as in export_deephase.py)
class DummySkeleton:
    def __init__(self, num_joints):
        self.num_joints = num_joints

# Minimal PAE_AI4Animation class (same as in export_deephase.py)
class PAE_AI4Animation(nn.Module):
    def __init__(self, n_phases, n_joints, length, key_range=1., window=2.0):
        super(PAE_AI4Animation, self).__init__()
        
        self.embedding_channels = n_phases
        input_channels = (n_joints) * 3
        time_range = length
        intermediate_channels = int(input_channels / 3)

        self.conv1 = nn.Conv1d(input_channels, intermediate_channels, time_range, stride=1,
                               padding=int((time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.bn_conv1 = nn.BatchNorm1d(num_features=intermediate_channels)
        self.conv2 = nn.Conv1d(intermediate_channels, self.embedding_channels, time_range, stride=1,
                               padding=int((time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.bn_conv2 = nn.BatchNorm1d(num_features=self.embedding_channels)

        self.fc = nn.ModuleList()
        self.bn = nn.ModuleList()
        for i in range(self.embedding_channels):
            self.fc.append(nn.Linear(time_range, 2))
            self.bn.append(nn.BatchNorm1d(num_features=2))
        self.parallel_fc0 = nn.Linear(time_range, self.embedding_channels)
        self.parallel_fc1 = nn.Linear(time_range, self.embedding_channels)

        self.deconv1 = nn.Conv1d(self.embedding_channels, intermediate_channels, time_range, stride=1,
                                 padding=int((time_range - 1) / 2), dilation=1, groups=1, bias=True,
                                 padding_mode='zeros')
        self.bn_deconv1 = nn.BatchNorm1d(num_features=intermediate_channels)
        self.deconv2 = nn.Conv1d(intermediate_channels, input_channels, time_range, stride=1,
                                 padding=int((time_range - 1) / 2), dilation=1, groups=1, bias=True,
                                 padding_mode='zeros')

    def forward(self, x):
        # Minimal forward pass to match output shape and count
        y = self.conv1(x)
        y = self.bn_conv1(y)
        y = self.conv2(y)
        y = self.bn_conv2(y)
        latent = y

        y_reconstructed = self.deconv1(latent)
        y_reconstructed = self.bn_deconv1(y_reconstructed)
        y_reconstructed = self.deconv2(y_reconstructed)

        batch_size = x.shape[0]
        p = torch.randn(batch_size, self.embedding_channels, 1)
        f = torch.randn(batch_size, self.embedding_channels, 1)
        a = torch.randn(batch_size, self.embedding_channels, 1)
        b = torch.randn(batch_size, self.embedding_channels, 1)

        return y_reconstructed, p, a, f, b

# Minimal DeepPhaseNet class (same as in export_deephase.py)
class DeepPhaseNet(nn.Module):
    def __init__(self, n_phase, skeleton, length, dt, batch_size):
        super(DeepPhaseNet, self).__init__()
        self.model = PAE_AI4Animation(n_phase, skeleton.num_joints, length)

    def forward(self, input):
        return self.model(input)

if __name__ == "__main__":
    # Parameters for DeepPhaseNet
    n_phases = 10
    skeleton = DummySkeleton(num_joints=22)
    length = 61
    dt = 1.0 / 30.0
    batch_size = 32

    # Instantiate the model
    model = DeepPhaseNet(n_phases, skeleton, length, dt, batch_size)

    # Define output path for the dummy model
    dummy_model_path = 'migration_workspace/models/dummy_deephase_model.pth'
    os.makedirs(os.path.dirname(dummy_model_path), exist_ok=True)

    # Save the state_dict
    torch.save(model.state_dict(), dummy_model_path)
    print(f"Dummy DeepPhase model saved to {dummy_model_path}")
