
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

# Define the network architecture (copied from network.py)
class MotionGenerator_RNN(nn.Module):
    def __init__(self,
                 aud_dim, aud_hid_dim, aud_embed_dim,
                 mo_dim,
                 lxm_dim,
                 rnn_hid_dim, rnn_out_dim, rnn_depth):
        super().__init__()

        self.audio_encoder = AudioEncoder_Conv1d(aud_dim, aud_hid_dim, aud_embed_dim)

        self.motion_decoder = GRUDecoder(mo_dim, aud_embed_dim, lxm_dim, rnn_hid_dim, rnn_out_dim, rnn_depth)
        self.cell_state_init = CellStateInitializer(mo_dim+lxm_dim, rnn_hid_dim, rnn_depth)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.)

    def forward(self, aud, mo, lxm):
        """
        aud, mo: [N, D, L]
        lxm: [N, D, B]. B: num_block
        """
        N, D, L = aud.shape
        B = lxm.shape[-1]
        BL = L // B  # BL: block length

        # initialize the hidden state of rnn
        cell_state = self.cell_state_init(mo[:, :, BL-1], lxm[:, :, 0])

        mo_hat = []  # without the first and the last blocks. element shape: [N, D, 1]
        for b_idx in range(B-2):
            aud_embed_b = self.audio_encoder(aud[:, :, (b_idx+0)*BL: (b_idx+3)*BL])

            for f in range(BL):  # f: frame
                # get previous pose
                pre_mo = mo[:, :, BL-1] if b_idx == 0 and f == 0 else mo_hat[-1][:, :, 0]

                # get current audio embedding and lexeme
                cur_aud_embed = aud_embed_b[:, :, f]
                cur_lxm = lxm[:, :, b_idx+1]

                mo_pred, cell_state = self.motion_decoder(pre_mo, cur_aud_embed, cur_lxm, cell_state)

                mo_hat.append(mo_pred.unsqueeze(-1))

        mo_hat = torch.cat(mo_hat, dim=-1)

        return mo_hat


class AudioEncoder_Conv1d(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim) -> None:
        super().__init__()

        self.layer0 = nn.Conv1d(in_dim, hidden_dim, 7, 1, 0)
        self.layer1 = nn.Conv1d(hidden_dim, hidden_dim, 5, 1, 0)
        self.layer2 = nn.Conv1d(hidden_dim, out_dim, 4, 2, 1)
        self.layer3 = nn.Linear(out_dim, out_dim)

        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.layer0(x))
        x = F.gelu(self.layer1(x))
        x = F.gelu(self.layer2(x))

        x = torch.einsum('NDL->NLD', x)

        x = F.gelu(self.layer3(x))
        x = self.norm(x)

        x = torch.einsum('NLD->NDL', x)

        return x


class GRUDecoder(nn.Module):
    def __init__(self, mo_dim, aud_embed_dim, lxm_dim, hidden_dim, out_dim, depth):
        super().__init__()

        in_dim = mo_dim + aud_embed_dim + lxm_dim

        self.embed_in = nn.Linear(in_dim, hidden_dim)
        self.gru = nn.GRU(in_dim+hidden_dim, hidden_dim, depth, batch_first=True)
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.embed_pred = nn.Linear(hidden_dim, out_dim)

    def forward(self, mo, aud_embed, lxm, cell_state):
        in_embed = F.gelu(self.embed_in(torch.cat([mo, aud_embed, lxm], dim=-1)))
        cell_out, cell_state = self.gru(
            torch.cat([in_embed, mo, aud_embed, lxm], dim=-1).unsqueeze(1), cell_state
        )
        out = self.decoder_norm(cell_out)
        out = self.embed_pred(out.squeeze(1))

        return out, cell_state


class CellStateInitializer(nn.Module):
    def __init__(self, in_dim, hidden_dim, rnn_depth):
        super().__init__()

        self.rnn_depth = rnn_depth

        self.layer0 = nn.Linear(in_dim, hidden_dim)
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim*rnn_depth)

    def forward(self, mo, lxm):
        x = F.gelu(self.layer0(torch.cat([mo, lxm], dim=-1)))
        x = F.gelu(self.layer1(x))
        x = self.layer2(x)

        x = x.reshape(x.shape[0], self.rnn_depth, -1)
        x = torch.einsum('NPD->PND', x).contiguous()

        return x

# Model hyperparameters from config.json5
aud_dim = 80
aud_hid_dim = 64
aud_embed_dim = 64
mo_dim = 48
lxm_dim = 96
rnn_hid_dim = 1024
rnn_out_dim = 48
rnn_depth = 4

# Instantiate the model
model = MotionGenerator_RNN(
    aud_dim, aud_hid_dim, aud_embed_dim,
    mo_dim,
    lxm_dim,
    rnn_hid_dim, rnn_out_dim, rnn_depth
)

# Create dummy input tensors
N = 1  # Batch size
L = 100  # Sequence length
B = 10  # Number of blocks (L must be divisible by B)

aud_input = torch.randn(N, aud_dim, L)
mo_input = torch.randn(N, mo_dim, L)
lxm_input = torch.randn(N, lxm_dim, B)

# Perform a forward pass
model.eval()
with torch.no_grad():
    output = model(aud_input, mo_input, lxm_input)

# Export to ONNX
onnx_path = "motion_generator.onnx"
torch.onnx.export(
    model,
    (aud_input, mo_input, lxm_input),
    onnx_path,
    input_names=['aud_input', 'mo_input', 'lxm_input'],
    output_names=['output'],
    dynamic_axes={
        'aud_input': {0: 'batch_size', 2: 'sequence_length'},
        'mo_input': {0: 'batch_size', 2: 'sequence_length'},
        'lxm_input': {0: 'batch_size', 2: 'num_blocks'},
        'output': {0: 'batch_size', 2: 'output_sequence_length'}
    },
    opset_version=12
)
print(f"Model exported to {onnx_path}")

# Save input and output data for verification
data = {
    "aud_input": aud_input.numpy().tolist(),
    "mo_input": mo_input.numpy().tolist(),
    "lxm_input": lxm_input.numpy().tolist(),
    "output": output.numpy().tolist()
}

with open("sample_data.json", "w") as f:
    json.dump(data, f)
print("Sample input/output data saved to sample_data.json")
