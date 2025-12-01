
import torch
import torch.nn as nn

class TinyGPTPhysics(nn.Module):
    def __init__(self, d_model=64, n_heads=4, n_layers=2, seq_len=16):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        self.input_proj = nn.Linear(3, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4*d_model,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.output_head = nn.Linear(d_model, 2)

    def forward(self, x, return_hidden=False):
        """
        x: (B, T, 3)
        """
        B, T, _ = x.shape
        h = self.input_proj(x)

        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = h + self.pos_emb(positions)

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        h_trans = self.transformer(h, mask=mask)

        y = self.output_head(h_trans)

        if return_hidden:
            return y, h_trans
        return y
