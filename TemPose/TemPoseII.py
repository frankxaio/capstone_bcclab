import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from TemPose.NN_models import PreNorm, FeedForward, Attention, Transformer
from TemPose.utility_tempose import get_2d_sincos_pos_embed

class TemPoseII_TF(nn.Module):
    def __init__(self, *, poses_numbers, time_steps, num_people, num_classes, dim=50, kernel_size=5, depth=4, depth_int=3,
                 heads=6, scale_dim=4, mlp_dim=512, pool='cls', dim_head=75, dropout=0.3, emb_dropout=0.3):
        super().__init__()

        self.heads = heads
        self.time_sequence = time_steps
        self.people = num_people + 3  # 2 players + 1 shuttle
        self.to_patch_embedding = nn.Linear(poses_numbers, dim)

        self.temporal_embedding = nn.Parameter(torch.zeros(1, self.people, time_steps + 1, dim), requires_grad=True)
        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

        self.interaction_token = nn.Parameter(torch.randn(1, 1, dim))
        self.interaction_embedding = nn.Parameter(torch.zeros(1, self.people + 1, dim))
        self.interaction_transformer = Transformer(dim, depth_int, heads, dim_head, dim * scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, scale_dim * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(scale_dim * dim, num_classes)
        )

        self.num_channels = [dim // 2, dim]
        self.kernel_size = kernel_size
        input_size = 2

        layers = []
        num_levels = len(self.num_channels)
        for i in range(num_levels):
            dilation_size = (2 * i) + 1
            in_channels = input_size if i == 0 else self.num_channels[i - 1]
            out_channels = self.num_channels[i]
            padding = (kernel_size - 1) * dilation_size // 2
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation_size, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout)
            ]
        self.tcn1 = nn.Sequential(*layers)
        self.tcn2 = nn.Sequential(*layers)
        self.tcn3 = nn.Sequential(*layers)

        self.initialize_weights()

    def initialize_weights(self):
        temp_embed = repeat(
            torch.from_numpy(get_2d_sincos_pos_embed(self.temporal_embedding.shape[-1], int(self.time_sequence), cls_token=True)).float().unsqueeze(0),
            '() t d -> n t d', n=self.people
        )
        self.temporal_embedding.data.copy_(temp_embed.unsqueeze(0))

        int_embed = get_2d_sincos_pos_embed(self.interaction_embedding.shape[-1], self.people, cls_token=True)
        self.interaction_embedding.data.copy_(torch.from_numpy(int_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.interaction_token, std=.02)
        torch.nn.init.normal_(self.temporal_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, sp, t_pad=None):
        x = rearrange(x, 'b t n d -> b n t d')
        x = self.to_patch_embedding(x)
        b, n, t, _ = x.shape

        input_seq = rearrange(sp, 'b t d -> b d t')
        n = self.people

        x_pos1 = rearrange(self.tcn1(input_seq[:, :2, :]), 'b d t -> b t d').unsqueeze(1)
        x_pos2 = rearrange(self.tcn2(input_seq[:, 2:4, :]), 'b d t -> b t d').unsqueeze(1)
        x_shuttle = rearrange(self.tcn3(input_seq[:, -2:, :]), 'b d t -> b t d').unsqueeze(1)

        x = torch.cat((x, x_pos1, x_pos2, x_shuttle), dim=1)

        # ============================== TODO [Ex 3.5a] ==============================
        # x: (b, n, t, dim) - n = 5 streams (2 poses, 2 positions, shuttle)
        # TEMPORAL stage (every stream attends over time on its own):
        #   1. prepend the learned temporal CLS token to every stream -> (b, n, t+1, dim),
        #      add self.temporal_embedding[:, :, :t+1] and apply self.dropout
        #   2. flatten the streams into the batch -> ((b n), t+1, dim)
        #   3. padding mask mask_t of shape ((b n), t+1): 1 = attend, 0 = padding.
        #      Without t_pad everything is 1. t_pad (shape (b,)) holds the number of
        #      valid frames L of every sample: repeat each value n times (one per
        #      stream, same order as step 2); for a stream with L valid frames keep
        #      positions 0 .. L (CLS + L frames) and set positions L+1 .. t to 0.
        #   4. run self.temporal_transformer(x, mask_t[:, None, None, :]) and keep only
        #      the CLS output -> (b, n, dim)
        # INTERACTION stage (the streams attend to each other, no mask):
        #   5. prepend self.interaction_token -> (b, n+1, dim), add
        #      self.interaction_embedding[:, :n+1], run self.interaction_transformer
        #   6. pool: mean over the n+1 tokens if self.pool == "mean", else the CLS
        #      token -> (b, dim). Return self.mlp_head(...) -> logits (b, num_classes).
        # ============================================================================
        raise NotImplementedError("TODO [Ex 3.5a]")

    def predict(self, x, sp, pad=None):
        device = next(self.parameters()).device
        x = x.to(device)
        sp = sp.to(device)
        if pad is not None:
            pad = pad.to(device)
        # ============================== TODO [Ex 3.5b] ==============================
        # Return the predicted class index for every sample, as a CPU tensor (b,).
        # ============================================================================
        raise NotImplementedError("TODO [Ex 3.5b]")


# helpers
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemPoseII_TF(
        poses_numbers=94,
        time_steps=50,
        num_people=2,
        num_classes=13,
        dim=4,
        depth=3,
        depth_int=200,
        dim_head=128,
        emb_dropout=0.2
    )
    model.to(device)
    print(model)
