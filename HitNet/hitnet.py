import torch
import torch.nn as nn
import torch.nn.functional as F

class HitNet(nn.Module):
    def __init__(self, input_dim, num_consec):
        super().__init__()
        self.num_consec = num_consec

        # Temporal CNN Block
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.batchnorm = nn.BatchNorm1d(128)

        # Bi-GRU
        self.bigru = nn.GRU(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            dropout=0.5,
            bidirectional=True,
            batch_first=True
        )

        # Self-Attention
        self.query_layer = nn.Linear(512, 128)
        self.key_layer = nn.Linear(512, 128)
        self.value_layer = nn.Linear(512, 128)
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        # Pooling
        self.pool = nn.AdaptiveMaxPool1d(1)

        # Classifier
        self.fc1 = nn.Linear(256, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.output = nn.Linear(256, 1)

    def forward(self, x):
        # x: (B, T, F) = (B, 7, 70)
        # ============================== TODO [Ex 2.2] ==============================
        # x: (B, T, F) = (B, 7, 70)  ->  logits (B, 1)
        #   1. Temporal CNN: conv1 -> ReLU -> conv2 -> ReLU -> batchnorm
        #      (Conv1d wants (B, F, T); transpose back to (B, T, 128) afterwards)
        #   2. Bi-GRU -> (B, T, 512)
        #   3. Self-attention: project the GRU output with query/key/value layers,
        #      run self.attention(query, key, value), then add a residual:
        #      attn_out + query -> (B, T, 128)
        #   4. Pool over time with BOTH max (self.pool) and mean and concatenate
        #      [max, mean] -> (B, 256)
        #   5. Classifier: fc1 -> ReLU -> dropout -> fc2 -> ReLU -> dropout2 -> output
        # ===========================================================================
        raise NotImplementedError("TODO [Ex 2.2]")

