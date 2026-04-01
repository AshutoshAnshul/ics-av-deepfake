import torch.nn as nn
import torch.nn.functional as F
from torch import cat

class SparseAttentionEncoder(nn.Module):
    def __init__(self, feature_dim, kernel_size=5, num_groups=8):
        super(SparseAttentionEncoder, self).__init__()
        self.feature_dim = feature_dim
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2  # To maintain same length

        self.ln = nn.LayerNorm(feature_dim//2)

        self.shared_conv_q = nn.Conv1d(in_channels=feature_dim//2, out_channels=feature_dim//2, kernel_size=kernel_size, padding=self.padding, groups=feature_dim//2)

        self.shared_conv_v = nn.Conv1d(in_channels=feature_dim//2, out_channels=feature_dim//2, kernel_size=kernel_size, padding=self.padding, groups=feature_dim//2)

        # 1D Convolutions for Q and V
        self.pointwise_conv_q = nn.Conv1d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=1)

        self.pointwise_conv_v = nn.Conv1d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=1)

        # Group normalization
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=feature_dim)

        # Two convolutional layers for the residual block
        self.residual_conv1 = nn.Conv1d(in_channels=feature_dim, out_channels=2*feature_dim, kernel_size=1, padding=0)
        self.residual_conv2 = nn.Conv1d(in_channels=2*feature_dim, out_channels=feature_dim, kernel_size=1, padding=0)

    def forward(self, video_encoding, audio_encoding, mask=None):
        video_encoding = self.ln(video_encoding)
        audio_encoding = self.ln(audio_encoding)

        video_encoding = video_encoding.transpose(1,2)
        audio_encoding = audio_encoding.transpose(1,2)

        # Concatenate video and audio to form original x signal (B, F, T)
        x = cat([video_encoding, audio_encoding], dim=1)

        video_q = self.shared_conv_q(video_encoding)
        audio_q = self.shared_conv_q(audio_encoding)

        video_v = self.shared_conv_v(video_encoding)
        audio_v = self.shared_conv_v(audio_encoding)

        q = cat([video_q, audio_q], dim=1)
        v = cat([video_v, audio_v], dim=1)

        # Apply convolution to get Q and V
        q = self.pointwise_conv_q(q)  # Shape: (B, F, T)
        v = self.pointwise_conv_v(v)  # Shape: (B, F, T)

        # Apply mask to both Q and V
        # Mask is of size (B, T), expand it to (B, 1, T) to apply it to the Q and V
        if mask is not None:
            mask = (~mask).unsqueeze(1)  # Shape: (B, 1, T)
            q = q * mask
            v = v * mask

        # Element-wise multiplication between Q and V, followed by residual addition
        attention_result = q * v
        x = x + attention_result

        # Residual block
        residual = F.relu(self.residual_conv1(self.group_norm(x)))
        residual = self.residual_conv2(residual)
        x = x + residual

        # Transpose back to the original shape (B, T, F)
        x = x.transpose(1, 2)

        return x
