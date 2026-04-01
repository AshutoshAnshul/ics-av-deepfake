# modified from https://github.com/ControlNet/LAV-DF/blob/master/model/audio_encoder.py

from typing import Literal

from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor
from torch.nn import Module, Sequential, LeakyReLU, MaxPool2d, Linear
from model.vit_encoder import Encoder as ViTEncoder
import torch

from utils import Conv2d

class CNNAudioEncoder(Module):
    """
    Audio encoder (E_a): Process log mel spectrogram to extract features.
    Input:
        A': (B, F_m, T_a)
    Output:
        E_a: (B, C_f, T)
    """

    def __init__(self, n_features=(32, 64, 64)):
        super().__init__()

        n_dim0, n_dim1, n_dim2 = n_features

        # (B, 64, 2048) -> (B, 1, 64, 2048) -> (B, 32, 32, 1024)
        self.block0 = Sequential(
            Rearrange("b c t -> b 1 c t"),
            Conv2d(1, n_dim0, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            MaxPool2d(2)
        )

        # (B, 32, 32, 1024) -> (B, 64, 16, 512)
        self.block1 = Sequential(
            Conv2d(n_dim0, n_dim1, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            Conv2d(n_dim1, n_dim1, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            MaxPool2d(2)
        )

        # (B, 64, 16, 512) -> (B, 64, 4, 512) -> (B, 256, 512)
        self.block2 = Sequential(
            Conv2d(n_dim1, n_dim2, kernel_size=(2, 1), stride=1, padding=(1, 0), build_activation=LeakyReLU),
            MaxPool2d((2, 1)),
            Conv2d(n_dim2, n_dim2, kernel_size=(3, 1), stride=1, padding=(1, 0), build_activation=LeakyReLU),
            MaxPool2d((2, 1)),
            Rearrange("b f c t -> b (f c) t")
        )

    def forward(self, audio: Tensor) -> Tensor:
        x = self.block0(audio)
        x = self.block1(x)
        x = self.block2(x)
        return x


class SelfAttentionAudioEncoder(Module):

    def __init__(self, block_type: Literal["vit_t", "vit_s", "vit_b"], a_cla_feature_in: int = 256, temporal_size: int = 512):
        super().__init__()
        # The ViT configurations are from:
        # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
        if block_type == "vit_t":
            self.n_features = 192
            self.block = ViTEncoder(
                max_seq_len=temporal_size,
                num_layers=12,
                num_heads=3,
                hidden_dim=self.n_features,
                mlp_dim=self.n_features * 4,
                dropout=0.,
                attention_dropout=0.
            )
        elif block_type == "vit_s":
            self.n_features = 384
            self.block = ViTEncoder(
                max_seq_len=temporal_size,
                num_layers=12,
                num_heads=6,
                hidden_dim=self.n_features,
                mlp_dim=self.n_features * 4,
                dropout=0.,
                attention_dropout=0.
            )
        elif block_type == "vit_b":
            self.n_features = 768
            self.block = ViTEncoder(
                max_seq_len=temporal_size,
                num_layers=12,
                num_heads=12,
                hidden_dim=self.n_features,
                mlp_dim=self.n_features * 4,
                dropout=0.,
                attention_dropout=0.
            )
        else:
            raise ValueError(f"Unknown block type: {block_type}")

        self.input_proj = Conv2d(1, self.n_features, kernel_size=(64, 4), stride=(64, 4))
        self.output_proj = Linear(self.n_features, a_cla_feature_in)

    def forward(self, audio: Tensor, att_mask) -> Tensor:
        x = audio.unsqueeze(1)  # (B, 64, 2048) -> (B, 1, 64, 2048)
        x = self.input_proj(x)  # (B, 1, 64, 2048) -> (B, feat, 1, 512)
        x = rearrange(x, "b f 1 t -> b t f")  # (B, feat, 1, 512) -> (B, 512, feat)
        x = self.block(x, att_mask)
        x = self.output_proj(x)  # (B, 512, feat) -> (B, 512, 256)
        x = x.permute(0, 2, 1)  # (B, 512, 256) -> (B, 256, 512)
        return x

class SelfAttentionFrameLevelAudioEncoder(Module):

    def __init__(self, block_type: Literal["vit_t", "vit_s", "vit_b"], a_cla_feature_in: int = 256, temporal_size: int = 512, cls_tok=True):
        super().__init__()
        # The ViT configurations are from:
        # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
        if block_type == "vit_t":
            self.n_features = 192
            self.block = ViTEncoder(
                max_seq_length=temporal_size,
                num_layers=12,
                num_heads=3,
                hidden_dim=self.n_features,
                mlp_dim=self.n_features * 4,
                dropout=0.,
                attention_dropout=0.
            )
        elif block_type == "vit_s":
            self.n_features = 384
            self.block = ViTEncoder(
                max_seq_length=temporal_size,
                num_layers=12,
                num_heads=6,
                hidden_dim=self.n_features,
                mlp_dim=self.n_features * 4,
                dropout=0.,
                attention_dropout=0.
            )
        elif block_type == "vit_b":
            self.n_features = 768
            self.block = ViTEncoder(
                max_seq_length=temporal_size,
                num_layers=12,
                num_heads=12,
                hidden_dim=self.n_features,
                mlp_dim=self.n_features * 4,
                dropout=0.,
                attention_dropout=0.
            )
        else:
            raise ValueError(f"Unknown block type: {block_type}")

        self.input_proj = Conv2d(1, self.n_features, kernel_size=(16, 4), stride=(16, 4))
        self.output_proj = Linear(4*self.n_features, a_cla_feature_in)
    
    def forward(self, audio: Tensor) -> Tensor:
        num_frames = audio.shape[-1]//4
        x = rearrange(audio, "b n (k s) -> b k n s", s=4) # (B, 64, n) -> (B, 512, k, 4) , k = n//4
        x = self.threeD_to_2D_tensor(x) # (B, 512, 64, 4) -> (B*k, 1, 64, 4)
        x = self.input_proj(x)  # (B*k, 1, 64, 4) -> (B*k, feat, 4, 1)
        x = rearrange(x, "b f t 1 -> b t f")  # (B*k, feat, 4, 1) -> (B*k, 4, feat)
        x = self.block(x)
        x = torch.flatten(x, start_dim=1)
        x = rearrange(x, "(b n) k -> b n k", n=num_frames)
        
        x = self.output_proj(x)  # (B, k, feat) -> (B, k, 256)
        x = x.permute(0, 2, 1)  # (B, 512, k) -> (B, 256, k)
        return x 
    
    def threeD_to_2D_tensor(self, x):
        n_batch, s_time, sx, sy = x.shape
        return x.reshape(n_batch*s_time, sx, sy).unsqueeze(1)


def get_audio_encoder(a_cla_feature_in, temporal_size, a_encoder):
    a_encoder = str(a_encoder).strip()
    if a_encoder == "vit_t":
        audio_encoder = SelfAttentionAudioEncoder(block_type="vit_t", a_cla_feature_in=a_cla_feature_in, temporal_size=temporal_size)
    elif a_encoder == "vit_s":
        audio_encoder = SelfAttentionAudioEncoder(block_type="vit_s", a_cla_feature_in=a_cla_feature_in, temporal_size=temporal_size)
    elif a_encoder == "vit_b":
        audio_encoder = SelfAttentionAudioEncoder(block_type="vit_b", a_cla_feature_in=a_cla_feature_in, temporal_size=temporal_size)
    elif a_encoder == "frame_vit_t":
        audio_encoder = SelfAttentionFrameLevelAudioEncoder(block_type="vit_t", a_cla_feature_in=a_cla_feature_in, temporal_size=temporal_size)
    elif a_encoder == "frame_vit_s":
        audio_encoder = SelfAttentionFrameLevelAudioEncoder(block_type="vit_s", a_cla_feature_in=a_cla_feature_in, temporal_size=temporal_size)
    elif a_encoder == "frame_vit_b":
        audio_encoder = SelfAttentionFrameLevelAudioEncoder(block_type="vit_b", a_cla_feature_in=a_cla_feature_in, temporal_size=temporal_size)
    else:
        raise ValueError(f"Invalid audio encoder: '{a_encoder}'")
    return audio_encoder
