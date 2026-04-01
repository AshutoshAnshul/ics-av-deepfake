import torch
import torch.nn as nn
import torch.nn.functional as F
from model.transformer import SparseAttentionEncoder
from typing import Optional
from torch import Tensor
from einops import rearrange
import math

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class SyncModel(nn.Module):
    def __init__(self, segment_size=5, feature_size=256, num_heads=4, depth=3, max_len=750):
        super(SyncModel, self).__init__()

        self.position_embed = PositionalEncoding(feature_size, 0, max_len)

        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=num_heads, batch_first=True, dim_feedforward=2*feature_size)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)

        self.conv = nn.Conv1d(in_channels=feature_size, out_channels=feature_size, kernel_size=segment_size, padding=segment_size//2, groups=feature_size)

        self.linear_q = nn.Linear(feature_size, feature_size)
        self.linear_v = nn.Linear(feature_size, feature_size)

    def forward(self, video_encoding: Tensor, audio_encoding: Tensor, src_mask: Tensor, tgt_mask: Tensor, shift:int):
        
        bs, _ , fs = video_encoding.shape

        # print(video_encoding.device, audio_encoding.device, tgt_mask.device, src_mask.device, padding_mask.device)

        # Shift the audio encoding
        if shift<0:
            pad = torch.zeros(bs, abs(shift), fs).to(video_encoding.device)
            audio_encoding = torch.cat([pad, audio_encoding[:, :shift, :]], dim=1)
        elif shift>0:
            pad = torch.zeros(bs, shift, fs).to(video_encoding.device)
            audio_encoding = torch.cat([audio_encoding[:, shift:, :], pad], dim=1)

        # Add positional Encoding
        video_encoding = self.position_embed(video_encoding)
        audio_encoding = self.position_embed(audio_encoding)

        # Pass the encodings through cross attention
        video_encoding_feat = self.transformer_decoder(video_encoding, audio_encoding, tgt_mask, src_mask)

        # print(shift, torch.isnan(video_encoding_feat).sum())

        # Permute to match the input shape expected by Conv1d: (B, F, T)
        video_encoding = video_encoding_feat.permute(0, 2, 1)
        audio_encoding = audio_encoding.permute(0, 2, 1)
        
        # Apply 1D convolution to incorporate context
        video_encoding = self.conv(video_encoding).permute(0, 2, 1)
        audio_encoding = self.conv(audio_encoding).permute(0, 2, 1)

        # Generate q and v learnable parameter
        video_encoding = self.linear_q(video_encoding)
        audio_encoding = self.linear_v(audio_encoding)

        # Get the similarity score
        return torch.einsum('bif,bif->bi', video_encoding, audio_encoding), video_encoding_feat


class SyncModelSparse(nn.Module):
    def __init__(self, segment_size=5, feature_size=256, depth=3, max_len=750):
        super(SyncModelSparse, self).__init__()

        self.feature_size = feature_size
        self.position_embed = PositionalEncoding(feature_size, 0, max_len)

        sparse_attention_layers = [
            SparseAttentionEncoder(feature_dim=2*feature_size, kernel_size=segment_size) for _ in range(depth)
        ]
        self.sparse_encoder = nn.Sequential(*sparse_attention_layers)

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=feature_size*2, out_features=256),
            nn.Linear(in_features=256, out_features=64),
            nn.Linear(in_features=64, out_features=1)
        )


    def forward(self, video_encoding: Tensor, audio_encoding: Tensor, padding_mask: Tensor, shift:int):
        
        bs, _ , fs = video_encoding.shape

        # print(video_encoding.device, audio_encoding.device, tgt_mask.device, src_mask.device, padding_mask.device)

        # Shift the audio encoding
        if shift<0:
            pad = torch.zeros(bs, abs(shift), fs).to(video_encoding.device)
            audio_encoding = torch.cat([pad, audio_encoding[:, :shift, :]], dim=1)
            padding_mask[:, :abs(shift)] = True
        elif shift>0:
            pad = torch.zeros(bs, shift, fs).to(video_encoding.device)
            audio_encoding = torch.cat([audio_encoding[:, shift:, :], pad], dim=1)
            extra_mask = torch.ones(bs, shift, dtype=torch.bool).to(video_encoding.device)
            padding_mask = torch.cat([padding_mask[:, shift:], extra_mask], dim=1)

        # Add positional Encoding
        video_encoding = self.position_embed(video_encoding)
        audio_encoding = self.position_embed(audio_encoding)

        # Pass the encodings through cross attention
        x_feat = None
        for layer in self.sparse_encoder:
            x_feat = layer(video_encoding, audio_encoding, padding_mask)
            video_encoding = x_feat[:, :, :self.feature_size]
            audio_encoding = x_feat[:, :, self.feature_size:]

        # Get the similarity score
        return self.linear_layers(x_feat).squeeze(-1), x_feat
