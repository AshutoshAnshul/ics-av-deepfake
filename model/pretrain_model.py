from typing import Dict, Optional, Union, Sequence, Tuple
import torch
import math
from pytorch_lightning import LightningModule
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from einops import rearrange
from pytorch_lightning import LightningModule

from model.sync_model import SyncModel, SyncModelSparse
from model.video_encoder import get_video_encoder
from model.audio_encoder import get_audio_encoder

from torch.optim import AdamW

from utils import padding_video, padding_audio

import datetime

class ConsistencyPretrainModel(LightningModule):
    def __init__(self,
        v_encoder: str = "mvit_v2_t", a_encoder: str = "vit_t", v_cla_feature_in=256, a_cla_feature_in=256,
        temporal_dim=512, weight_decay=0.0001, learning_rate=0.0001, final_learning_rate=0.000025, total_optimizer_step=50, distributed=False, sync_loss = 'gaussian', modal_consistency = 'Sync', select_tgt_layer = 'combiner',
        context_size=15, segment_size=15, num_heads=4, depth=3, d_model=512, is_pre_padded=True, fps=25, max_len=750
    ):
        super().__init__()
        self.save_hyperparameters()

        assert modal_consistency in ['Sync', 'AutoReg'], "Intra Modal Consistency should be Sync or AutoReg"

        assert sync_loss in ['single', 'gaussian'], "Synchronization loss should be in single or gaussian"

        assert v_cla_feature_in == a_cla_feature_in, "Visual and Audio Feature size should be same"

        assert select_tgt_layer in ['combiner', 'original'], "The target layer is the layer from where the autoregressive will compute its loss. it can be only to the combiner encoder's output (combiner) or the original feature set (original)"

        self.cla_feature_in = v_cla_feature_in
        self.temporal_dim = temporal_dim

        self.segment_size = segment_size
        self.context_size = context_size
        
        self.top_mask = torch.rot90(torch.triu(torch.ones(context_size, context_size, dtype=torch.bool)), k=1, dims=(0, 1)).requires_grad_(False)
        self.bottom_mask = torch.rot90(self.top_mask, k=2, dims=(0, 1)).requires_grad_(False)

        self.video_encoder = get_video_encoder(v_cla_feature_in, temporal_dim, v_encoder, segment_size)
        self.audio_encoder = get_audio_encoder(a_cla_feature_in, temporal_dim, a_encoder)

        self.modal_consistency = modal_consistency
        self.sync_loss = sync_loss
        self.select_tgt_layer = select_tgt_layer
        self.is_pre_padded = is_pre_padded
        self.fps = fps
        self.max_len = max_len

        self.learning_rate = learning_rate
        self.final_learning_rate = final_learning_rate
        self.total_steps = total_optimizer_step
        self.weight_decay = weight_decay
        self.distributed = distributed

        self.visual_enocoding = nn.Parameter(torch.randn(1, 1, 1))
        self.audio_enocoding = nn.Parameter(torch.randn(1, 1, 1))

        self.tgt_mask = self.generate_mask(segment_size, 0, max_len)
        self.src_mask = self.generate_mask(segment_size, 0, max_len)
        self.num_heads = num_heads

        if modal_consistency=='Sync':
            self.cross_modal_consistency= SyncModel(segment_size=segment_size, feature_size=v_cla_feature_in, num_heads=num_heads, depth=depth)
            self.video_consistency = SyncModel(segment_size=segment_size, feature_size=v_cla_feature_in, num_heads=num_heads, depth=depth)
            self.audio_consistency = SyncModel(segment_size=segment_size, feature_size=v_cla_feature_in, num_heads=num_heads, depth=depth)

    def generate_mask(self, segment_size:int, shift:int, size:int):
        # function to generate a diagonal mask with width segment_size shifted by value shift.
        # hence it will generate a (size, size) boolean matrix, where for i-th row, entry in 
        # column range from (i-(segment_size // 2)+shift) to (i+(segment_size // 2)+shift) 
        # will be false and rest everything will be true

        half_k = segment_size // 2

        # Create a base mask with all zeros
        mask = torch.ones((size, size))

        # Create the index ranges for each row
        row_indices = torch.arange(size).unsqueeze(1)
        col_indices = torch.arange(-half_k + shift, half_k + shift + 1).unsqueeze(0)

        # Create the mask using broadcasting
        col_indices = col_indices + row_indices
        col_indices = col_indices.clamp(0, size - 1)
        
        # Fill the mask
        mask[row_indices, col_indices] = 0
        
        return mask.requires_grad_(False).bool()
    
    def maskify_tgt_tensor(self, mask: Tensor, video_frames: Tensor, shift=0):
        bs = len(video_frames)
        num_frames = mask.size(0)
        tot_len = torch.arange(bs).to(video_frames.device)
        mask = mask.unsqueeze(0).expand(bs, -1, -1).clone().to(video_frames.device)
        row_indices = video_frames.unsqueeze(1) - 1 - shift
        col_indices = torch.tensor([1, 2]).unsqueeze(0).to(video_frames.device)
        col_indices = col_indices + row_indices
        col_indices = col_indices.clamp(0, self.temporal_dim - 1)
        
        mask[tot_len, row_indices.transpose(0,1), col_indices.transpose(0,1)] = True
        
        row_indices = video_frames.unsqueeze(1) - 2 - shift
        col_indices = torch.tensor([2]).unsqueeze(0).to(video_frames.device)
        col_indices = col_indices + row_indices
        col_indices = col_indices.clamp(0, self.temporal_dim - 1)
        
        mask[tot_len, row_indices.transpose(0,1), col_indices.transpose(0,1)] = True
        
        full_length_list = tot_len[video_frames==self.temporal_dim]
        mask[full_length_list, self.temporal_dim-2, self.temporal_dim-1] = False
        mask[full_length_list, self.temporal_dim-1, self.temporal_dim-1] = False

        mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1).reshape(-1, num_frames, num_frames)
        
        return mask.requires_grad_(False).bool()
    
    def maskify_src_tensor(self, mask: Tensor, video_frames: Tensor, shift):
        
        if shift<=0:
            if shift<0:
                shift = abs(shift)
                if shift==1:
                    mask[1, 0] = True
                    mask[2, 0] = True
                else:
                    mask[shift, shift-2] = True
                    mask[shift, shift-1] = True
                    mask[shift+1, shift-1] = True
            return self.maskify_tgt_tensor(mask, video_frames, 0)
        
        else:
            return self.maskify_tgt_tensor(mask, video_frames, shift)
    
    def forward(self, video: Tensor, audio: Tensor, video_frames: Tensor) -> Sequence[Tensor]:

        batch_size, _, num_frames, _, _ = video.shape

        src_mask = self.src_mask[:num_frames, :num_frames].to(video.device)
        tgt_mask = self.tgt_mask[:num_frames, :num_frames].to(video.device)
        
        v_feat = self.video_encoder(video) 
        a_feat = self.audio_encoder(audio, tgt_mask)

        del video, audio

        tgt_mask = self.maskify_tgt_tensor(tgt_mask, video_frames).to(v_feat.device)

        v_feat = v_feat.permute(0, 2, 1)
        a_feat = a_feat.permute(0, 2, 1)

        v_feat+=self.visual_enocoding
        a_feat+=self.audio_enocoding

        if self.modal_consistency=='Sync':

            cross_modal_out = torch.zeros(batch_size, num_frames, 2*self.context_size+1).to(v_feat.device)
            v_out = torch.zeros(batch_size, num_frames, 2*self.context_size+1).to(v_feat.device)
            a_out = torch.zeros(batch_size, num_frames, 2*self.context_size+1).to(v_feat.device)

            av_hidden_feat = torch.zeros(batch_size, num_frames, self.cla_feature_in).to(v_feat.device)
            v_hidden_feat = torch.zeros(batch_size, num_frames, self.cla_feature_in).to(v_feat.device)
            a_hidden_feat = torch.zeros(batch_size, num_frames, self.cla_feature_in).to(v_feat.device)
            
            context_range = 2*self.context_size+1

            for i in range(context_range):
                src_mask_shifted = self.maskify_src_tensor(src_mask, video_frames, i-self.context_size).to(v_feat.device)

                if i==0:
                    cross_modal_out[:, :, i], av_hidden_feat = self.cross_modal_consistency(v_feat, a_feat, src_mask_shifted, tgt_mask, i-self.context_size)
                    v_out[:, :, i], v_hidden_feat = self.video_consistency(v_feat, v_feat, src_mask_shifted, tgt_mask, i-self.context_size)
                    a_out[:, :, i], a_hidden_feat = self.audio_consistency(a_feat, a_feat, src_mask_shifted, tgt_mask, i-self.context_size) 
                else:
                    cross_modal_out[:, :, i], _ = self.cross_modal_consistency(v_feat, a_feat, src_mask_shifted, tgt_mask, i-self.context_size)
                    v_out[:, :, i], _ = self.video_consistency(v_feat, v_feat, src_mask_shifted, tgt_mask, i-self.context_size)
                    a_out[:, :, i], _ = self.audio_consistency(a_feat, a_feat, src_mask_shifted, tgt_mask, i-self.context_size)

        return (cross_modal_out, v_out, a_out, av_hidden_feat, v_hidden_feat, a_hidden_feat, v_feat, a_feat)
    
    def gaussian_targets(self, size, center, std_dev):
        """
        Generate a Gaussian distribution centered at the middle index.
        
        Parameters:
        size (int): The size of the context window.
        center (int): The center index.
        std_dev (float): The standard deviation of the Gaussian distribution.
        
        Returns:
        torch.Tensor: The Gaussian distribution.
        """
        x = torch.arange(size).float()
        gaussian = torch.exp(-0.5 * ((x - center) ** 2) / (std_dev ** 2))
        gaussian = gaussian / gaussian.sum()  # Normalize to ensure it sums to 1
        return gaussian
    
    def sync_kl_loss(self, sync_output: Tensor, video_frames, std_dev=1.0, mode='av'):
        """
        Compute the AV synchronization KL divergence loss.
        
        Parameters:
        av_sync_output (torch.Tensor): The output from the AVSyncModel, shaped (batch_size, num_frames, 2*context_size+1).
        context_size (int): The size of the context window.
        std_dev (float): The standard deviation for the Gaussian distribution.
        
        Returns:
        torch.Tensor: The computed loss.
        """
        batch_size, num_frames, context_range = sync_output.shape
        assert context_range == 2 * self.context_size + 1, "Context range does not match the expected size."

        # Step 1: Compute mask without including the padding
        mask = torch.zeros_like(sync_output, dtype=torch.bool).to(sync_output.device).requires_grad_(False)
        actual_frame = torch.min(video_frames, torch.tensor(self.temporal_dim))

        mask[:, :self.context_size, :self.context_size] = self.top_mask
        
        for i in range(batch_size):
            mask[i, actual_frame[i]-self.context_size:actual_frame[i], -self.context_size:] = self.bottom_mask

        # Step 2: Maskify output and Compute softmax along the context dimension
        # log_softmax_output = F.log_softmax(sync_output, dim=-1)  # (batch_size, num_frames, context_range)
        add_val = torch.zeros_like(sync_output, dtype=torch.bool).to(sync_output.device).requires_grad_(False)
        add_val = add_val.masked_fill(mask, float('-inf'))
        softmax_output = F.softmax((sync_output + add_val), dim=-1) # (batch_size, num_frames, context_range)

        # Step 3: Generate Gaussian targets
        center = self.context_size
        if mode=='av':
            gaussian_target = self.gaussian_targets(context_range, center, std_dev).to(sync_output.device)  # (context_range)
        else:
            gaussian_target = torch.maximum(self.gaussian_targets(context_range, center-1, std_dev), self.gaussian_targets(context_range, center+1, std_dev)).to(sync_output.device)  # (context_range)
            gaussian_target = gaussian_target / gaussian_target.sum()
        
        # Step 3: Compute KL-Divergence/MSE
        # kl_div = F.kl_div(log_softmax_output, gaussian_target.expand(batch_size, num_frames, -1), reduction='none')  # (batch_size, num_frames, context_range)
        kl_div = (softmax_output - gaussian_target.expand(batch_size, num_frames, -1)) ** 2
        
        
        # Step 4: Include padding in mask
        for i in range(batch_size):
            mask[i, actual_frame[i]:] = True

        # Step 5: Compute the mean loss all frames and batch

        # mask = ~mask
        # softmax_output = softmax_output[mask]
        # gaussian_target = gaussian_target.expand(batch_size, num_frames, -1)[mask]
        # loss = F.mse_loss(softmax_output, gaussian_target)
        kl_div[mask] = 0
        mask = ~mask
        loss = kl_div.sum()/mask.sum()

        return loss
    
    def sync_single_loss(self, sync_output, padding_mask, mode='av'):
        """
        Compute the AV synchronization loss.
        
        Parameters:
        av_sync_output (torch.Tensor): The output from the AVSyncModel, shaped (batch_size, num_frames, 2*context_size+1).
        context_size (int): The size of the context window.
        
        Returns:
        torch.Tensor: The computed loss.
        """
        _, _, context_range = sync_output.shape
        assert context_range == 2 * self.context_size + 1, "Context range does not match the expected size."

        # Step 1: Compute softmax along the context dimension
        softmax_output = F.softmax(sync_output, dim=-1)  # (batch_size, num_frames, context_range)

        # Step 2: Extract the middle value
        middle_index = self.context_size

        if mode == 'av':
            middle_values = softmax_output[:, :, middle_index]  # (batch_size, num_frames)
        else:
            middle_values = softmax_output[:, :, middle_index-1] * softmax_output[:, :, middle_index+1]  # (batch_size, num_frames)

        # Step 3: Compute the negative log of the middle values
        log_middle_values = torch.log(middle_values)  # (batch_size, num_frames)

        log_middle_values[padding_mask] = float('nan')

        loss = -log_middle_values.nanmean()  # Average over all frames and batch

        return loss

    def loss_fn(self, cross_modal_out: Tensor, v_out: Tensor, a_out: Tensor, padding_mask: Tensor, video_frames):
        
        # In case of only maximizing single value
        if self.sync_loss == 'single':
            av_loss = self.sync_single_loss(cross_modal_out, padding_mask, 'av')
            v_loss = self.sync_single_loss(v_out, padding_mask, 'v')
            a_loss = self.sync_single_loss(a_out, padding_mask, 'a')
        
        # In case of fitting to a gaussian curve
        else:
            av_loss = self.sync_kl_loss(cross_modal_out, video_frames, 1.5, 'av')
            v_loss = self.sync_kl_loss(v_out, video_frames, 1.5, 'v')
            a_loss = self.sync_kl_loss(a_out, video_frames, 1.5, 'a')
                
        loss = av_loss + v_loss + a_loss
        loss_dict = {"loss": loss, "av_loss": av_loss, "v_loss": v_loss, "a_loss": a_loss}

        return {k: v for k, v in loss_dict.items() if v is not None}

    def step(self, batch: Sequence[Tensor]) -> Dict[str, Tensor]:

        (cross_modal_out, v_out, a_out, _, _, _, _, _) = self.forward(batch['video'], batch['audio'], batch['video_frames'])

        loss_dict = self.loss_fn(cross_modal_out, v_out, a_out, batch['padding_mask'], batch['video_frames'])

        del batch
        
        return loss_dict
    
    
    def training_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None
    ) -> Tensor:
        loss_dict = self.step(batch)
        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=True, sync_dist=self.distributed)
        
        return loss_dict["loss"]
    
    def validation_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None
    ) -> Tensor:
        loss_dict = self.step(batch)

        self.log_dict({f"val_{k}": v for k, v in loss_dict.items()}, on_step=False, on_epoch=True,
            prog_bar=False, sync_dist=self.distributed)
        return loss_dict["loss"]

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Tuple [Tensor, Tensor, Tensor, Tensor, Tensor]:
        
        (cross_modal_out, v_out, a_out, av_hidden_feat, v_hidden_feat, a_hidden_feat, v_feat, a_feat) = self.forward(batch['video'], batch['audio'], batch['video_frames'])
        return cross_modal_out, v_out, a_out, av_hidden_feat, v_hidden_feat, a_hidden_feat, v_feat, a_feat

        # video = batch['video']
        # audio = batch['audio']
        # video_frames = batch['video_frames']

        # batch_size, _, num_frames, _, _ = video.shape

        # src_mask = self.src_mask[:num_frames, :num_frames].to(video.device)
        # tgt_mask = self.tgt_mask[:num_frames, :num_frames].to(video.device)
        
        # v_feat = self.video_encoder(video) 
        # a_feat = self.audio_encoder(audio, tgt_mask)

        # del video, audio

        # tgt_mask = self.maskify_tgt_tensor(tgt_mask, video_frames).to(v_feat.device)

        # v_feat = v_feat.permute(0, 2, 1)
        # a_feat = a_feat.permute(0, 2, 1)

        # v_feat+=self.visual_enocoding
        # a_feat+=self.audio_enocoding

        # src_mask_shifted = self.maskify_src_tensor(src_mask, video_frames, 0-self.context_size).to(v_feat.device)
        # _, av_hidden_feat = self.cross_modal_consistency(v_feat, a_feat, src_mask_shifted, tgt_mask, 0-self.context_size)

        # return av_hidden_feat, v_feat, a_feat
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True, min_lr=1e-8),
                "monitor": "val_loss"
            }
        }
    



#- Consistency code for sparse attention encoder



class ConsistencyPretrainEncoderModel(LightningModule):
    def __init__(self,
        v_encoder: str = "mvit_v2_t", a_encoder: str = "vit_t", v_cla_feature_in=256, a_cla_feature_in=256,
        temporal_dim=512, weight_decay=0.0001, learning_rate=0.0001, final_learning_rate=0.000025, total_optimizer_step=50, distributed=False, sync_loss = 'single', modal_consistency = 'Sync', select_tgt_layer = 'combiner',
        context_size=15, segment_size=15, num_heads=4, depth=3, d_model=512, is_pre_padded=True, fps=25, max_len=750
    ):
        super().__init__()
        self.save_hyperparameters()

        assert modal_consistency in ['Sync', 'AutoReg'], "Intra Modal Consistency should be Sync or AutoReg"

        assert sync_loss in ['single', 'gaussian'], "Synchronization loss should be in single or gaussian"

        assert v_cla_feature_in == a_cla_feature_in, "Visual and Audio Feature size should be same"

        assert select_tgt_layer in ['combiner', 'original'], "The target layer is the layer from where the autoregressive will compute its loss. it can be only to the combiner encoder's output (combiner) or the original feature set (original)"

        self.cla_feature_in = v_cla_feature_in
        self.temporal_dim = temporal_dim

        self.segment_size = segment_size
        self.context_size = context_size
        
        self.top_mask = torch.rot90(torch.triu(torch.ones(context_size, context_size, dtype=torch.bool)), k=1, dims=(0, 1)).requires_grad_(False)
        self.bottom_mask = torch.rot90(self.top_mask, k=2, dims=(0, 1)).requires_grad_(False)

        self.video_encoder = get_video_encoder(v_cla_feature_in, temporal_dim, v_encoder, segment_size)
        self.audio_encoder = get_audio_encoder(a_cla_feature_in, temporal_dim, a_encoder)

        self.modal_consistency = modal_consistency
        self.sync_loss = sync_loss
        self.select_tgt_layer = select_tgt_layer
        self.is_pre_padded = is_pre_padded
        self.fps = fps
        self.max_len = max_len

        self.learning_rate = learning_rate
        self.final_learning_rate = final_learning_rate
        self.total_steps = total_optimizer_step
        self.weight_decay = weight_decay
        self.distributed = distributed

        self.visual_enocoding = nn.Parameter(torch.randn(1, 1, 1))
        self.audio_enocoding = nn.Parameter(torch.randn(1, 1, 1))

        self.tgt_mask = self.generate_mask(segment_size, 0, max_len)
        self.num_heads = num_heads

        if modal_consistency=='Sync':
            self.cross_modal_consistency= SyncModelSparse(segment_size=segment_size, feature_size=v_cla_feature_in, depth=depth)
            self.video_consistency = SyncModelSparse(segment_size=segment_size, feature_size=v_cla_feature_in, depth=depth)
            self.audio_consistency = SyncModelSparse(segment_size=segment_size, feature_size=v_cla_feature_in, depth=depth)
    
    def generate_mask(self, segment_size:int, shift:int, size:int):
        # function to generate a diagonal mask with width segment_size shifted by value shift.
        # hence it will generate a (size, size) boolean matrix, where for i-th row, entry in 
        # column range from (i-(segment_size // 2)+shift) to (i+(segment_size // 2)+shift) 
        # will be false and rest everything will be true

        half_k = segment_size // 2

        # Create a base mask with all zeros
        mask = torch.ones((size, size))

        # Create the index ranges for each row
        row_indices = torch.arange(size).unsqueeze(1)
        col_indices = torch.arange(-half_k + shift, half_k + shift + 1).unsqueeze(0)

        # Create the mask using broadcasting
        col_indices = col_indices + row_indices
        col_indices = col_indices.clamp(0, size - 1)
        
        # Fill the mask
        mask[row_indices, col_indices] = 0
        
        return mask.requires_grad_(False).bool()

    def forward(self, video: Tensor, audio: Tensor, padding_mask: Tensor) -> Sequence[Tensor]:

        batch_size, _, num_frames, _, _ = video.shape

        tgt_mask = self.tgt_mask[:num_frames, :num_frames].to(video.device)
        
        v_feat = self.video_encoder(video) 
        a_feat = self.audio_encoder(audio, tgt_mask)

        del video, audio

        v_feat = v_feat.permute(0, 2, 1)
        a_feat = a_feat.permute(0, 2, 1)

        v_feat+=self.visual_enocoding
        a_feat+=self.audio_enocoding

        if self.modal_consistency=='Sync':

            cross_modal_out = torch.zeros(batch_size, num_frames, 2*self.context_size+1).to(v_feat.device)
            v_out = torch.zeros(batch_size, num_frames, 2*self.context_size+1).to(v_feat.device)
            a_out = torch.zeros(batch_size, num_frames, 2*self.context_size+1).to(v_feat.device)

            av_hidden_feat = torch.zeros(batch_size, num_frames, self.cla_feature_in).to(v_feat.device)
            v_hidden_feat = torch.zeros(batch_size, num_frames, self.cla_feature_in).to(v_feat.device)
            a_hidden_feat = torch.zeros(batch_size, num_frames, self.cla_feature_in).to(v_feat.device)
            
            context_range = 2*self.context_size+1

            for i in range(context_range):

                if i==0:
                    cross_modal_out[:, :, i], av_hidden_feat = self.cross_modal_consistency(v_feat, a_feat, padding_mask, i-self.context_size)
                    v_out[:, :, i], v_hidden_feat = self.video_consistency(v_feat, v_feat, padding_mask, i-self.context_size)
                    a_out[:, :, i], a_hidden_feat = self.audio_consistency(a_feat, a_feat, padding_mask, i-self.context_size) 
                else:
                    cross_modal_out[:, :, i], _ = self.cross_modal_consistency(v_feat, a_feat, padding_mask, i-self.context_size)
                    v_out[:, :, i], _ = self.video_consistency(v_feat, v_feat, padding_mask, i-self.context_size)
                    a_out[:, :, i], _ = self.audio_consistency(a_feat, a_feat, padding_mask, i-self.context_size)

        return (cross_modal_out, v_out, a_out, av_hidden_feat, v_hidden_feat, a_hidden_feat, v_feat, a_feat)
    
    def gaussian_targets(self, size, center, std_dev):
        """
        Generate a Gaussian distribution centered at the middle index.
        
        Parameters:
        size (int): The size of the context window.
        center (int): The center index.
        std_dev (float): The standard deviation of the Gaussian distribution.
        
        Returns:
        torch.Tensor: The Gaussian distribution.
        """
        x = torch.arange(size).float()
        gaussian = torch.exp(-0.5 * ((x - center) ** 2) / (std_dev ** 2))
        gaussian = gaussian / gaussian.sum()  # Normalize to ensure it sums to 1
        return gaussian
    
    def sync_kl_loss(self, sync_output: Tensor, video_frames, std_dev=1.0, mode='av'):
        """
        Compute the AV synchronization KL divergence loss.
        
        Parameters:
        av_sync_output (torch.Tensor): The output from the AVSyncModel, shaped (batch_size, num_frames, 2*context_size+1).
        context_size (int): The size of the context window.
        std_dev (float): The standard deviation for the Gaussian distribution.
        
        Returns:
        torch.Tensor: The computed loss.
        """
        batch_size, num_frames, context_range = sync_output.shape
        assert context_range == 2 * self.context_size + 1, "Context range does not match the expected size."

        # Step 1: Compute mask without including the padding
        mask = torch.zeros_like(sync_output, dtype=torch.bool).to(sync_output.device).requires_grad_(False)
        actual_frame = torch.min(video_frames, torch.tensor(self.temporal_dim))

        mask[:, :self.context_size, :self.context_size] = self.top_mask
        
        for i in range(batch_size):
            mask[i, actual_frame[i]-self.context_size:actual_frame[i], -self.context_size:] = self.bottom_mask

        # Step 2: Maskify output and Compute softmax along the context dimension
        # log_softmax_output = F.log_softmax(sync_output, dim=-1)  # (batch_size, num_frames, context_range)
        add_val = torch.zeros_like(sync_output, dtype=torch.bool).to(sync_output.device).requires_grad_(False)
        add_val = add_val.masked_fill(mask, float('-inf'))
        softmax_output = F.softmax((sync_output + add_val), dim=-1) # (batch_size, num_frames, context_range)

        # Step 3: Generate Gaussian targets
        center = self.context_size
        if mode=='av':
            gaussian_target = self.gaussian_targets(context_range, center, std_dev).to(sync_output.device)  # (context_range)
        else:
            gaussian_target = torch.maximum(self.gaussian_targets(context_range, center-1, std_dev), self.gaussian_targets(context_range, center+1, std_dev)).to(sync_output.device)  # (context_range)
            gaussian_target = gaussian_target / gaussian_target.sum()
        
        # Step 3: Compute KL-Divergence/MSE
        # kl_div = F.kl_div(log_softmax_output, gaussian_target.expand(batch_size, num_frames, -1), reduction='none')  # (batch_size, num_frames, context_range)
        kl_div = (softmax_output - gaussian_target.expand(batch_size, num_frames, -1)) ** 2
        
        
        # Step 4: Include padding in mask
        for i in range(batch_size):
            mask[i, actual_frame[i]:] = True

        # Step 5: Compute the mean loss all frames and batch

        # mask = ~mask
        # softmax_output = softmax_output[mask]
        # gaussian_target = gaussian_target.expand(batch_size, num_frames, -1)[mask]
        # loss = F.mse_loss(softmax_output, gaussian_target)
        kl_div[mask] = 0
        mask = ~mask
        loss = kl_div.sum()/mask.sum()

        return loss
    
    def sync_single_loss(self, sync_output, padding_mask, mode='av'):
        """
        Compute the AV synchronization loss.
        
        Parameters:
        av_sync_output (torch.Tensor): The output from the AVSyncModel, shaped (batch_size, num_frames, 2*context_size+1).
        context_size (int): The size of the context window.
        
        Returns:
        torch.Tensor: The computed loss.
        """
        _, _, context_range = sync_output.shape
        assert context_range == 2 * self.context_size + 1, "Context range does not match the expected size."

        # Step 1: Compute softmax along the context dimension
        softmax_output = F.softmax(sync_output, dim=-1)  # (batch_size, num_frames, context_range)

        # Step 2: Extract the middle value
        middle_index = self.context_size

        if mode == 'av':
            middle_values = softmax_output[:, :, middle_index]  # (batch_size, num_frames)
        else:
            middle_values = softmax_output[:, :, middle_index-1] * softmax_output[:, :, middle_index+1]  # (batch_size, num_frames)

        # Step 3: Compute the negative log of the middle values
        log_middle_values = torch.log(middle_values)  # (batch_size, num_frames)

        log_middle_values[padding_mask] = float('nan')

        loss = -log_middle_values.nanmean()  # Average over all frames and batch

        return loss

    def loss_fn(self, cross_modal_out: Tensor, v_out: Tensor, a_out: Tensor, padding_mask: Tensor, video_frames):
        
        # In case of only maximizing single value
        if self.sync_loss == 'single':
            av_loss = self.sync_single_loss(cross_modal_out, padding_mask, 'av')
            v_loss = self.sync_single_loss(v_out, padding_mask, 'v')
            a_loss = self.sync_single_loss(a_out, padding_mask, 'a')
        
        # In case of fitting to a gaussian curve
        else:
            av_loss = self.sync_kl_loss(cross_modal_out, video_frames, 1.5, 'av')
            v_loss = self.sync_kl_loss(v_out, video_frames, 1.5, 'v')
            a_loss = self.sync_kl_loss(a_out, video_frames, 1.5, 'a')
                
        loss = av_loss + v_loss + a_loss
        loss_dict = {"loss": loss, "av_loss": av_loss, "v_loss": v_loss, "a_loss": a_loss}

        return {k: v for k, v in loss_dict.items() if v is not None}

    def step(self, batch: Sequence[Tensor]) -> Dict[str, Tensor]:

        (cross_modal_out, v_out, a_out, _, _, _, _, _) = self.forward(batch['video'], batch['audio'], batch['padding_mask'])

        loss_dict = self.loss_fn(cross_modal_out, v_out, a_out, batch['padding_mask'], batch['video_frames'])

        del batch
        
        return loss_dict
    
    
    def training_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None
    ) -> Tensor:
        loss_dict = self.step(batch)
        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=True, sync_dist=self.distributed)
        
        return loss_dict["loss"]
    
    def validation_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None
    ) -> Tensor:
        loss_dict = self.step(batch)

        self.log_dict({f"val_{k}": v for k, v in loss_dict.items()}, on_step=False, on_epoch=True,
            prog_bar=False, sync_dist=self.distributed)
        return loss_dict["loss"]

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Tuple [Tensor, Tensor, Tensor, Tensor, Tensor]:
        
        (cross_modal_out, v_out, a_out, av_hidden_feat, v_hidden_feat, a_hidden_feat, v_feat, a_feat) = self.forward(batch['video'], batch['audio'], batch['padding_mask'])
        return cross_modal_out, v_out, a_out, av_hidden_feat, v_hidden_feat, a_hidden_feat, v_feat, a_feat

        # video = batch['video']
        # audio = batch['audio']
        # padding_mask = batch['padding_mask']

        # batch_size, _, num_frames, _, _ = video.shape

        # tgt_mask = self.tgt_mask[:num_frames, :num_frames].to(video.device)
        
        # v_feat = self.video_encoder(video) 
        # a_feat = self.audio_encoder(audio, tgt_mask)

        # del video, audio

        # v_feat = v_feat.permute(0, 2, 1)
        # a_feat = a_feat.permute(0, 2, 1)

        # v_feat+=self.visual_enocoding
        # a_feat+=self.audio_enocoding

        # _, av_hidden_feat = self.cross_modal_consistency(v_feat, a_feat, padding_mask, 0-self.context_size)

        # return av_hidden_feat, v_feat, a_feat
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True, min_lr=1e-8),
                "monitor": "val_loss"
            }
        }





