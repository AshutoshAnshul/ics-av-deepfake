# modified from https://github.com/ControlNet/LAV-DF/blob/master/model/batfd_plus.py


import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoderLayer
from pytorch_lightning import LightningModule
from typing import Optional, Union, Sequence, Tuple, Dict

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from metrics import AP
from utils import Conv1d
from torch.nn import BCEWithLogitsLoss
from regression_loss import MaskedBsnppLoss, MaskedFrameLoss
from model.boundary_module_plus import BoundaryModulePlus, ModalFeatureAttnBoundaryMapFusion

import pickle as pkl

from nms_utils.nms import batched_nms

class FrameLogisticRegression(nn.Module):
    """
    Frame classifier (FC_v and FC_a) for video feature (F_v) and audio feature (F_a).
    Input:
        F_v or F_a: (B, C_f, T)
    Output:
        Y^: (B, 1, T)
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.lr_layer = Conv1d(n_features, 1, kernel_size=1)

    def forward(self, features: Tensor) -> Tensor:
        return self.lr_layer(features)


class LocalizationModelBMNPlus(LightningModule):
    def __init__(self, d_model=256, nhead=4, dim_feedforward=512, dropout=0.1, K=3, linear_layer_out=256, num_head_layers=3, num_classes=1, weight_decay=0.0001, learning_rate=0.0001, distributed=False, loss_alpha=0.977, max_len=750, context_size=15, fps: float = 25, max_duration=100, exp_name_prefix: str = "lavdf_inference", include_encoders: bool = True):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.distributed = distributed

        self.context_size = context_size
        self.top_mask = torch.rot90(torch.triu(torch.ones(context_size, context_size, dtype=torch.bool)), k=1, dims=(0, 1)).requires_grad_(False)
        self.bottom_mask = torch.rot90(self.top_mask, k=2, dims=(0, 1)).requires_grad_(False)

        self.sigmoid_loss_alpha = loss_alpha

        self.max_len = max_len
        self.num_levels = K

        self.fps = fps
        self.max_duration = max_duration

        self.linear_av = nn.Linear(d_model, d_model) # convert this to nn.Linear(2*d_model, d_model) in case using sparse encoder variant but keep to nn.Linear(d_model, d_model) in case using dencoder variant
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_a = nn.Linear(d_model, d_model)
        self.exp_name_prefix = exp_name_prefix

        self.include_encoders = include_encoders
        
        # Stack K instances of TransformerDecoderLayer (av -> v)
        self.decoder_layers_1 = nn.ModuleList(
            [TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True) for _ in range(K)]
        )
        
        # Stack K instances of TransformerDecoderLayer (av -> a)
        self.decoder_layers_2 = nn.ModuleList(
            [TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True) for _ in range(K)]
        )

        if include_encoders:
            transformer_layer1 = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=dim_feedforward)
            self.video_encoder = nn.TransformerEncoder(transformer_layer1, num_layers=2)

            transformer_layer2 = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=dim_feedforward)
            self.audio_encoder = nn.TransformerEncoder(transformer_layer2, num_layers=2)
        else:
            transformer_layer1 = None
            transformer_layer2 = None
            self.video_encoder = None
            self.audio_encoder = None

        self.fusion_frame_level_classifier = FrameLogisticRegression(n_features=d_model)
        self.vid_frame_level_classifier = FrameLogisticRegression(n_features=d_model)
        self.aud_frame_level_classifier = FrameLogisticRegression(n_features=d_model)


        fusion_dim = d_model+1
        self.fusion_boundary_module = BoundaryModulePlus(n_feature_in=fusion_dim, temporal_dim=max_len, max_duration=max_duration)
        self.vid_boundary_module = BoundaryModulePlus(n_feature_in=fusion_dim, temporal_dim=max_len, max_duration=max_duration)
        self.aud_boundary_module = BoundaryModulePlus(n_feature_in=fusion_dim, temporal_dim=max_len, max_duration=max_duration)

        self.a_v_fusion_p = ModalFeatureAttnBoundaryMapFusion(fusion_dim, fusion_dim, max_duration)
        self.a_v_fusion_c = ModalFeatureAttnBoundaryMapFusion(fusion_dim, fusion_dim, max_duration)
        self.a_v_fusion_pc = ModalFeatureAttnBoundaryMapFusion(fusion_dim, fusion_dim, max_duration)

        self.frame_loss = MaskedFrameLoss(BCEWithLogitsLoss())
        self.bm_loss = MaskedBsnppLoss(0, 1)
        
        self.ln = nn.LayerNorm(d_model)
        self.val_epoch_predict_dict = {}
    
    def maskify(self, x: Tensor, video_frames):
        mask = torch.zeros_like(x, dtype=torch.bool).to(x.device).requires_grad_(False)
        actual_frame = torch.min(video_frames, torch.tensor(self.max_len))    

        mask[:, :self.context_size, :self.context_size] = self.top_mask
        
        for i in range(x.shape[0]):
            mask[i, actual_frame[i]-self.context_size:actual_frame[i], -self.context_size:] = self.bottom_mask
        
        add_val = torch.zeros_like(x, dtype=torch.bool).to(x.device).requires_grad_(False)
        add_val = add_val.masked_fill(mask, float('-inf'))
        softmax_output = F.softmax((x + add_val), dim=-1) # (batch_size, num_frames, context_range)
        return softmax_output
    
    def forward(self, av: Tensor, v: Tensor, a: Tensor, video_frames, padding_mask=None):
        batch_size = av.shape[0]

        # Use maskify only while using the actual 31 dimensional similarity values instead of features
        # av = self.maskify(av, video_frames)
        # v = self.maskify(v, video_frames)
        # a = self.maskify(a, video_frames)

        av = self.linear_av(av)
        v = self.linear_v(v)
        a = self.linear_a(a)

        if self.include_encoders:
            v = self.video_encoder(v, src_key_padding_mask=padding_mask)
            a = self.audio_encoder(a, src_key_padding_mask=padding_mask)

        # pass through decoder
        output = av
        for decoder_layer_1, decoder_layer_2 in zip(self.decoder_layers_1, self.decoder_layers_2):
            output = decoder_layer_1(output, v, tgt_key_padding_mask=padding_mask, memory_key_padding_mask=padding_mask)  # TransformerDecoderLayer 1 (av as tgt, v as src)
            output = decoder_layer_2(output, a, tgt_key_padding_mask=padding_mask, memory_key_padding_mask=padding_mask)  # TransformerDecoderLayer 2 (av as tgt, a as src)

        v= self.ln(v)
        a= self.ln(a)
        output= self.ln(output)
        # change [BxTxf] to [BxfxT]
        v = v.permute(0, 2, 1)
        a = a.permute(0, 2, 1)
        output = output.permute(0, 2, 1)

        # get frame level labels
        vid_frame_level_class = self.vid_frame_level_classifier(v)
        aud_frame_level_class = self.aud_frame_level_classifier(a)
        fusion_frame_level_class = self.fusion_frame_level_classifier(output)

        # add the labels to features
        vid_bm_in = torch.column_stack([v, vid_frame_level_class])
        aud_bm_in = torch.column_stack([a, aud_frame_level_class])
        fusion_bm_in = torch.column_stack([output, fusion_frame_level_class])

        # pass through boundary module
        vid_map_p, vid_map_c, vid_map_pc = self.vid_boundary_module(vid_bm_in)
        aud_map_p, aud_map_c, aud_map_pc = self.aud_boundary_module(aud_bm_in)
        fusion_map_p, fusion_map_c, fusion_map_pc = self.fusion_boundary_module(fusion_bm_in)

        if torch.isnan(vid_map_p).any():
            print('nan in vid p')
        if torch.isnan(vid_map_c).any():
            print('nan in vid c')
        if torch.isnan(vid_map_pc).any():
            print('nan in vid pc')
        
        if torch.isnan(aud_map_p).any():
            print('nan in aud p')
        if torch.isnan(aud_map_c).any():
            print('nan in aud c')
        if torch.isnan(aud_map_pc).any():
            print('nan in aud pc')
        
        if torch.isnan(fusion_map_p).any():
            print('nan in fusion p')
        if torch.isnan(fusion_map_c).any():
            print('nan in fusion c')
        if torch.isnan(fusion_map_pc).any():
            print('nan in fusion pc')

        # fusion audio and visual features
        fusion_map_p = self.a_v_fusion_p(vid_bm_in, aud_bm_in, fusion_bm_in, vid_map_p, aud_map_p, fusion_map_p)
        fusion_map_c = self.a_v_fusion_c(vid_bm_in, aud_bm_in, fusion_bm_in, vid_map_c, aud_map_c, fusion_map_c)
        fusion_map_pc = self.a_v_fusion_pc(vid_bm_in, aud_bm_in, fusion_bm_in, vid_map_pc, aud_map_pc, fusion_map_pc)

        return fusion_map_p, fusion_map_c, fusion_map_pc, fusion_frame_level_class, vid_map_p, vid_map_c, vid_map_pc, vid_frame_level_class, aud_map_p, aud_map_c, aud_map_pc, aud_frame_level_class

    def loss_fn(self, fusion_map_p: Tensor, fusion_map_c: Tensor, fusion_map_pc: Tensor, fusion_label_map_iou: Tensor,
                video_map_p: Tensor, video_map_c: Tensor, video_map_pc: Tensor, video_label_map_iou: Tensor,
                audio_map_p: Tensor, audio_map_c: Tensor, audio_map_pc: Tensor, audio_label_map_iou: Tensor, 
                fusion_frame_level_class:Tensor, fusion_frame_level_label: Tensor, 
                video_frame_level_class:Tensor, video_frame_level_label: Tensor, 
                audio_frame_level_class:Tensor, audio_frame_level_label: Tensor, video_frames: Tensor):
        
        fusion_bm_loss, *_ = self.bm_loss(fusion_map_p, fusion_map_c, fusion_map_pc, None, None, None, None, fusion_label_map_iou, None, None, video_frames)
        video_bm_loss, *_ = self.bm_loss(video_map_p, video_map_c, video_map_pc, None, None, None, None, video_label_map_iou, None, None, video_frames)
        audio_bm_loss, *_ = self.bm_loss(audio_map_p, audio_map_c, audio_map_pc, None, None, None, None, audio_label_map_iou, None, None, video_frames)

        fusion_frame_loss = self.frame_loss(fusion_frame_level_class.squeeze(1), fusion_frame_level_label, video_frames)
        video_frame_loss = self.frame_loss(video_frame_level_class.squeeze(1), video_frame_level_label, video_frames)
        audio_frame_loss = self.frame_loss(audio_frame_level_class.squeeze(1), audio_frame_level_label, video_frames)

        loss = fusion_bm_loss + fusion_frame_loss + (video_bm_loss + audio_bm_loss)/2 + (video_frame_loss + audio_frame_loss)/2
        loss_dict = {"loss": loss, "fusion_bm_loss": fusion_bm_loss, "fusion_frame_loss": fusion_frame_loss, 
                     "video_bm_loss": video_bm_loss, "video_frame_loss": video_frame_loss,
                     "audio_bm_loss": audio_bm_loss, "audio_frame_loss": audio_frame_loss}

        return {k: v for k, v in loss_dict.items() if v is not None}

    def step(self, batch: Sequence[Tensor]) -> Dict[str, Tensor]:

        fusion_map_p, fusion_map_c, fusion_map_pc, fusion_frame_level_class, vid_map_p, vid_map_c, vid_map_pc, vid_frame_level_class, aud_map_p, aud_map_c, aud_map_pc, aud_frame_level_class = self.forward(batch['av_sync'], batch['v_sync'], batch['a_sync'], batch['video_frames'], batch['padding_mask'])

        # fusion_map_* = [B, max_duration, max_len]
        # frame_level_class = [B, 1, max_len]
        
        loss_dict = self.loss_fn(
            fusion_map_p, fusion_map_c, fusion_map_pc, batch['fusion_gt_iou_map'],
            vid_map_p, vid_map_c, vid_map_pc, batch['vid_gt_iou_map'],
            aud_map_p, aud_map_c, aud_map_pc, batch['aud_gt_iou_map'],
            fusion_frame_level_class, batch['fusion_frame_label'],
            vid_frame_level_class, batch['vid_frame_label'],
            aud_frame_level_class, batch['aud_frame_label'], batch['video_frames']
            )

        del batch
        
        return loss_dict

    def training_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None
    ) -> Tensor:
        loss_dict = self.step(batch)
        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=True, sync_dist=self.distributed)
        
        return loss_dict["loss"]
    
    def inference_single_video(self, bm_map: Tensor, n_frames:int):

        # Get the indices (start) and the corresponding values (score)
        duration, begin = torch.nonzero(bm_map, as_tuple=True)
        score = bm_map[duration, begin]

        # Calculate the 'end' values
        end = duration + begin

        valid = (duration > 0) & (end <= n_frames)

        # Apply the filter
        duration = duration[valid]
        begin = begin[valid]
        score = score[valid]
        end = end[valid]

        # # Sort by 'begin' and then 'end'
        # sorted_indices = torch.lexsort((end, begin))
        # begin = begin[sorted_indices]
        # end = end[sorted_indices]
        # score = score[sorted_indices]

        segments = torch.stack([begin, end], dim=1)

        if segments.shape[0] > 0:
            segments = segments.float()/float(self.fps)
            segments = segments.clamp(min=0, max=float(n_frames)/float(self.fps))

        segments, score = batched_nms(segments.type(torch.float32), score.type(torch.float32), iou_threshold=0.1, min_score=0.001, max_seg_num=100, sigma=0.75, voting_thresh=0.9)

        if len(score.shape)==1:
            score = score.unsqueeze(-1)

        return torch.cat([score, segments], dim=1)



    def validation_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None
    ) -> Tensor:
        loss_dict = self.step(batch)
        self.log_dict({f"val_{k}": v for k, v in loss_dict.items()}, on_step=False, on_epoch=True,
            prog_bar=False, sync_dist=self.distributed)
        return loss_dict["loss"]

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Tuple [Tensor, Tensor, Tensor, Tensor, Tensor]:
        
        fusion_map_p, fusion_map_c, fusion_map_pc, *_ = self.forward(batch['av_sync'], batch['v_sync'], batch['a_sync'], batch['video_frames'], batch['padding_mask'])

        fusion_map = (fusion_map_p + fusion_map_c + fusion_map_pc)/3

        # print(fusion_map.shape)

        for i in range(batch['padding_mask'].shape[0]):
            bm_map = fusion_map[i]
            n_frames = batch['video_frames'][i]

            segments = self.inference_single_video(bm_map, n_frames)

            filename = str(str(batch['filepath'][i]).split("/")[-1]).split('_')[-1]
            filename = str(filename).split('.')[0] + '.mp4'
            self.val_epoch_predict_dict[batch['filepath'][i]] = {
                'proposals' : segments,
                'labels' : batch['segments'][i]
            }
    
    def on_predict_epoch_end(self) -> None:
        with open(f"{self.exp_name_prefix}results.pkl", 'wb') as pkl_file:
            pkl.dump(self.val_epoch_predict_dict, pkl_file)

        ap_scores = AP(iou_thresholds=[0.5, 0.75, 0.9, 0.95], device=self.device)(self.val_epoch_predict_dict)

        self.val_epoch_predict_dict.clear()

        log_dict = {}
        AP_sum = 0
        for iou_threshold in [0.5, 0.75, 0.9, 0.95]:
            log_dict[f'val_AP@{iou_threshold}'] = ap_scores[iou_threshold]
            AP_sum += ap_scores[iou_threshold]
        
        log_dict['val_mAP'] = float(AP_sum)/float(4)

        print(log_dict)
        return log_dict
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": CosineAnnealingLR(optimizer=optimizer, T_max=1000, eta_min=0.00001),
                "monitor": "val_loss"
            }
        }