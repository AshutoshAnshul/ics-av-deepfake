import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoderLayer
from pytorch_lightning import LightningModule
from typing import Optional, Union, Sequence, Tuple, Dict

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from metrics import AP, AR

import math

from nms_utils.nms import batched_nms

class ClassificationModel(LightningModule):
    def __init__(self, d_model=256, nhead=4, dim_feedforward=512, dropout=0.1, K=3, linear_layer_out=256, num_encoder_layer=3, num_classes=1, weight_decay=0.0001, learning_rate=0.0001, distributed=False, loss_alpha=0.977, max_len=500, context_size=15):
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

        self.linear_av = nn.Linear(d_model, d_model) # convert this to nn.Linear(2*d_model, d_model) in case using sparse encoder variant but keep to nn.Linear(d_model, d_model) in case using dencoder variant
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_a = nn.Linear(d_model, d_model)
        
        # Stack K instances of TransformerDecoderLayer (av -> v)
        self.decoder_layers_1 = nn.ModuleList(
            [TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True) for _ in range(K)]
        )
        
        # Stack K instances of TransformerDecoderLayer (av -> a)
        self.decoder_layers_2 = nn.ModuleList(
            [TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True) for _ in range(K)]
        )

        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, linear_layer_out) for _ in range(K)]
        )
        
        self.linear_transform = nn.Linear(self.max_len, self.max_len)

        # 1D Convolutional layers with stride
        self.conv1 = nn.Conv1d(K*linear_layer_out, 2*linear_layer_out, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(2*linear_layer_out, linear_layer_out, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(linear_layer_out, linear_layer_out//2, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(linear_layer_out//2, linear_layer_out//4, kernel_size=3, stride=2, padding=1)
        
        # Calculate the output size of the last conv layer
        conv_output_size = self.max_len // 16  # Because we have 4 layers with stride 2
        if self.max_len%16!=0:
            conv_output_size+=1
        linear_size = linear_layer_out//4
        
        # Fully connected layers
        self.fc1 = nn.Linear(linear_size * conv_output_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Activation
        self.relu = nn.ReLU()

        # self.maxpool = nn.MaxPool1d(4, stride=4) # uncomment this only when trying on kodf
    
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

        # uncomment the following 4 lines only when trying on kodf
        # av = self.maxpool(av.permute(0,2,1)).permute(0,2,1)
        # v = self.maxpool(v.permute(0,2,1)).permute(0,2,1)
        # a = self.maxpool(a.permute(0,2,1)).permute(0,2,1)
        # padding_mask = padding_mask[:, ::4]

        # print(av.shape)
        av = self.linear_av(av)
        v = self.linear_v(v)
        a = self.linear_a(a)

        output = av

        
        x = []
        for decoder_layer_1, decoder_layer_2, linear_layer in zip(self.decoder_layers_1, self.decoder_layers_2, self.linear_layers):
            output = decoder_layer_1(output, v, tgt_key_padding_mask=padding_mask, memory_key_padding_mask=padding_mask)  # TransformerDecoderLayer 1 (av as tgt, v as src)
            output = decoder_layer_2(output, a, tgt_key_padding_mask=padding_mask, memory_key_padding_mask=padding_mask)  # TransformerDecoderLayer 2 (av as tgt, a as src)

            x.append(linear_layer(output))
        
        
        x = torch.cat(x, dim=-1)

        x = x*(~padding_mask.unsqueeze(-1))
        x = x.permute(0, 2, 1)
        x = self.linear_transform(x)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def loss_fn(self, pred: Tensor, labels: Tensor):
        
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, labels.unsqueeze(1).float())
        
        loss_dict = {"loss": loss}

        return {k: v for k, v in loss_dict.items() if v is not None}

    def step(self, batch: Sequence[Tensor]) -> Dict[str, Tensor]:

        pred = self.forward(batch['av_sync'], batch['v_sync'], batch['a_sync'], batch['video_frames'], batch['padding_mask'])

        loss_dict = self.loss_fn(pred, batch['label'])

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
        
        pred = self.forward(batch['av_sync'], batch['v_sync'], batch['a_sync'], batch['video_frames'], batch['padding_mask'])

        return pred
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": CosineAnnealingLR(optimizer=optimizer, T_max=1000, eta_min=0.00001),
                "monitor": "val_loss"
            }
        }
