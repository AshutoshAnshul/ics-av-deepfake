# taken and modified from https://github.com/ControlNet/LAV-DF/blob/master/utils.py

from typing import Any, List, Tuple, Optional
from torch.nn import functional as F, Module
from einops import rearrange
from torch import Tensor
from abc import ABC

import numpy as np
import torchvision
import torchaudio
import random
import torch
import json
import os
import gc

from pytorch_lightning import Callback, Trainer, LightningModule
import re

def reproducibility(random_seed):                                  
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    cudnn_deterministic = True
    cudnn_benchmark = False
    print("cudnn_deterministic set to False")
    print("cudnn_benchmark set to True")
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
    return

def read_json(path: str, object_hook=None):
    with open(path, 'r') as f:
        return json.load(f, object_hook=object_hook)


def read_video(path: str):
    video, audio, info = torchvision.io.read_video(path, pts_unit="sec")
    video = video.permute(0, 3, 1, 2) / 255
    audio = audio.permute(1, 0)
    return video, audio, info


def read_audio(path: str):
    return torchaudio.load(path)


def read_image(path: str):
    return torchvision.io.read_image(path).float() / 255.0


def padding_video(tensor: Tensor, target: int, padding_method: str = "zero", padding_position: str = "tail") -> Tensor:
    t, c, h, w = tensor.shape
    padding_size = target - t

    pad = _get_padding_pair(padding_size, padding_position)

    if padding_method == "zero":
        return F.pad(tensor, pad=[0, 0, 0, 0, 0, 0] + pad)
    elif padding_method == "same":
        tensor = rearrange(tensor, "t c h w -> c h w t")
        tensor = F.pad(tensor, pad=pad + [0, 0], mode="replicate")
        return rearrange(tensor, "c h w t -> t c h w")
    else:
        raise ValueError("Wrong padding method. It should be zero or tail or average.")


def padding_audio(tensor: Tensor, target: int,
    padding_method: str = "zero",
    padding_position: str = "tail"
) -> Tensor:
    t, c = tensor.shape
    padding_size = target - t
    pad = _get_padding_pair(padding_size, padding_position)

    if padding_method == "zero":
        return F.pad(tensor, pad=[0, 0] + pad)
    elif padding_method == "same":
        tensor = rearrange(tensor, "t c -> 1 c t")
        tensor = F.pad(tensor, pad=pad, mode="replicate")
        return rearrange(tensor, "1 c t -> t c")
    else:
        raise ValueError("Wrong padding method. It should be zero or tail or average.")


def _get_padding_pair(padding_size: int, padding_position: str) -> List[int]:
    if padding_position == "tail":
        pad = [0, padding_size]
    elif padding_position == "head":
        pad = [padding_size, 0]
    elif padding_position == "average":
        padding_head = padding_size // 2
        padding_tail = padding_size - padding_head
        pad = [padding_head, padding_tail]
    else:
        raise ValueError("Wrong padding position. It should be zero or tail or average.")
    return pad


def resize_video(tensor: Tensor, size: Tuple[int, int], resize_method: str = "bicubic") -> Tensor:
    return F.interpolate(tensor, size=size, mode=resize_method)

def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute jaccard score between a box and the anchors."""

    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors - inter_len + box_max - box_min
    iou = inter_len / union_len
    return iou


def ioa_with_anchors(anchors_min, anchors_max, box_min, box_max):
    # calculate the overlap proportion between the anchor and all bbox for supervise signal,
    # the length of the anchor is 0.01
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    scores = np.divide(inter_len, len_anchors)
    return scores

def iou_1d(proposal, target) -> Tensor:
    """
    Calculate 1D IOU for N proposals with L labels.

    Args:
        proposal (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The predicted array with [M, 2]. First column is
            beginning, second column is end.
        target (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The label array with [N, 2]. First column is
            beginning, second column is end.

    Returns:
        :class:`~torch.Tensor`: The iou result with [M, N].
    """
    if type(proposal) is np.ndarray:
        proposal = torch.from_numpy(proposal)

    if type(target) is np.ndarray:
        target = torch.from_numpy(target)

    proposal_begin = proposal[:, 0].unsqueeze(0).T
    proposal_end = proposal[:, 1].unsqueeze(0).T
    target_begin = target[:, 0]
    target_end = target[:, 1]

    inner_begin = torch.maximum(proposal_begin, target_begin)
    inner_end = torch.minimum(proposal_end, target_end)
    outer_begin = torch.minimum(proposal_begin, target_begin)
    outer_end = torch.maximum(proposal_end, target_end)

    inter = torch.clamp(inner_end - inner_begin, min=0.)
    union = outer_end - outer_begin
    return inter / union


class _ConvNd(Module, ABC):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0,
        build_activation: Optional[callable] = None
    ):
        super().__init__()
        self.conv = self.PtConv(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        if build_activation is not None:
            self.activation = build_activation()
        else:
            self.activation = None

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Conv1d(_ConvNd):
    PtConv = torch.nn.Conv1d


class Conv2d(_ConvNd):
    PtConv = torch.nn.Conv2d


class Conv3d(_ConvNd):
    PtConv = torch.nn.Conv3d


def create_segments(encoding: Tensor, num_frames, segment_size=5):
    encoding = encoding.clone().detach()
    padding = segment_size // 2
    padded_encoding = F.pad(encoding, (0, 0, padding, padding), mode='constant', value=0)
    segments = []
    for i in range(num_frames):
        segment = padded_encoding[:, i:i+segment_size, :]
        segments.append(segment)
        
    segments = torch.stack(segments, dim=1)
    return segments

def create_pairs(video_segments: Tensor, audio_segments: Tensor, context_size=15):
    video_segments = video_segments.clone().detach()
    audio_segments = audio_segments.clone().detach()
    num_frames = video_segments.shape[1] 
    context_range = 2 * context_size + 1

    # Initialize pairs with zeros
    pairs = torch.zeros(video_segments.shape[0], num_frames, context_range, video_segments.shape[2], video_segments.shape[3] * 2)

    padded_audio_segments = F.pad(audio_segments, (0, 0, 0, 0, context_size, context_size), mode='constant', value=0)
    
    for i in range(num_frames):
        audio_context = padded_audio_segments[:, i:i + context_range, :, :]
        video_segment = video_segments[:, i, :, :].unsqueeze(1).repeat(1, context_range, 1, 1)
#         print(video_segment.shape, audio_context.shape)
        pair = torch.cat((video_segment, audio_context), dim=-1)
        pairs[:, i, :, :, :] = pair

    return pairs

## IO

RESULT_DIR = "%s_%s_RESULT"
CONFIG_NAME = "config.json"
RESULT_NAME = "result.json"

def dump_everything(output_path: str, operation: str, config: dict[str, Any] = {}, results: dict[str, Any] = {}):
    if len(config) > 1:
        bm = config["base_model_name_or_path"].replace("/","_")
        pm = config.get("peft_model_name_or_path", None)
        model_name = f"{bm}_tuned" if pm else bm
        dir = os.path.join(output_path, RESULT_DIR % (model_name, operation))
        os.makedirs(dir, exist_ok=True)
        with open(os.path.join(dir, CONFIG_NAME), 'w') as config_file:
            config_file.write(json.dumps(config, indent=4))
        with open(os.path.join(dir, RESULT_NAME), 'w') as result_file:
            result_file.write(json.dumps(results, indent=4))
    elif len(results) > 1:
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, RESULT_NAME)) as result_file:
            result_file.write(json.dumps(results, indent=4))

def continuous_2_col_zip_csv_dumper(data: zip, file_path: str):
    with open(file_path, 'a') as dump_file:
        for row1, row2 in data:
            dump_file.writelines([f"{row1}, {row2}\n"])
    del data
    gc.collect()
    
# Checkpoints

## Constants 
CHKPS_PATH = "checkpoints"
CHKP = "CHECKPOINT-%d"

def dump_checkpoint(experiment_path: str, checkpoint: int, model: Any, **kwargs) -> str:
    chkp_path = os.path.join(experiment_path, CHKPS_PATH)
    save_path = os.path.join(chkp_path, CHKP % checkpoint)
    os.makedirs(save_path, exist_ok=True)
    if kwargs.get('log', False):
        print(kwargs.get('message', f'Saving at step [{checkpoint}]'))
    model.save_pretrained(
        save_path, state_dict=None, safe_serialization=True
    )
    del kwargs, model
    gc.collect()
    return (CHKP % checkpoint, save_path)

CHKP_RESULT_FILE = "%s_eval_results.json"

def dump_checkpoint_result(chkp: str, chkp_path: str, result: dict[str, Any], **kwargs):
    file_path = os.path.join(chkp_path, CHKP_RESULT_FILE % chkp)
    os.makedirs(chkp_path, exist_ok=True)
    if kwargs.get('log', False):
        print(kwargs.get('message', f'Saving results for checkpoint [{chkp}] at [{file_path}]'))
    with open(file_path, 'w') as result_file:
        result_file.write(json.dumps(result))
    del kwargs, result, chkp, chkp_path
    gc.collect()



class LrLogger(Callback):
    """Log learning rate in each epoch start."""

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        for i, optimizer in enumerate(trainer.optimizers):
            for j, params in enumerate(optimizer.param_groups):
                key = f"opt{i}_lr{j}"
                value = params["lr"]
                pl_module.logger.log_metrics({key: value}, step=trainer.global_step)
                pl_module.log(key, value, logger=False, sync_dist=pl_module.distributed)


class EarlyStoppingLR(Callback):
    """Early stop model training when the LR is lower than threshold."""

    def __init__(self, lr_threshold: float, mode="all"):
        self.lr_threshold = lr_threshold

        if mode in ("any", "all"):
            self.mode = mode
        else:
            raise ValueError(f"mode must be one of ('any', 'all')")

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._run_early_stop_checking(trainer)

    def _run_early_stop_checking(self, trainer: Trainer) -> None:
        metrics = trainer._logger_connector.callback_metrics
        if len(metrics) == 0:
            return
        all_lr = []
        for key, value in metrics.items():
            if re.match(r"opt\d+_lr\d+", key):
                all_lr.append(value)

        if len(all_lr) == 0:
            return

        if self.mode == "all":
            if all(lr <= self.lr_threshold for lr in all_lr):
                trainer.should_stop = True
                print(f"Early stopping due to LR [{all_lr[-1]}] / Threshold: [{self.lr_threshold}]")
        elif self.mode == "any":
            if any(lr <= self.lr_threshold for lr in all_lr):
                trainer.should_stop = True
                print(f"Early stopping due to LR [{all_lr[-1]}] / Threshold: [{self.lr_threshold}]")
