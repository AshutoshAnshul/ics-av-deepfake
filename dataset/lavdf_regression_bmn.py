# modified from https://github.com/ControlNet/LAV-DF/blob/master/dataset/lavdf.py

import os
from dataclasses import dataclass
from typing import Optional, List, Callable, Any, Tuple, Union

import torch
import pickle
import torchaudio
import numpy as np
from einops import rearrange
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch import Tensor
from torch.nn import functional as F, Identity
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import Dataset
from dataset.collater import BatchCollaterRegressionBMNPTLightning

from utils import read_json, iou_with_anchors

@dataclass
class Metadata:
    file: str
    n_fakes: int
    fake_periods: List[List[float]]
    duration: float
    original: Optional[str]
    modify_video: bool
    modify_audio: bool
    split: str
    video_frames: int
    audio_channels: int
    audio_frames: int
    timestamps: List[List[str | int]]
    transcript: str

T_LABEL = Union[Tensor, Tuple[Tensor, Tensor, Tensor]]

class LavdfBmn(Dataset):

    def __init__(self, subset: str, root: str = "data", frame_padding: int = 750, fps: int = 25, img_size = 96, take_factor = 4, num_dists = 32,
        video_transform: Callable[[Tensor], Tensor] = Identity(), audio_transform: Callable[[Tensor], Tensor] = Identity(), sample: bool = True,
        pad_to_max_len: bool = True, metadata = None, max_duration = 40, leave_section = 'None'
    ):
        self.subset = subset
        self.root = root
        self.video_padding = frame_padding
        self.audio_padding = int(frame_padding / fps * 16000)
        self.video_transform = video_transform
        self.audio_transform = audio_transform
        self.img_size = img_size
        self.fps = fps
        self.pad_to_max_len = pad_to_max_len
        self.max_duration = max_duration


        # label_dir = os.path.join(self.root, "label")
        # if not os.path.exists(label_dir):
        #     os.mkdir(label_dir)

        self.metadata: List[Metadata] = []

        if metadata is None:
            metadata_mid: List[Metadata] = read_json(os.path.join(self.root, "metadata.json"), lambda x: Metadata(**x))
            metadata: List[Metadata] = [each for each in metadata_mid if each.split == subset]

        # if subset == "train":
        #     self.metadata = self.metadata[:20000]
        if leave_section is None:
            self.metadata = metadata
            print('got here', len(metadata))
        elif leave_section.strip().lower() == 'none':
            self.metadata = metadata
            print('got-here', len(metadata))
        elif 'test' in subset:
            for meta in metadata:
                if meta.modify_audio==False and meta.modify_video==False:
                    self.metadata.append(meta)
                elif leave_section=='RVFA' and meta.modify_audio==True and meta.modify_video==False:
                    self.metadata.append(meta)
                elif leave_section=='FVRA' and meta.modify_audio==False and meta.modify_video==True:
                    self.metadata.append(meta)
                elif leave_section=='FVFA' and meta.modify_audio==True and meta.modify_video==True:
                    self.metadata.append(meta)
        else:
            for meta in metadata:
                if meta.modify_audio==False and meta.modify_video==False:
                    self.metadata.append(meta)
                elif leave_section=='RVFA' and not (meta.modify_audio==True and meta.modify_video==False):
                    self.metadata.append(meta)
                elif leave_section=='FVRA' and not (meta.modify_audio==False and meta.modify_video==True):
                    self.metadata.append(meta)
                elif leave_section=='FVFA' and not (meta.modify_audio==True and meta.modify_video==True):
                    self.metadata.append(meta)
        
        dropped_number = len(metadata)-len(self.metadata)
        print(f"Load {len(self.metadata)} data in {subset}. Dropped number {dropped_number}")
    
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int) -> List[Tensor]:
        meta = self.metadata[index]

        feature_file = '_'.join(str(meta.file).strip().split('/'))
        feature_file_path = os.path.join(
            self.root, 'decoder_voxceleb_feat', str(feature_file)
        ).replace('.mp4', '.pkl')

        with open(feature_file_path, 'rb') as feature_file:
            video_data = pickle.load(feature_file)

        av_sync = torch.tensor(video_data['av_hidden_feat'])
        v_sync = torch.tensor(video_data['v_feat'])
        a_sync = torch.tensor(video_data['a_feat'])
        video_frames = video_data['video_frames']

        frame_padding = self.video_padding - video_frames
        pad_val = [0, 0, 0, frame_padding]
        av_sync = torch.nn.functional.pad(av_sync, pad_val)
        v_sync = torch.nn.functional.pad(v_sync, pad_val)
        a_sync = torch.nn.functional.pad(a_sync, pad_val)

        fusion_gt_iou_map = self.get_label(meta)
        gt_iou_map_real = torch.zeros_like(fusion_gt_iou_map)
        vid_gt_iou_map = fusion_gt_iou_map if meta.modify_video else gt_iou_map_real
        aud_gt_iou_map = fusion_gt_iou_map if meta.modify_audio else gt_iou_map_real
        # print(fusion_gt_iou_map.shape)
        # print('got map')

        #generate frame level labels
        fusion_frame_label = torch.zeros(self.video_padding)

        for begin, end in meta.fake_periods:
            begin = int(begin*self.fps)
            end = int(end*self.fps)
            fusion_frame_label[begin: end] = 1
        
        vid_frame_label = fusion_frame_label if meta.modify_video else torch.zeros(self.video_padding)
        aud_frame_label = fusion_frame_label if meta.modify_audio else torch.zeros(self.video_padding)

        outputs = {
            'filepath': video_data['filepath'],
            'av_sync': av_sync,
            'v_sync': v_sync,
            'a_sync': a_sync,
            'video_frames': video_frames,
            'segments': meta.fake_periods,
            'fusion_gt_iou_map': fusion_gt_iou_map,
            'fusion_frame_label': fusion_frame_label,
            'vid_gt_iou_map': vid_gt_iou_map,
            'vid_frame_label': vid_frame_label,
            'aud_gt_iou_map': aud_gt_iou_map,
            'aud_frame_label': aud_frame_label,
        }
        return outputs
    
    def get_label(self, meta: Metadata) -> Tensor:
        label_file = self.subset + '_' + '_'.join(str(meta.file).strip().split('/'))
        label_file_path = os.path.join(self.root, "label", str(label_file).split('.')[0]) + '.npy'
        if os.path.exists(label_file_path):
            try:
                arr = np.load(label_file_path)
            except ValueError:
                pass
            else:
                return torch.tensor(arr['fusion_map'])

        fusion_gt_iou_map = self._get_train_label(meta.video_frames, meta.fake_periods, meta.video_frames)
        # vid_gt_iou_map = self._get_train_label(meta.video_frames, meta.visual_fake_segments, meta.video_frames)
        # aud_gt_iou_map = self._get_train_label(meta.video_frames, meta.audio_fake_segments, meta.video_frames)
        # cache label
        np.savez(label_file_path, 
                 fusion_map = fusion_gt_iou_map.numpy())
        
        return fusion_gt_iou_map

    def _get_train_label(self, frames, video_labels, temporal_scale, fps=25) -> T_LABEL:
        corrected_second = frames / fps
        temporal_gap = 1 / temporal_scale

        ##############################################################################################
        # change the measurement from second to percentage
        gt_bbox = []
        for j in range(len(video_labels)):
            tmp_start = max(min(1, video_labels[j][0] / corrected_second), 0)
            tmp_end = max(min(1, video_labels[j][1] / corrected_second), 0)
            gt_bbox.append([tmp_start, tmp_end])

        ####################################################################################################
        # generate R_s and R_e
        gt_bbox = torch.tensor(gt_bbox)
        if len(gt_bbox) > 0:
            gt_xmins = gt_bbox[:, 0]
            gt_xmaxs = gt_bbox[:, 1]
        else:
            gt_xmins = np.array([])
            gt_xmaxs = np.array([])
        #####################################################################################################
        gt_iou_map = torch.zeros([self.max_duration, temporal_scale])
        if len(gt_bbox) > 0:
            for begin in range(temporal_scale):
                for duration in range(self.max_duration):
                    end = begin + duration
                    if end > temporal_scale:
                        break
                    gt_iou_map[duration, begin] = torch.max(
                        iou_with_anchors(begin * temporal_gap, (end + 1) * temporal_gap, gt_xmins, gt_xmaxs))
                    # [i, j]: Start in i, end in j.

        ##########################################################################################################
        gt_iou_map = F.pad(gt_iou_map.float(), pad=[0, self.video_padding - frames, 0, 0])

        return gt_iou_map    

class LavdfBmnDataModule(LightningDataModule):
    train_dataset: LavdfBmn
    dev_dataset: LavdfBmn
    test_dataset: LavdfBmn
    metadata: List[Metadata]

    def __init__(self, train_subset: str='train', val_subset: str='val', test_subset: str='test', root: str = "data", frame_padding: int = 750, fps: int = 25, img_size = 96, val_take_factor = 2, num_dists = 32,
        video_transform: Callable[[Tensor], Tensor] = Identity(), audio_transform: Callable[[Tensor], Tensor] = Identity(), sample: bool = True, pad_to_max_len: bool = True, batch_size: int=4, num_workers: int=8,
        max_duration: int = 40, leave_section='None'
    ):
        super().__init__()
        self.root = root
        self.frame_padding = frame_padding
        self.fps = fps
        self.img_size = img_size
        self.video_transform = video_transform
        self.audio_transform = audio_transform
        self.sample = sample
        self.pad_to_max_len = pad_to_max_len
        self.train_subset = train_subset
        self.val_subset = val_subset
        self.test_subset = test_subset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collator = BatchCollaterRegressionBMNPTLightning(pad_to_max_len, frame_padding)
        self.max_duration = max_duration

        self.val_take_factor = val_take_factor
        self.num_dists = num_dists

        self.leave_section = leave_section

    def setup(self, stage: Optional[str] = None) -> None:
        self.val_dataset = LavdfBmn(subset=self.val_subset, root=self.root, frame_padding=self.frame_padding, fps=self.fps, img_size=self.img_size, video_transform=self.video_transform, audio_transform=self.audio_transform, sample=self.sample, pad_to_max_len=self.pad_to_max_len, take_factor=self.val_take_factor, num_dists=self.num_dists, max_duration=self.max_duration, leave_section=self.leave_section)

        self.train_dataset = LavdfBmn(subset=self.train_subset, root=self.root, frame_padding=self.frame_padding, fps=self.fps, img_size=self.img_size, video_transform=self.video_transform, audio_transform=self.audio_transform, sample=self.sample, pad_to_max_len=self.pad_to_max_len, take_factor=self.val_take_factor, num_dists=self.num_dists, max_duration=self.max_duration, leave_section=self.leave_section)

        self.test_dataset = LavdfBmn(subset=self.test_subset, root=self.root, frame_padding=self.frame_padding, fps=self.fps, img_size=self.img_size, video_transform=self.video_transform, audio_transform=self.audio_transform, sample=self.sample, pad_to_max_len=self.pad_to_max_len, take_factor=self.val_take_factor, num_dists=self.num_dists, max_duration=self.max_duration, leave_section=self.leave_section)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collator, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=3*self.batch_size, num_workers=self.num_workers, collate_fn=self.collator, shuffle=False)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=3*self.batch_size, num_workers=self.num_workers, collate_fn=self.collator, shuffle=False)


# The dictionary is used to map the feature type to the dataset type
# The key is a tuple of (visual_feature_type, audio_feature_type), ``None`` means using end-to-end encoder.
feature_type_to_dataset_type = {
    (None, None): LavdfBmn
}