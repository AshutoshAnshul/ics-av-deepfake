from pathlib import Path
from typing import Optional, List, Callable, Any, Union, Tuple
import os
import math
import random
import json
import torch
import torchaudio
from einops import rearrange
from torch import Tensor
from torch.nn import Identity
from torch.utils.data import Dataset, DataLoader, RandomSampler
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from utils import read_video, padding_video, padding_audio, resize_video
from dataset.collater import BatchCollaterInferencePTLightning


class DFDCInference(Dataset):
    def __init__(self, subset: str, root: str = "data", frame_padding: int = 512, fps: int = 25, img_size = 96, take_factor = 4, num_dists = 32,
        video_transform: Callable[[Tensor], Tensor] = Identity(), audio_transform: Callable[[Tensor], Tensor] = Identity(), sample: bool = True, pad_to_max_len: bool = True) :

        self.subset = subset
        self.root = root
        self.video_padding = frame_padding
        self.audio_padding = int(frame_padding / fps * 16000)
        self.video_transform = video_transform
        self.audio_transform = audio_transform
        self.img_size = img_size
        self.fps = fps
        self.pad_to_max_len = pad_to_max_len

        src_dir = Path(root)

        metadata_file = os.path.join(src_dir, "selected_metadata.json")

        assert os.path.isfile(metadata_file), f"No file name {metadata_file} found" 

        with open(metadata_file, 'r') as meta_file:
            self.metadata = json.load(meta_file)
        
        def filter_existing(meta):
            feature_file_path = os.path.join(self.root, 'decoder_voxceleb_feat', meta['file'])
            feature_file_path_final = '.'.join(feature_file_path.split('.')[:-1]) + '.pkl'
            return not os.path.isfile(feature_file_path_final)  

        self.metadata = list(filter(filter_existing, self.metadata))

        print(f"Load {len(self.metadata)} data in {subset}.")

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        
        meta = self.metadata[index]

        filename = meta['file']

        feature_file_path = os.path.join(self.root, 'decoder_voxceleb_feat', filename)
        
        filePath = os.path.join(self.root, str(meta['folder']).split('/')[-1], str(meta['file']).split('.')[0]) + '_merged.mp4'

        video, audio, _ = read_video(filePath)
        video_frames = min(self.video_padding, video.shape[0])

        video = padding_video(video, target=self.video_padding)
        audio = padding_audio(audio, target=self.audio_padding)
            
        video = self.video_transform(video)
        audio = self.audio_transform(audio)

        video = rearrange(resize_video(video, (self.img_size, self.img_size)), "t c h w -> c t h w")
        audio = self._get_log_mel_spectrogram(audio)

        outputs = {
            'video': video,
            'audio': audio,
            'video_frames': video_frames,
            'filePath': filePath,
            'feature_file_path': feature_file_path
        }
        return outputs

    @staticmethod
    def _get_log_mel_spectrogram(audio: Tensor) -> Tensor:
        ms = torchaudio.transforms.MelSpectrogram(n_fft=321, n_mels=64)
        spec = torch.log(ms(audio[:, 0]) + 0.01)
        # assert spec.shape == (64, 2048), "Wrong log mel-spectrogram setup in Dataset"
        return spec
    
class DFDCInferenceDataModule(LightningDataModule):
    # train_dataset = FakeAVCelebPretrain
    val_dataset = DFDCInference

    def __init__(self, train_subset: str='dev', val_subset: str='test', root: str = "data", frame_padding: int = 512, fps: int = 25, img_size = 96, val_take_factor = 2, num_dists = 32,
        video_transform: Callable[[Tensor], Tensor] = Identity(), audio_transform: Callable[[Tensor], Tensor] = Identity(), sample: bool = True, pad_to_max_len: bool = True, batch_size: int=4, num_workers: int=8):
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
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collator = BatchCollaterInferencePTLightning(pad_to_max_len, fps, frame_padding)

        self.val_take_factor = val_take_factor
        self.num_dists = num_dists

    def setup(self, stage: Optional[str] = None) -> None:
        # self.train_dataset = FakeAVCelebPretrain(subset=self.train_subset, root=self.root, frame_padding=self.frame_padding, fps=self.fps, img_size=self.img_size, video_transform=self.video_transform, audio_transform=self.audio_transform, sample=self.sample, pad_to_max_len=self.pad_to_max_len, take_factor=self.val_take_factor*2, num_dists=self.num_dists)
        self.val_dataset = DFDCInference(subset=self.val_subset, root=self.root, frame_padding=self.frame_padding, fps=self.fps, img_size=self.img_size, video_transform=self.video_transform, audio_transform=self.audio_transform, sample=self.sample, pad_to_max_len=self.pad_to_max_len, take_factor=self.val_take_factor, num_dists=self.num_dists)

    # def train_dataloader(self) -> TRAIN_DATALOADERS:
    #     return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collator, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collator, shuffle=False)

# The dictionary is used to map the feature type to the dataset type
# The key is a tuple of (visual_feature_type, audio_feature_type), ``None`` means using end-to-end encoder.
feature_type_to_dataset_type = {
    (None, None): DFDCInference
}