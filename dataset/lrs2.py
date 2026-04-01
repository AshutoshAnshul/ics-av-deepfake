from pathlib import Path
from typing import Optional, List, Callable, Any, Union, Tuple
import os
import errno
import math
import random
import torch
import pickle
import torchaudio
from einops import rearrange
from torch import Tensor
from torch.nn import Identity
from torch.utils.data import Dataset, DataLoader, RandomSampler
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from utils import read_video, padding_video, padding_audio, resize_video
from dataset.collater import BatchCollaterPTLightning

class LRS2(Dataset):
    def __init__(self, subset: str, root: str = "data", frame_padding: int = 512, fps: int = 25, img_size = 96, num_dists = 32,
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
        
        filename = "filtered_pretrain_"+subset+"_file_100.pkl"
        metadata_file = os.path.join(src_dir, filename)

        if os.path.isfile(metadata_file):
            with open(metadata_file, 'rb') as meta_file:
                self.metadata = pickle.load(meta_file)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)
        
        sample_size = len(self.metadata)
        
        self.metadata = random.sample(self.metadata, sample_size)

        sample_size = (int(sample_size/num_dists))*num_dists
        self.metadata = self.metadata[:sample_size]
        
        print(f"Load {len(self.metadata)} data in {subset}.")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        filePath = str(self.metadata[idx]).strip()
        video, audio, _ = read_video(filePath)
        video_frames = min(self.video_padding, video.shape[0])

        video = padding_video(video, target=self.video_padding)
        audio = padding_audio(audio, target=self.audio_padding)
            
        video = self.video_transform(video)
        audio = self.audio_transform(audio)

        video = rearrange(resize_video(video, (self.img_size, self.img_size)), "t c h w -> c t h w")
        audio = self._get_log_mel_spectrogram(audio)

        # outputs = [video, audio, video_frames, filePath]
        outputs = {
            'video': video,
            'audio': audio,
            'video_frames': video_frames,
            'filePath': filePath
        }
        return outputs
    
    @staticmethod
    def _get_log_mel_spectrogram(audio: Tensor) -> Tensor:
        ms = torchaudio.transforms.MelSpectrogram(n_fft=321, n_mels=64)
        spec = torch.log(ms(audio[:, 0]) + 0.01)
        # assert spec.shape == (64, 2048), "Wrong log mel-spectrogram setup in Dataset"
        return spec
    
class LRS2DataModule(LightningDataModule):
    train_dataset = LRS2
    val_dataset = LRS2

    def __init__(self, train_subset: str='train', val_subset: str='val', root: str = "data", frame_padding: int = 512, fps: int = 25, img_size = 96, num_dists = 32,
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
        self.collator = BatchCollaterPTLightning(pad_to_max_len, fps, frame_padding)

        self.num_dists = num_dists

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = LRS2(subset=self.train_subset, root=self.root, frame_padding=self.frame_padding, fps=self.fps, img_size=self.img_size, video_transform=self.video_transform, audio_transform=self.audio_transform, sample=self.sample, pad_to_max_len=self.pad_to_max_len, num_dists=self.num_dists)
        self.val_dataset = LRS2(subset=self.val_subset, root=self.root, frame_padding=self.frame_padding, fps=self.fps, img_size=self.img_size, video_transform=self.video_transform, audio_transform=self.audio_transform, sample=self.sample, pad_to_max_len=self.pad_to_max_len, num_dists=self.num_dists)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collator, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size*3, num_workers=self.num_workers, collate_fn=self.collator, shuffle=False)

# The dictionary is used to map the feature type to the dataset type
# The key is a tuple of (visual_feature_type, audio_feature_type), ``None`` means using end-to-end encoder.
feature_type_to_dataset_type = {
    (None, None): LRS2
}


