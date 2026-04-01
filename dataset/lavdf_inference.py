import os
from dataclasses import dataclass
from typing import Optional, List, Callable, Any, Tuple

import torch
import torchaudio
from einops import rearrange
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch import Tensor
from torch.nn import functional as F, Identity
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import Dataset
from dataset.collater import BatchCollaterInferencePTLightning

from utils import read_json, read_video, padding_video, padding_audio, resize_video

@dataclass
class Metadata:
    file: str
    n_fakes: int
    fake_periods: List[List[int]]
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

class LavdfInference(Dataset):

    def __init__(self, subset: str, root: str = "data", frame_padding: int = 512, fps: int = 25, img_size = 96, take_factor = 4, num_dists = 32,
        video_transform: Callable[[Tensor], Tensor] = Identity(), audio_transform: Callable[[Tensor], Tensor] = Identity(), sample: bool = True, pad_to_max_len: bool = True
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

        self.metadata: List[Metadata] = read_json(os.path.join(self.root, "metadata.json"), lambda x: Metadata(**x))

        def filter_existing(meta):
            feature_file = '_'.join(str(meta.file).strip().split('/'))
            feature_file_path = os.path.join(root, 'decoder_voxceleb_feat', str(feature_file)).replace('.mp4', '.pkl')
            return not os.path.isfile(feature_file_path)

        print(len(self.metadata))
        self.metadata = list(filter(filter_existing, self.metadata))
        print(f"Load Filtered {len(self.metadata)} data in {subset}.")

    def __getitem__(self, index: int) -> List[Tensor]:
        meta = self.metadata[index]

        feature_file = '_'.join(str(meta.file).strip().split('/'))
        feature_file_path = os.path.join(self.root, 'decoder_voxceleb_feat', str(feature_file))

        filePath = os.path.join(self.root, str(meta.file).strip())

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

    def __len__(self):
        return len(self.metadata)

    def _get_frame_and_segment_labels(self, meta: Metadata, num_frames: int, fps: float):
        fusion_frame_label = torch.zeros(num_frames)
        video_frame_label = torch.zeros(num_frames)
        audio_frame_label = torch.zeros(num_frames)

        for begin, end in meta.fake_periods:
            begin = int(begin * fps)
            end = int(end * fps)
            fusion_frame_label[begin: end] = 1

            if meta.modify_video:
                video_frame_label[begin: end] = 1
            if meta.modify_audio:
                audio_frame_label[begin: end] = 1
        
        video_fake_period = []
        audio_fake_period = []

        if meta.modify_audio:
            audio_fake_period = meta.fake_periods
        if meta.modify_video:
            video_fake_period = meta.fake_periods
        
        return [fusion_frame_label, meta.fake_periods, video_frame_label, video_fake_period, audio_frame_label, audio_fake_period]

    @staticmethod
    def _get_log_mel_spectrogram(audio: Tensor) -> Tensor:
        ms = torchaudio.transforms.MelSpectrogram(n_fft=321, n_mels=64)
        spec = torch.log(ms(audio[:, 0]) + 0.01)
        # assert spec.shape == (64, 2048), "Wrong log mel-spectrogram setup in Dataset"
        return spec
    
class LavdfInferenceDataModule(LightningDataModule):
    train_dataset: LavdfInference
    dev_dataset: LavdfInference
    test_dataset: LavdfInference
    metadata: List[Metadata]

    def __init__(self, train_subset: str='dev', val_subset: str='test', root: str = "data", frame_padding: int = 512, fps: int = 25, img_size = 96, val_take_factor = 2, num_dists = 32,
        video_transform: Callable[[Tensor], Tensor] = Identity(), audio_transform: Callable[[Tensor], Tensor] = Identity(), sample: bool = True, pad_to_max_len: bool = True, batch_size: int=4, num_workers: int=8
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
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collator = BatchCollaterInferencePTLightning(pad_to_max_len, fps, frame_padding)

        self.val_take_factor = val_take_factor
        self.num_dists = num_dists

    def setup(self, stage: Optional[str] = None) -> None:

        self.val_dataset = LavdfInference(subset=self.val_subset, root=self.root, frame_padding=self.frame_padding, fps=self.fps, img_size=self.img_size, video_transform=self.video_transform, audio_transform=self.audio_transform, sample=self.sample, pad_to_max_len=self.pad_to_max_len, take_factor=self.val_take_factor, num_dists=self.num_dists)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collator, shuffle=False)


# The dictionary is used to map the feature type to the dataset type
# The key is a tuple of (visual_feature_type, audio_feature_type), ``None`` means using end-to-end encoder.
feature_type_to_dataset_type = {
    (None, None): LavdfInference
}