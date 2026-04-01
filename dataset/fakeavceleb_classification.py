from pathlib import Path
from typing import Optional, List
import os
import pickle
from torch import tensor
from torch.nn.functional import pad
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from utils import read_json
from dataset.collater import BatchCollaterClassificationPTLightning

from dataclasses import dataclass

@dataclass
class Metadata:
    source: str
    target1: str
    target2: str
    method: str
    category: str
    type: str
    race: str
    gender: str
    filename: str
    path: str
    feature_file: str


class FakeAVCelebClassification(Dataset):
    def __init__(self, subset: str, root: str = "data", frame_padding: int = 512, take_factor = 4, num_dists = 32, pad_to_max_len: bool = True, leave_section='None') :

        self.subset = subset
        self.root = root
        self.video_padding = frame_padding
        self.pad_to_max_len = pad_to_max_len

        src_dir = Path(root)
        metadata_filename = "FakeAVCeleb/" + str(subset) + "_data.json"
        metadata_file = os.path.join(src_dir, metadata_filename)

        assert os.path.isfile(metadata_file), f"No file name {metadata_file} found" 

        metadata : List[Metadata] = read_json(metadata_file, lambda x: Metadata(**x))
        self.metadata : List[Metadata] = []

        if leave_section is None:
            self.metadata = metadata
            print('got here', len(metadata))
        elif leave_section.strip().lower() == 'none':
            self.metadata = metadata
            print('got-here', len(metadata))
        elif 'test' in subset:
            for meta in metadata:
                if meta.method.strip() == 'real':
                    self.metadata.append(meta)
                elif leave_section=='RVFA' and meta.type.strip()=='RealVideo-FakeAudio':
                    self.metadata.append(meta)
                elif leave_section=='FVRA-WL' and meta.type.strip()=='FakeVideo-RealAudio' and meta.method.strip()=='wav2lip':
                    self.metadata.append(meta)
                elif leave_section=='FVRA-FS' and meta.type.strip()=='FakeVideo-RealAudio' and meta.method.strip()=='faceswap':
                    self.metadata.append(meta)
                elif leave_section=='FVRA-GAN' and meta.type.strip()=='FakeVideo-RealAudio' and meta.method.strip()=='fsgan':
                    self.metadata.append(meta)
                elif leave_section=='FVFA-WL' and meta.type.strip()=='FakeVideo-FakeAudio' and meta.method.strip()=='wav2lip':
                    self.metadata.append(meta)
                elif leave_section=='FVFA-FS' and meta.type.strip()=='FakeVideo-FakeAudio' and meta.method.strip()=='faceswap-wav2lip':
                    self.metadata.append(meta)
                elif leave_section=='FVFA-GAN' and meta.type.strip()=='FakeVideo-FakeAudio' and meta.method.strip()=='fsgan-wav2lip':
                    self.metadata.append(meta)
        else:
            for meta in metadata:
                if meta.method.strip() == 'real':
                    self.metadata.append(meta)
                elif leave_section=='RVFA' and not (meta.type.strip()=='RealVideo-FakeAudio'):
                    self.metadata.append(meta)
                elif leave_section=='FVRA-WL' and not (meta.type.strip()=='FakeVideo-RealAudio' and meta.method.strip()=='wav2lip'):
                    self.metadata.append(meta)
                elif leave_section=='FVRA-FS' and not (meta.type.strip()=='FakeVideo-RealAudio' and meta.method.strip()=='faceswap'):
                    self.metadata.append(meta)
                elif leave_section=='FVRA-GAN' and not (meta.type.strip()=='FakeVideo-RealAudio' and meta.method.strip()=='fsgan'):
                    self.metadata.append(meta)
                elif leave_section=='FVFA-WL' and not (meta.type.strip()=='FakeVideo-FakeAudio' and meta.method.strip()=='wav2lip'):
                    self.metadata.append(meta)
                elif leave_section=='FVFA-FS' and not (meta.type.strip()=='FakeVideo-FakeAudio' and meta.method.strip()=='faceswap-wav2lip'):
                    self.metadata.append(meta)
                elif leave_section=='FVFA-GAN' and not (meta.type.strip()=='FakeVideo-FakeAudio' and meta.method.strip()=='fsgan-wav2lip'):
                    self.metadata.append(meta)

        voxceleb_metadata_filename = "FakeAVCeleb/voxceleb_data.json"
        voxceleb_metadata_file = os.path.join(src_dir, voxceleb_metadata_filename)

        assert os.path.isfile(voxceleb_metadata_file), f"No file name {voxceleb_metadata_file} found" 

        voxceleb_metadata : List[Metadata] = read_json(voxceleb_metadata_file, lambda x: Metadata(**x))
        train_data_len = int(len(voxceleb_metadata)*0.8)
        print(len(self.metadata), len(voxceleb_metadata), train_data_len)

        dropped_number = len(metadata)-len(self.metadata)

        if 'train' in subset:
            self.metadata.extend(voxceleb_metadata[:train_data_len])
        elif 'val' in subset:
            self.metadata.extend(voxceleb_metadata[train_data_len:])      

        print(f"Load {len(self.metadata)} data in {subset}. Dropped number {dropped_number}")

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        
        meta = self.metadata[index]

        feature_file_path = os.path.join(self.root, 'FakeAVCeleb/decoder_voxceleb_feat', str(meta.feature_file)).replace('.json', '.pkl')

        with open(feature_file_path, 'rb') as feature_file:
            video_data = pickle.load(feature_file)

        av_sync = tensor(video_data['av_hidden_feat'])
        v_sync = tensor(video_data['v_feat'])
        a_sync = tensor(video_data['a_feat'])
        video_frames = video_data['video_frames']

        frame_padding = self.video_padding - video_frames
        pad_val = [0, 0, 0, frame_padding]
        av_sync = pad(av_sync, pad_val)
        v_sync = pad(v_sync, pad_val)
        a_sync = pad(a_sync, pad_val)

        label = 0
        if str(meta.type).strip() == "RealVideo-RealAudio":
            label = 1

        outputs = {
            'filepath': video_data['filepath'],
            'av_sync': av_sync,
            'v_sync': v_sync,
            'a_sync': a_sync,
            'video_frames': video_frames,
            'label': label
        }
        return outputs
    
class FakeAVCelebClassificationDataModule(LightningDataModule):
    train_dataset = FakeAVCelebClassification
    val_dataset = FakeAVCelebClassification
    test_dataset = FakeAVCelebClassification

    def __init__(self, train_subset: str='dev', val_subset: str='test', test_subset: str='test', root: str = "data", frame_padding: int = 512, val_take_factor = 2, num_dists = 32,
        pad_to_max_len: bool = True, batch_size: int=4, num_workers: int=8, leave_section='None'):
        super().__init__()
        self.root = root
        self.frame_padding = frame_padding
        self.pad_to_max_len = pad_to_max_len
        self.train_subset = train_subset
        self.val_subset = val_subset
        self.test_subset = test_subset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collator = BatchCollaterClassificationPTLightning(pad_to_max_len, frame_padding)

        self.val_take_factor = val_take_factor
        self.num_dists = num_dists

        self.leave_section = leave_section

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = FakeAVCelebClassification(subset=self.train_subset, root=self.root, frame_padding=self.frame_padding, pad_to_max_len=self.pad_to_max_len, take_factor=self.val_take_factor*2, num_dists=self.num_dists, leave_section=self.leave_section)
        self.val_dataset = FakeAVCelebClassification(subset=self.val_subset, root=self.root, frame_padding=self.frame_padding, pad_to_max_len=self.pad_to_max_len, take_factor=self.val_take_factor, num_dists=self.num_dists, leave_section=self.leave_section)
        self.test_dataset = FakeAVCelebClassification(subset=self.test_subset, root=self.root, frame_padding=self.frame_padding, pad_to_max_len=self.pad_to_max_len, take_factor=self.val_take_factor, num_dists=self.num_dists, leave_section=self.leave_section)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collator, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=3*self.batch_size, num_workers=self.num_workers, collate_fn=self.collator, shuffle=False)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collator, shuffle=False)

# The dictionary is used to map the feature type to the dataset type
# The key is a tuple of (visual_feature_type, audio_feature_type), ``None`` means using end-to-end encoder.
feature_type_to_dataset_type = {
    (None, None): FakeAVCelebClassification
}