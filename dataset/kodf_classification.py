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


class KoDFClassification(Dataset):
    def __init__(self, subset: str, root: str = "data", frame_padding: int = 512, take_factor = 4, num_dists = 32, pad_to_max_len: bool = True, leave_section='None') :

        self.subset = subset
        self.root = root
        self.video_padding = frame_padding
        self.pad_to_max_len = pad_to_max_len

        src_dir = Path(root)

        metadata_file = os.path.join(src_dir, "all_files.pkl")

        assert os.path.isfile(metadata_file), f"No file name {metadata_file} found" 

        with open(metadata_file, 'rb') as meta_file:
            self.metadata : List[str] = pickle.load(meta_file) 

        print(f"Load {len(self.metadata)} data in {subset}")

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        
        meta = self.metadata[index]

        filename = '_'.join(str(meta).split('/')[-2:]).replace('.mp4', '.pkl')

        feature_file_path = os.path.join(self.root, 'decoder_lrs2_feat', filename)

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
        if 'origin' in meta:
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
    
class KoDFClassificationDataModule(LightningDataModule):
    train_dataset = KoDFClassification
    val_dataset = KoDFClassification
    test_dataset = KoDFClassification

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
        self.test_dataset = KoDFClassification(subset=self.test_subset, root=self.root, frame_padding=self.frame_padding, pad_to_max_len=self.pad_to_max_len, take_factor=self.val_take_factor, num_dists=self.num_dists, leave_section=self.leave_section)

    # def train_dataloader(self) -> TRAIN_DATALOADERS:
    #     return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collator, shuffle=True)

    # def val_dataloader(self) -> EVAL_DATALOADERS:
    #     return DataLoader(self.val_dataset, batch_size=3*self.batch_size, num_workers=self.num_workers, collate_fn=self.collator, shuffle=False)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collator, shuffle=False)

# The dictionary is used to map the feature type to the dataset type
# The key is a tuple of (visual_feature_type, audio_feature_type), ``None`` means using end-to-end encoder.
feature_type_to_dataset_type = {
    (None, None): KoDFClassification
}