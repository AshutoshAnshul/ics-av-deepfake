from pytorch_lightning import Trainer, Callback, LightningModule
import pickle

from typing import Any

from sklearn.metrics import confusion_matrix, accuracy_score, average_precision_score, roc_auc_score
import numpy as np

from torch import sigmoid

from dataset.fakeavceleb_classification import FakeAVCelebClassificationDataModule
from dataset.kodf_classification import KoDFClassificationDataModule
from dataset.dfdc_classification import DFDCClassificationDataModule
from model.class_model import ClassificationModel

import fire

class SaveFeatures(Callback):
    def __init__(self, filesave:str) -> None:
        super().__init__() 
        self.predict_dict = {}
        self.filesave = filesave

    def on_predict_batch_end(self, 
                             trainer: Trainer, 
                             pl_module: LightningModule, 
                             outputs: Any, batch: Any, 
                             batch_idx: int, 
                             dataloader_idx: int = 0) -> None:
        
        pred = sigmoid(outputs)

        batch_size = len(pred)

        for i in range(batch_size):
            filename = str(batch['filepath'][i])
            label = int(batch['label'][i])
            prob = float(pred[i][0].cpu())
            self.predict_dict[filename] = {}
            self.predict_dict[filename]['pred'] = prob
            self.predict_dict[filename]['label'] = label

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        with open(self.filesave, 'wb') as pred_file:
            pickle.dump(self.predict_dict, pred_file)

        print('file saved')

def main(
    dataset_path: str,
    dataset_name: str,
    dataset_padding_num_frames: int,
    dataset_train_split: str,
    dataset_val_split: str,
    dataset_test_split: str,
    eval_batch_size: int,
    model_checkpoint_path: str | None = None,
    dataset_pad_to_max_length: bool = True,
    n_gpu: int = 1,
    leave_section='None',
    pretrain_dataset_name: str = "voxceleb",
    sync_model_type: str = "decoder"
) -> None:

    model = ClassificationModel.load_from_checkpoint(model_checkpoint_path)

    if dataset_name.lower() == 'fakeavceleb':
        dm = FakeAVCelebClassificationDataModule(
            train_subset=dataset_train_split,
            val_subset=dataset_val_split,
            test_subset=dataset_test_split,
            root=dataset_path,
            frame_padding=dataset_padding_num_frames,
            pad_to_max_len=dataset_pad_to_max_length,
            batch_size=eval_batch_size,
            num_workers=8,
            leave_section=leave_section
        )
        dm.setup()
    elif dataset_name.lower() == 'kodf':
        dm = KoDFClassificationDataModule(
            train_subset=dataset_train_split,
            val_subset=dataset_val_split,
            test_subset=dataset_test_split,
            root=dataset_path,
            frame_padding=dataset_padding_num_frames,
            pad_to_max_len=dataset_pad_to_max_length,
            batch_size=eval_batch_size,
            num_workers=8,
            leave_section=leave_section
        )
        dm.setup()
    elif dataset_name.lower() == 'dfdc':
        dm = DFDCClassificationDataModule(
            train_subset=dataset_train_split,
            val_subset=dataset_val_split,
            test_subset=dataset_test_split,
            root=dataset_path,
            frame_padding=dataset_padding_num_frames,
            pad_to_max_len=dataset_pad_to_max_length,
            batch_size=eval_batch_size,
            num_workers=8,
            leave_section=leave_section
        )
        dm.setup()
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented yet")
    
    file_save_path = f"./{sync_model_type}_{pretrain_dataset_name}/class_ckpt_{leave_section}/prediction.pkl"

    trainer = Trainer(logger=False,
        enable_checkpointing=False, devices=n_gpu if n_gpu > 0 else None,
        accelerator="gpu" if n_gpu > 0 else "cpu",
        callbacks=[SaveFeatures(file_save_path)]
    )

    trainer.predict(model=model, dataloaders=dm.test_dataloader())

    with open(file_save_path, 'rb') as pred_file:
        data = pickle.load(pred_file)

    pred = []
    label = []

    filenames = data.keys()
    for filen in filenames:
        pred.append(data[filen]['pred'])
        label.append(data[filen]['label'])

    print(len(label), sum(label))

    label = 1- np.array(label)
    pred = 1- np.array(pred)

    print('AP', average_precision_score(label, pred))
    print('AUC', roc_auc_score(label, pred))

    threshs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for thres in threshs:
        pred_thres = (pred>thres)
        print(confusion_matrix(label, pred_thres))
        print(accuracy_score(label, pred_thres))

if __name__ == "__main__":
    fire.Fire(main)

## python evaluate_classification_pt_lightning.py --dataset_path '<dataset_path>' --dataset_name 'fakeavceleb' --dataset_padding_num_frames 500 --dataset_train_split train --dataset_val_split val --dataset_test_split test --eval_batch_size 12 --model_checkpoint_path '<model_path>' --n_gpu 1 --leave_section None --pretrain_dataset_name <voxceleb/lrs2> --sync_model_type <decoder/sparse_encoder>
