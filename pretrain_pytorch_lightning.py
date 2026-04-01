import argparse

import math
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging
from pytorch_lightning.tuner import Tuner

from dataset.voxceleb import VoxCelebDataModule
from dataset.lrs2 import LRS2DataModule
from model.pretrain_model import ConsistencyPretrainModel, ConsistencyPretrainEncoderModel

from utils import EarlyStoppingLR, LrLogger, reproducibility
import fire

def main(
    dataset_path: str,
    dataset_padding_num_frames: int,
    dataset_train_split: str,
    dataset_eval_split: str,
    model_v_encoder_variant: str,
    model_a_encoder_variant: str,
    model_sync_loss: str,
    modal_consistency: str,
    model_depth: int,
    model_feature_in: int,
    model_temporal_dim: int,
    model_select_tgt_layer: str,
    train_batch_size: int,
    resume_checkpoint_path: str | None = None,
    dataset_pad_to_max_length: bool = True,
    dataset_fps: int = 25,
    dataset_img_size: int = 96,
    n_gpu: int = 1,
    epochs = 50,
    final_learning_rate: float = None,
    gradient_accumulation_steps: int = 1,
    dataset_name: str = "voxceleb",
    sync_model_type: str = "decoder"
) -> None:
    
    total_num_steps = math.ceil(train_batch_size * epochs / gradient_accumulation_steps)

    dists = n_gpu*train_batch_size*gradient_accumulation_steps
    val_take_factor = 2

    assert sync_model_type in ['decoder', 'sparse_encoder'], "Invalid model type. Can be decoder or sparse_encoder"
    print(f"sync-model-type: {sync_model_type}")

    if sync_model_type == 'decoder':
        model = ConsistencyPretrainModel(
            v_encoder=model_v_encoder_variant,
            a_encoder=model_a_encoder_variant,
            v_cla_feature_in=model_feature_in,
            a_cla_feature_in=model_feature_in,
            temporal_dim=model_temporal_dim,
            sync_loss=model_sync_loss,
            depth=model_depth,
            modal_consistency=modal_consistency,
            select_tgt_layer=model_select_tgt_layer,
            total_optimizer_step=total_num_steps,
            final_learning_rate=final_learning_rate,
            distributed= n_gpu>1
        )
    elif sync_model_type == 'sparse_encoder':
        model = ConsistencyPretrainEncoderModel(
            v_encoder=model_v_encoder_variant,
            a_encoder=model_a_encoder_variant,
            v_cla_feature_in=model_feature_in,
            a_cla_feature_in=model_feature_in,
            temporal_dim=model_temporal_dim,
            depth=model_depth,
            sync_loss=model_sync_loss,
            modal_consistency=modal_consistency,
            select_tgt_layer=model_select_tgt_layer,
            total_optimizer_step=total_num_steps,
            final_learning_rate=final_learning_rate,
            distributed= n_gpu>1
        )
    else:
        raise ValueError(f"Invalid model sync type name: {sync_model_type}")

    if dataset_name == 'voxceleb':
        dm = VoxCelebDataModule(
            train_subset=dataset_train_split,
            val_subset=dataset_eval_split,
            root=dataset_path,
            frame_padding=dataset_padding_num_frames,
            img_size=dataset_img_size,
            sample=True,
            fps=dataset_fps,
            pad_to_max_len=dataset_pad_to_max_length,
            batch_size=train_batch_size,
            num_workers=8,
            val_take_factor=val_take_factor,
            num_dists=dists
        )
    elif dataset_name == 'lrs2':
        dm = LRS2DataModule(
            train_subset=dataset_train_split,
            val_subset=dataset_eval_split,
            root=dataset_path,
            frame_padding=dataset_padding_num_frames,
            img_size=dataset_img_size,
            sample=True,
            fps=dataset_fps,
            pad_to_max_len=dataset_pad_to_max_length,
            batch_size=train_batch_size,
            num_workers=8,
            num_dists=dists
        )
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    monitor = "val_loss"
    
    trainer = Trainer(log_every_n_steps=50, precision="16-mixed", gradient_clip_val=0.5,
                      max_epochs=epochs, accumulate_grad_batches=gradient_accumulation_steps,
                      callbacks=[
                          ModelCheckpoint( dirpath=f"./{sync_model_type}_{dataset_name}/ckpt_var_2", save_last=True, filename=f'consistency_model_{modal_consistency}_{model_sync_loss}' + "-{epoch}-{val_loss:.3f}", monitor=monitor, mode="min"),
                          LrLogger(),
                          EarlyStoppingLR(lr_threshold=1e-7),
                          EarlyStopping(monitor=monitor, mode='min', verbose=False, patience=7),
                          StochasticWeightAveraging(1e-2)
                          ], 
                      enable_checkpointing=True,
                      benchmark=True,
                      accelerator="gpu",
                      devices=n_gpu,
                      strategy="auto" if n_gpu < 2 else "ddp"
                    )

    
    if resume_checkpoint_path is None:
        print('starting training')
        # tuner = Tuner(trainer)
        # tuner.lr_find(model,datamodule=dm)
        # print(model.learning_rate)
        trainer.fit(model, dm)
    else:
        print('resuming traning')
        trainer.fit(model=model, datamodule=dm, ckpt_path=resume_checkpoint_path)

if __name__ == "__main__":
    fire.Fire(main)

