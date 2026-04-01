from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging
from pytorch_lightning.tuner import Tuner

from dataset.fakeavceleb_classification import FakeAVCelebClassificationDataModule
from model.class_model import ClassificationModel

from utils import EarlyStoppingLR, LrLogger, reproducibility
import fire

def main(
    dataset_path: str,
    dataset_padding_num_frames: int,
    dataset_train_split: str,
    dataset_val_split: str,
    dataset_test_split: str,
    model_depth: int,
    model_feature_in: int,
    train_batch_size: int,
    resume_checkpoint_path: str | None = None,
    dataset_pad_to_max_length: bool = True,
    n_gpu: int = 1,
    epochs = 100,
    gradient_accumulation_steps: int = 1,
    leave_section='None',
    dataset_name: str = "voxceleb",
    sync_model_type: str = "decoder",
    train_dataset_name: str = "fakeavceleb"
) -> None:

    model = ClassificationModel(
        d_model=model_feature_in,
        K=model_depth,
        linear_layer_out=model_feature_in,
        num_encoder_layer=3*model_depth,
        num_classes=1,
        distributed=n_gpu>1
    )

    if train_dataset_name.lower() == 'fakeavceleb':
        dm = FakeAVCelebClassificationDataModule(
            train_subset=dataset_train_split,
            val_subset=dataset_val_split,
            test_subset=dataset_test_split,
            root=dataset_path,
            frame_padding=dataset_padding_num_frames,
            pad_to_max_len=dataset_pad_to_max_length,
            batch_size=train_batch_size,
            num_workers=8,
            leave_section=leave_section
        )


    # dm.setup()
    # train_data = dm.train_dataset.__getitem__(0)
    # print(train_data['filepath'], train_data['video_frames'], train_data['label'])
    # print(train_data['av_sync'].shape, train_data['v_sync'].shape, train_data['a_sync'].shape)

    monitor = "val_loss"
    
    trainer = Trainer(log_every_n_steps=50, precision="16-mixed", gradient_clip_val=0.5,
                      max_epochs=epochs, accumulate_grad_batches=gradient_accumulation_steps,
                      callbacks=[
                          ModelCheckpoint( dirpath=f"./{sync_model_type}_{dataset_name}/class_ckpt_{leave_section}", save_last=True, filename=f'class_model_{train_dataset_name.lower()}' + "-{epoch}-{val_loss:.3f}", monitor=monitor, mode="min"),
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

