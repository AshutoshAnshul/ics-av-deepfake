from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging
from pytorch_lightning.tuner import Tuner
from dataset.lavdf_regression_bmn import LavdfBmnDataModule
from model.reg_model_bmn_plus import LocalizationModelBMNPlus

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
    train_dataset_name: str = "avdf1m",
    include_encoders: bool = True,
    pretrain_model_checkpoint: str = '',
    regression_head: str = 'bmn'
) -> None:

    if train_dataset_name.lower() == 'lavdf':

        if regression_head == 'bmn':
            max_duration = 40
            model = LocalizationModelBMNPlus(
                d_model=model_feature_in,
                K=model_depth,
                linear_layer_out=model_feature_in,
                num_head_layers=model_depth,
                num_classes=1,
                distributed=n_gpu>1,
                max_len=dataset_padding_num_frames,
                max_duration=max_duration,
                exp_name_prefix='lavdf_inference'
            )

            dm = LavdfBmnDataModule(
                train_subset=dataset_train_split,
                val_subset=dataset_val_split,
                test_subset=dataset_test_split,
                root=dataset_path,
                frame_padding=dataset_padding_num_frames,
                pad_to_max_len=dataset_pad_to_max_length,
                batch_size=train_batch_size,
                num_workers=8,
                max_duration=max_duration,
                leave_section=leave_section
            )


    dm.setup()
    train_data = dm.train_dataset.__getitem__(0)
    print(train_data['filepath'], train_data['video_frames'], train_data['segments'], train_data['video_frames'])
    # print(train_data['av_sync'].shape, train_data['v_sync'].shape, train_data['a_sync'].shape)
    print(train_data['fusion_gt_iou_map'].shape, train_data['vid_gt_iou_map'].shape, train_data['aud_gt_iou_map'].shape)
    print(train_data['fusion_frame_label'].shape, train_data['vid_frame_label'].shape, train_data['aud_frame_label'].shape)

    if regression_head == 'bmn':
        monitor = "val_loss"
        mode = 'min'
        extra_info = "-{epoch}-{val_loss:.3f}"
        precision="16-mixed"
    else:
        monitor = 'val_mAP'
        mode = 'max'
        extra_info = "-{epoch}-{val_mAP:.3f}"
        precision="32"
    
    trainer = Trainer(log_every_n_steps=50, precision=precision, gradient_clip_val=0.5,
                      max_epochs=epochs, accumulate_grad_batches=gradient_accumulation_steps,
                      callbacks=[
                          ModelCheckpoint( dirpath=f"./{sync_model_type}_{dataset_name}/reg_ckpt_{leave_section}_{regression_head}", save_last=True, filename=f'class_model_{train_dataset_name.lower()}' + extra_info, monitor=monitor, mode=mode),
                          LrLogger(),
                          EarlyStoppingLR(lr_threshold=1e-7),
                          EarlyStopping(monitor=monitor, mode=mode, verbose=False, patience=7),
                          StochasticWeightAveraging(1e-2)
                          ], 
                      enable_checkpointing=True,
                      benchmark=True,
                      accelerator="gpu",
                      devices=n_gpu,
                      strategy="auto" if n_gpu < 2 else "ddp_find_unused_parameters_true"
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

