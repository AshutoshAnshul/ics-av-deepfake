from pytorch_lightning import Trainer

from dataset.lavdf_regression_bmn import LavdfBmnDataModule
from model.reg_model_bmn_plus import LocalizationModelBMNPlus

import fire

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
    max_duration: int = 40,
    exp_name_prefix: str = "lavdf_inference",
    finetuned: bool = False,
    regression_head: str = 'bmn',
    leave_section='None'
) -> None:

    if regression_head == 'bmn':
        model = LocalizationModelBMNPlus.load_from_checkpoint(model_checkpoint_path, exp_name_prefix = exp_name_prefix)
        print(f"Experiment: {model.exp_name_prefix}")
        print(model.max_len, model.fps)

    if dataset_name.lower() == 'lavdf':
        if regression_head == 'bmn':
            dm = LavdfBmnDataModule(
                train_subset=dataset_train_split,
                val_subset=dataset_val_split,
                test_subset=dataset_test_split,
                root=dataset_path,
                frame_padding=dataset_padding_num_frames,
                pad_to_max_len=dataset_pad_to_max_length,
                batch_size=eval_batch_size,
                num_workers=8,
                max_duration=max_duration,
                leave_section=leave_section
            )
        dm.setup()
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented yet")

    trainer = Trainer(logger=False,
        enable_checkpointing=False, devices=n_gpu if n_gpu > 0 else None,
        accelerator="gpu" if n_gpu > 0 else "cpu",
    )
    trainer.predict(model=model, dataloaders=dm.test_dataloader())

if __name__ == "__main__":
    fire.Fire(main)

## python evaluate_regression_bmn_pt_lightning.py --dataset_path '<dataset_path>' --dataset_name 'lavdf' --dataset_padding_num_frames 512 --dataset_train_split train --dataset_val_split dev --dataset_test_split test --eval_batch_size 4 --model_checkpoint_path '<model_path>'