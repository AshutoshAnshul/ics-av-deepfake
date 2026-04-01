import fire
from typing import Any
import pickle

from model.pretrain_model import ConsistencyPretrainModel, ConsistencyPretrainEncoderModel
from dataset.fakeavceleb_inference import FakeAVCelebPretrainDataModule
from dataset.kodf_inference import KoDFInferenceDataModule
from dataset.dfdc_inference import DFDCInferenceDataModule
from dataset.lavdf_inference import LavdfInferenceDataModule
from pytorch_lightning import LightningModule, Trainer, Callback

from model.sync_model import PositionalEncoding

class SaveFeatures(Callback):
    def __init__(self) -> None:
        super().__init__() 

    def on_predict_batch_end(self, 
                             trainer: Trainer, 
                             pl_module: LightningModule, 
                             outputs: Any, batch: Any, 
                             batch_idx: int, 
                             dataloader_idx: int = 0) -> None:
        
        cross_modal_out, v_out, a_out, av_hidden_feat, v_hidden_feat, a_hidden_feat, v_feat, a_feat = outputs

        batch_size = len(cross_modal_out)

        for i in range(batch_size):
            n_frames = int(batch['video_frames'][i])
            feature_file_path = '.'.join(str(batch['feature_file_path'][i]).split('.')[:-1]) + '.pkl'

            feat_json = {}
            feat_json['filepath'] = feature_file_path
            # feat_json['av_sync'] = cross_modal_out[i][:n_frames].tolist()
            # feat_json['v_sync'] = v_out[i][:n_frames].tolist()
            # feat_json['a_sync'] = a_out[i][:n_frames].tolist()
            feat_json['av_hidden_feat'] = av_hidden_feat[i][:n_frames].tolist()
            # feat_json['v_hidden_feat'] = v_hidden_feat[i][:n_frames].tolist()
            # feat_json['a_hidden_feat'] = a_hidden_feat[i][:n_frames].tolist()
            feat_json['v_feat'] = v_feat[i][:n_frames].tolist()
            feat_json['a_feat'] = a_feat[i][:n_frames].tolist()
            feat_json['video_frames'] = n_frames

            with open(feature_file_path, 'wb') as final_file:
                pickle.dump(feat_json, final_file)

def main(
    dataset_path: str,
    dataset_name: str,
    dataset_padding_num_frames: int,
    eval_batch_size: int,
    model_checkpoint_path: str | None = None,
    dataset_pad_to_max_length: bool = True,
    dataset_fps: int = 25,
    dataset_img_size: int = 96,
    n_gpu: int = 1,
    sync_model_type: str = "decoder"
) -> None:

    dists = n_gpu*eval_batch_size
    if sync_model_type=="decoder":
        model = ConsistencyPretrainModel.load_from_checkpoint(model_checkpoint_path, max_len=dataset_padding_num_frames)
        print(model.cross_modal_consistency.transformer_decoder.num_layers)
    else:
        model = ConsistencyPretrainEncoderModel.load_from_checkpoint(model_checkpoint_path, max_len=dataset_padding_num_frames)
    

    print(model.audio_encoder.block.pos_embedding.pe.shape)

    model.audio_encoder.block.pos_embedding = PositionalEncoding(192, 0, dataset_padding_num_frames)

    print(model.audio_encoder.block.pos_embedding.pe.shape)

    model.audio_consistency.position_embed = PositionalEncoding(256, 0, dataset_padding_num_frames)
    model.video_consistency.position_embed = PositionalEncoding(256, 0, dataset_padding_num_frames)
    model.cross_modal_consistency.position_embed = PositionalEncoding(256, 0, dataset_padding_num_frames)

    print(model.audio_consistency.position_embed.pe.shape)
    

    if dataset_name.lower() == 'fakeavceleb':
        dm = FakeAVCelebPretrainDataModule(
            root=dataset_path,
            frame_padding=dataset_padding_num_frames,
            img_size=dataset_img_size,
            fps=dataset_fps,
            pad_to_max_len=dataset_pad_to_max_length,
            batch_size=eval_batch_size,
            num_workers=8,
            num_dists=dists
        )
        dm.setup()
    elif dataset_name.lower() == 'kodf':
        dm = KoDFInferenceDataModule(
            root=dataset_path,
            frame_padding=dataset_padding_num_frames,
            img_size=dataset_img_size,
            fps=dataset_fps,
            pad_to_max_len=dataset_pad_to_max_length,
            batch_size=eval_batch_size,
            num_workers=8,
            num_dists=dists
        )
        dm.setup()
    elif dataset_name.lower() == 'dfdc':
        dm = DFDCInferenceDataModule(
            root=dataset_path,
            frame_padding=dataset_padding_num_frames,
            img_size=dataset_img_size,
            fps=dataset_fps,
            pad_to_max_len=dataset_pad_to_max_length,
            batch_size=eval_batch_size,
            num_workers=8,
            num_dists=dists
        )
        dm.setup()
    elif dataset_name.lower() == 'lavdf':
        dm = LavdfInferenceDataModule(
            root=dataset_path,
            frame_padding=dataset_padding_num_frames,
            img_size=dataset_img_size,
            fps=dataset_fps,
            pad_to_max_len=dataset_pad_to_max_length,
            batch_size=eval_batch_size,
            num_workers=8,
            num_dists=dists
        )
        dm.setup()
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented yet")

    trainer = Trainer(logger=False,
        enable_checkpointing=False, devices=n_gpu if n_gpu > 0 else None,
        accelerator="gpu" if n_gpu > 0 else "cpu",
        callbacks=[SaveFeatures()]
    )

    trainer.predict(model=model, dataloaders=dm.val_dataloader())

if __name__ == "__main__":
    fire.Fire(main)
            