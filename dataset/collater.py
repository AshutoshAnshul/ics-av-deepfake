from torch import Tensor, stack, arange, log as t_log, tensor
from torchaudio.transforms import MelSpectrogram
from utils import padding_audio, padding_video
from einops import rearrange
from typing import Any

class BatchCollater(object):
    def __init__(
        self,
        is_pre_padded: bool = True,
        fps: int = 25,
    ) -> None:
        self.is_pre_padded = is_pre_padded
        self.fps = fps
        
    @staticmethod
    def _get_log_mel_spectrogram(audio: Tensor) -> Tensor:
        ms = MelSpectrogram(n_fft=321, n_mels=64)
        if len(audio.shape)==3:
            spec = t_log(ms(audio[:, :, 0]) + 0.01)
        elif len(audio.shape)==2:
            spec = t_log(ms(audio[:, 0]) + 0.01)
        # assert spec.shape == (64, 2048), "Wrong log mel-spectrogram setup in Dataset"
        return spec
        
    def __call__(self, batch: list[Tensor]) -> tuple[Tensor]:
        # video, audio, video_frames, *_ =  batch
        video = []
        audio = []
        video_frames = []
        for x in batch:
            video.append(x['video'])
            audio.append(x['audio'])
            video_frames.append(x['video_frames'])

        del batch
        # print(video[0].shape, video[1])
        video = stack(video, dim=0)
        audio = stack(audio, dim=0)
        video_frames = tensor(video_frames)

        # print(video.shape, audio.shape)
        # print(len(video_frames), video_frames[0])
        
        if not self.is_pre_padded:
            max_frame_len = max(video_frames) + 1
            audio_padding = int((max_frame_len*16000) / self.fps)

            vids = []
            auds = []
            for vid, aud in zip(video, audio):
                vids.append(padding_video(vid, target=max_frame_len))
                auds.append(padding_audio(aud, target=audio_padding))
            
            video = stack(vids, dim=0)
            audio = stack(auds, dim=0)

            del vids
            del auds

        video = rearrange(video, "b t c h w -> b c t h w")
        audio = BatchCollater._get_log_mel_spectrogram(audio)

        batch_size, _, num_frames, _, _ = video.shape

        padding_mask = arange(num_frames).expand(batch_size, num_frames) >= video_frames.unsqueeze(1)
        result: dict[Tensor] = {'video': video, 'audio': audio, 'video_frames': video_frames, 'padding_mask': padding_mask}
        return result
    



class BatchCollaterPTLightning(object):
    def __init__(self, pad_to_max_len: bool = True, fps: int = 25, max_len: int = 750) -> None:
        self.pad_to_max_len = pad_to_max_len
        self.fps = fps
        self.max_len_video = max_len
        
    def __call__(self, batch: list[Tensor]) -> tuple[Tensor]:
        # video, audio, video_frames, *_ =  batch
        video = stack([x['video'] for x in batch], dim=0)
        audio = stack([x['audio'] for x in batch], dim=0)
        video_frames = tensor([x['video_frames'] for x in batch])
        
        video_padding = max(video_frames)
        if (not self.pad_to_max_len) and (video_padding<self.max_len_video):
            video = video[:, :, :video_padding, :, :]
            audio = audio[:, :, :4*video_padding]

        batch_size, _, num_frames, _, _ = video.shape
        padding_mask = arange(num_frames).expand(batch_size, num_frames) >= video_frames.unsqueeze(1)
        result: dict[Tensor] = {'video': video, 'audio': audio, 'video_frames': video_frames, 'padding_mask': padding_mask}
        return result
    
class BatchCollaterInferencePTLightning(object):
    def __init__(self, pad_to_max_len: bool = True, fps: int = 25, max_len: int = 750) -> None:
        self.pad_to_max_len = pad_to_max_len
        self.fps = fps
        self.max_len_video = max_len
        
    def __call__(self, batch: list[Tensor]) -> tuple[Tensor]:
        # video, audio, video_frames, *_ =  batch
        video = stack([x['video'] for x in batch], dim=0)
        audio = stack([x['audio'] for x in batch], dim=0)
        video_frames = tensor([x['video_frames'] for x in batch])
        feature_files = [x['feature_file_path'] for x in batch]
        
        video_padding = max(video_frames)
        if (not self.pad_to_max_len) and (video_padding<self.max_len_video):
            video = video[:, :, :video_padding, :, :]
            audio = audio[:, :, :4*video_padding]

        batch_size, _, num_frames, _, _ = video.shape
        padding_mask = arange(num_frames).expand(batch_size, num_frames) >= video_frames.unsqueeze(1)
        result: dict[Tensor] = {'video': video, 'audio': audio, 'video_frames': video_frames, 'padding_mask': padding_mask, 'feature_file_path': feature_files}
        return result
    
class BatchCollaterClassificationPTLightning(object):
    def __init__(self, pad_to_max_len: bool = True,max_len: int = 750) -> None:
        self.pad_to_max_len = pad_to_max_len
        self.max_len_video = max_len
        
    def __call__(self, batch: list[Tensor]) -> tuple[Tensor]:
        # video, audio, video_frames, *_ =  batch
        filepaths = [x['filepath'] for x in batch]
        av_sync = stack([x['av_sync'] for x in batch], dim=0)
        v_sync = stack([x['v_sync'] for x in batch], dim=0)
        a_sync = stack([x['a_sync'] for x in batch], dim=0)
        video_frames = tensor([x['video_frames'] for x in batch])
        label = tensor([x['label'] for x in batch])
        # label = stack([x['label'] for x in batch], dim=0)
        
        video_padding = max(video_frames)
        if (not self.pad_to_max_len) and (video_padding<self.max_len_video):
            av_sync = av_sync[:video_padding, :]
            v_sync = v_sync[:video_padding, :]
            a_sync = a_sync[:video_padding, :]

        batch_size, num_frames, _ = av_sync.shape
        padding_mask = arange(num_frames).expand(batch_size, num_frames) >= video_frames.unsqueeze(1)
        result: dict[Tensor] = {'av_sync': av_sync, 'v_sync': v_sync, 'a_sync': a_sync, 'video_frames': video_frames, 'label': label, 'padding_mask': padding_mask, 'filepath': filepaths}
        return result
    

class BatchCollaterRegressionPTLightning(object):
    def __init__(self, pad_to_max_len: bool = True,max_len: int = 750) -> None:
        self.pad_to_max_len = pad_to_max_len
        self.max_len_video = max_len
        
    def __call__(self, batch: list[Tensor]) -> tuple[Tensor]:
        # video, audio, video_frames, *_ =  batch
        filepaths = [x['filepath'] for x in batch]
        av_sync = stack([x['av_sync'] for x in batch], dim=0)
        v_sync = stack([x['v_sync'] for x in batch], dim=0)
        a_sync = stack([x['a_sync'] for x in batch], dim=0)
        video_frames = tensor([x['video_frames'] for x in batch])
        segments = [x['segments'] for x in batch]
        # label = stack([x['label'] for x in batch], dim=0)
        
        video_padding = max(video_frames)
        if (not self.pad_to_max_len) and (video_padding<self.max_len_video):
            av_sync = av_sync[:video_padding, :]
            v_sync = v_sync[:video_padding, :]
            a_sync = a_sync[:video_padding, :]

        batch_size, num_frames, _ = av_sync.shape
        padding_mask = arange(num_frames).expand(batch_size, num_frames) >= video_frames.unsqueeze(1)
        result: dict[Tensor] = {'av_sync': av_sync, 'v_sync': v_sync, 'a_sync': a_sync, 'video_frames': video_frames, 'segments': segments, 'padding_mask': padding_mask, 'filepath': filepaths}
        return result

class BatchCollaterRegressionBMNPTLightning(object):
    def __init__(self, pad_to_max_len: bool = True,max_len: int = 750) -> None:
        self.pad_to_max_len = pad_to_max_len
        self.max_len_video = max_len
        
    def __call__(self, batch: list[Tensor]) -> tuple[Tensor]:
        # video, audio, video_frames, *_ =  batch
        filepaths = [x['filepath'] for x in batch]
        av_sync = stack([x['av_sync'] for x in batch], dim=0)
        v_sync = stack([x['v_sync'] for x in batch], dim=0)
        a_sync = stack([x['a_sync'] for x in batch], dim=0)

        fusion_gt_iou_map = stack([x['fusion_gt_iou_map'] for x in batch], dim=0)
        vid_gt_iou_map = stack([x['vid_gt_iou_map'] for x in batch], dim=0)
        aud_gt_iou_map = stack([x['aud_gt_iou_map'] for x in batch], dim=0)

        fusion_frame_label = stack([x['fusion_frame_label'] for x in batch], dim=0)
        vid_frame_label = stack([x['vid_frame_label'] for x in batch], dim=0)
        aud_frame_label = stack([x['aud_frame_label'] for x in batch], dim=0)

        video_frames = tensor([x['video_frames'] for x in batch])
        segments = [x['segments'] for x in batch]
        
        video_padding = max(video_frames)
        if (not self.pad_to_max_len) and (video_padding<self.max_len_video):
            av_sync = av_sync[:video_padding, :]
            v_sync = v_sync[:video_padding, :]
            a_sync = a_sync[:video_padding, :]

        batch_size, num_frames, _ = av_sync.shape
        padding_mask = arange(num_frames).expand(batch_size, num_frames) >= video_frames.unsqueeze(1)
        result: dict[Tensor] = {'av_sync': av_sync, 
                                'v_sync': v_sync, 
                                'a_sync': a_sync, 
                                'video_frames': video_frames, 
                                'segments': segments,
                                'fusion_gt_iou_map': fusion_gt_iou_map,
                                'fusion_frame_label': fusion_frame_label,
                                'vid_gt_iou_map': vid_gt_iou_map,
                                'vid_frame_label': vid_frame_label,
                                'aud_gt_iou_map': aud_gt_iou_map,
                                'aud_frame_label': aud_frame_label,
                                'padding_mask': padding_mask, 
                                'filepath': filepaths}
        return result