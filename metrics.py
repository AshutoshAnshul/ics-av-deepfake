# taken and modified from https://github.com/ControlNet/LAV-DF/blob/master/metrics.py

from typing import List, Union

import torch
from torch import Tensor
from tqdm.auto import tqdm

from utils import iou_1d


class AP:
    """
    Average Precision

    The mean precision in Precision-Recall curve.
    """

    def __init__(self, iou_thresholds: Union[float, List[float]] = 0.5, tqdm_pos: int = 1, device='cuda:0'):
        super().__init__()
        self.iou_thresholds: List[float] = iou_thresholds if type(iou_thresholds) is list else [iou_thresholds]
        self.tqdm_pos = tqdm_pos
        self.n_labels = 0
        self.ap: dict = {}
        self.device = device

    def __call__(self, proposals_dict: dict) -> dict:

        for iou_threshold in self.iou_thresholds:
            values = []
            self.n_labels = 0
            i =0

            for prop in tqdm(proposals_dict.values()):
                proposals = prop['proposals']
                # labels = torch.tensor(meta.fake_periods)
                labels = torch.tensor(prop['labels']).to(self.device)
                values.append(AP.get_values(iou_threshold, proposals, labels, 25))
                self.n_labels += len(labels)

            # sort proposals
            values = torch.cat(values).to(self.device)
            ind = values[:, 0].sort(stable=True, descending=True).indices
            values = values[ind]

            # accumulate to calculate precision and recall
            curve = self.calculate_curve(values)
            ap = self.calculate_ap(curve)
            self.ap[iou_threshold] = ap

        return self.ap

    def calculate_curve(self, values):
        is_TP = values[:, 1]
        acc_TP = torch.cumsum(is_TP, dim=0)
        precision = acc_TP / (torch.arange(len(is_TP)).to(self.device) + 1)
        recall = acc_TP / self.n_labels
        curve = torch.stack([recall, precision]).T
        curve = torch.cat([torch.tensor([[1., 0.]]).to(self.device), torch.flip(curve, dims=(0,))])
        return curve

    @staticmethod
    def calculate_ap(curve):
        x, y = curve.T
        y_max = y.cummax(dim=0).values
        x_diff = x.diff().abs() 
        ap = (x_diff * y_max[:-1]).sum()
        return ap

    @staticmethod
    def get_values(
        iou_threshold: float,
        proposals: Tensor,
        labels: Tensor,
        fps: float,
    ) -> Tensor:
        n_labels = len(labels)
        n_proposals = len(proposals)
        if n_labels > 0:
            # ious = iou_1d(proposals[:, 1:] / fps, labels) # uncomment this if you proposals are in form of frames and not seconds
            ious = iou_1d(proposals[:, 1:], labels) # uncomment this if you proposals are in form of seconds and not frames
        else:
            ious = torch.zeros((n_proposals, 0))

        # values: (confidence, is_TP) rows
        n_labels = ious.shape[1]
        detected = torch.full((n_labels,), False)
        confidence = proposals[:, 0]
        potential_TP = ious > iou_threshold

        tp_indexes = []

        for i in range(n_labels):
            potential_TP_index = potential_TP[:, i].nonzero()
            for (j,) in potential_TP_index:
                if j not in tp_indexes:
                    tp_indexes.append(j)
                    break
        
        is_TP = torch.zeros(n_proposals, dtype=torch.bool, device=confidence.device)
        if len(tp_indexes) > 0:
            tp_indexes = torch.stack(tp_indexes)
            is_TP[tp_indexes] = True
        values = torch.column_stack([confidence, is_TP])
        return values