from .base_metric import BaseMetric

import torch


class R2(BaseMetric):
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def __call__(
        self, 
        predicted: torch.Tensor, 
        observed_data: torch.Tensor,
        observed_masks: torch.Tensor,
        gt_masks: torch.Tensor, 
        **batch) -> float:
        """
        Metric calculation logic.

        Args:
            predicted (Tensor): model output predictions.
            observed_data (Tensor): ground-truth objects.
            observed_masks (Tensor): indicator of objects.
            gt_masks (Tensor): indicator of train objects.

        Returns:
            metric (float): calculated R^2 metric.
        """

        target_mask = observed_masks - gt_masks
        target_data = observed_data * target_mask
        predicted_data = predicted * target_mask

        num_target_elems = target_mask.sum()
        num_target_elems = (num_target_elems if num_target_elems > 0 else 1.0)
        ss_res_numer = ((target_data - predicted_data)**2).sum()
        ss_res = ss_res_numer / num_target_elems

        target_sum = target_data.sum()
        target_mean = target_sum / num_target_elems
        ss_tot_numer = ((target_data - target_mean)**2).sum()
        ss_tot = ss_tot_numer / num_target_elems

        return 1.0 - ss_res / ss_tot 
       