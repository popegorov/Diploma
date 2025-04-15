import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    obs_data = []
    obs_masks = []
    gt_masks = []
    obs_tps = []

    for cur_dict in dataset_items:
        obs_data.append(cur_dict["observed_data"])
        obs_masks.append(cur_dict['observed_masks'])
        obs_tps.append(cur_dict['observed_timestamps'])
        gt_masks.append(cur_dict['gt_masks'])


    result_batch["observed_data"] = torch.stack(obs_data, dim=0).transpose(1, 2)
    result_batch["observed_masks"] = torch.stack(obs_masks, dim=0).transpose(1, 2)
    result_batch["observed_timestamps"] = torch.stack(obs_tps, dim=0)
    result_batch["gt_masks"] = torch.stack(gt_masks, dim=0).transpose(1, 2)
    return result_batch
