from typing import List
import logging
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base class for the datasets.

    Given a proper index (list[dict]), allows to process different datasets
    for the same task in the identical manner. Therefore, to work with
    several datasets, the user only have to define index in a nested class.
    """

    def __init__(
        self, 
        X: np.array,
        timestamps: np.array,
        news: pd.DataFrame,
        limit: int=None,
        seq_len: int=32,
        instance_transforms: dict=None,
    ):
        """
        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
            instance_transforms (dict[Callable] | None): transforms that
                should be applied on the instance. Depend on the
                tensor name.
        """

        self.seq_len = seq_len
        X, timestamps = self._limit_index(X, timestamps, limit)
        self.X = X
        self.timestamps = timestamps
        self.news = news

        self.instance_transforms = instance_transforms

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        timestamps = torch.tensor(self.timestamps[ind: ind+self.seq_len], dtype=torch.float32)
        observed_data = torch.tensor(self.X[ind: ind+self.seq_len], dtype=torch.float32)
        news.groupby('Sector')

        observed_masks = ~torch.isnan(observed_data)
        masks = observed_masks.clone()
        masks[-1] = False
        gt_masks = masks.float()
        observed_data = torch.nan_to_num(observed_data)
        observed_masks = observed_masks.float()

        instance_data = {
            "gt_masks": gt_masks,
            "observed_data": observed_data,
            "observed_masks": observed_masks,
            "observed_timestamps": timestamps,
        }

        return instance_data

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return len(self.X) - self.seq_len + 1

    def preprocess_data(self, instance_data):
        """
        Preprocess data with instance transforms.

        Each tensor in a dict undergoes its own transform defined by the key.

        Args:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element) (possibly transformed via
                instance transform).
        """
        if self.instance_transforms is not None:
            for transform_name in self.instance_transforms.keys():
                instance_data[transform_name] = self.instance_transforms[
                    transform_name
                ](instance_data[transform_name])
        return instance_data

    @staticmethod
    def _filter_records_from_dataset(
        index: list,
    ) -> list:
        """
        Filter some of the elements from the dataset depending on
        some condition.

        This is not used in the example. The method should be called in
        the __init__ before shuffling and limiting.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset that satisfied the condition. The dict has
                required metadata information, such as label and object path.
        """
        # Filter logic
        pass

    def _limit_index(self, X, timestamps, limit):
        """
        Limit the total number of elements.

        Args:
            X (np.array): list, containing all observed data
            timestamps (np.array): list containing all observed timestamps
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
        """
        if limit is not None:
            limit = self.seq_len + limit - 1
            X = X[:limit]
            timestamps = timestamps[:limit]
        return X, timestamps
