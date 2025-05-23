from typing import List
import logging
import numpy as np
import pandas as pd
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
        X: pd.DataFrame,
        tensor_news: torch.Tensor,
        stocks: list,
        limit: int=None,
        seq_len: int=32,
        instance_transforms: dict=None,
    ):
        """
        Args:
            X (pd.DataFrame): Frame with all stocks data
            tensor_news (torch.Tensor): tensor with relevant news
            stocks (list): observed stocks 
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            instance_transforms (dict[Callable] | None): transforms that
                should be applied on the instance. Depend on the
                tensor name.
        """

        self.seq_len = seq_len
        X = self._limit_index(X, limit)
        self.X = X
        self.tensor_news = tensor_news
        self.observed_stocks = stocks

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
        timestamps = torch.arange(self.seq_len, dtype=torch.float32)
        observed_data = torch.tensor(self.X[ind: ind+self.seq_len], dtype=torch.float32)

        observed_masks = ~torch.isnan(observed_data)
        masks = observed_masks.clone()
        masks[-1] = False
        gt_masks = masks.float()
        observed_data = torch.nan_to_num(observed_data)
        observed_masks = observed_masks.float()

        observed_news = self.tensor_news[ind: ind+self.seq_len]

        instance_data = {
            "gt_masks": gt_masks,
            "observed_data": observed_data,
            "observed_masks": observed_masks,
            "observed_news": observed_news,
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

    def _limit_index(self, X, limit):
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
        return X
