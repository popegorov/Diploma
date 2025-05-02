from .base_dataset import BaseDataset
from pathlib import Path
import numpy as np
import pandas as pd


class StocksDataset(BaseDataset):
    def __init__(
        self,
        data_dir: str,
        time_window: int,
        start_ratio: float,
        end_ratio: float,
        *args, 
        **kwargs) -> None:

        data_dir = Path(data_dir) / "preprocessed"
        assert data_dir.exists(),(
        f"{data_dir} doesn't exists."
        "Maybe you forgot to run preprocess.py?")

        # X_normalized = np.load(data_dir / 'X.npy')
        # timestamps = np.load(data_dir / 'timestamps.npy')
        X_normalized = pd.read_csv(data_dir / 'X.csv')
        X_normalized['Period'] = X_normalized['Day'] // time_window
        X_normalized['Period'] = X_normalized["Year"].astype(str) + "_" + X_normalized['period'].astype(str)
        news = pd.read_csv(data_dir / 'news.csv')
        news['Period'] = news['Day'] // time_window
        news['Period'] = news["Year"].astype(str) + "_" + news['period'].astype(str)

        start = int(start_ratio * len(X_normalized))
        end = int(end_ratio * len(X_normalized))

        X_normalized = X_normalized[start:end]
        timestamps = timestamps[start:end]

        super().__init__(X_normalized, timestamps, news, *args, **kwargs)
