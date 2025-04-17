from .base_dataset import BaseDataset
from pathlib import Path
import numpy as np


class StocksDataset(BaseDataset):
    def __init__(
        self,
        data_dir: str,
        start_ratio: float,
        end_ratio: float,
        *args, 
        **kwargs) -> None:

        data_dir = Path(data_dir) / "preprocessed"
        assert data_dir.exists(),(
        f"{data_dir} doesn't exists."
        "Maybe you forgot to run preprocess.py?")

        X_normalized = np.load(data_dir / 'X.npy')
        timestamps = np.load(data_dir / 'timestamps.npy')

        start = int(start_ratio * len(X_normalized))
        end = int(end_ratio * len(X_normalized))

        X_normalized = X_normalized[start:end]
        timestamps = timestamps[start:end]

        super().__init__(X_normalized, timestamps, *args, **kwargs)
