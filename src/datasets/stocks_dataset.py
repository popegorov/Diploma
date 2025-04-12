from src.datasets.base_dataset import BaseDataset
from pathlib import Path
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


class StocksDataset(BaseDataset):
    def __init__(self, 
         stocks_path: str, 
         column_names:list =['close', 'volume'],
         *args, 
         **kwargs) -> None:

        data = []
        for path in Path(stocks_path).iterdir():
            entry = {}
            if path.suffix == ".csv" and path.is_file():
                entry["path"] = str(path)
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)

        self.column_names = column_names
        self.means = {}
        self.stds = {}
        self.start_dates = {}

    def load_object(self, path: str) -> tuple[torch.Tensor, torch.Tensor]:
        data_object = super().load_object(path)
        stock = path.split('.')[0]

        X = data_object[self.column_names].to_numpy()
        dates = pd.to_datetime(data_object['date'])
        start_date = dates.min()
        pos_days = (dates - start_date).dt.total_seconds() / (24 * 3600)
        timestamps = torch.tensor(pos_days.values, dtype=torch.float32)

        scaler = StandardScaler()
        X_normalized = torch.tensor(
            data=scaler.fit_transform(X), 
            dtype=torch.float32,
        )

        self.start_dates[stock] = start_date
        self.means[stock] = scaler.mean_
        self.stds[stock] = scaler.scale_
        return timestamps, X_normalized

