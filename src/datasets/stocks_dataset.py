from .base_dataset import BaseDataset
from pathlib import Path
import json
import numpy as np
import pandas as pd


class StocksDataset(BaseDataset):
    def __init__(
        self,
        data_dir: str,
        start_ratio: float,
        end_ratio: float,
        time_window: int=1,
        *args, 
        **kwargs) -> None:

        data_dir = Path(data_dir) / "preprocessed"
        assert data_dir.exists(),(
        f"{data_dir} doesn't exists."
        "Maybe you forgot to run preprocess.py?")

        # X_normalized = np.load(data_dir / 'X.npy')
        # timestamps = np.load(data_dir / 'timestamps.npy')
        X = pd.read_csv(data_dir / 'X.csv')
        X['Period'] = X['Day'] // time_window
        X['Period'] = X["Year"] * 1000 + X['Period']

        agg_dict = {}
        for column in X.columns:
            if column.endswith('_open'):
                agg_dict[column] = 'first'
            elif column.endswith('_close'):
                agg_dict[column] = 'last'

        X = X.groupby('Period').agg(agg_dict).reset_index()
        stocks = list(set(col.split('_')[0] for col in X.columns if '_' in col))

        for stock in stocks:
            open_col = f"{stock}_open"
            close_col = f"{stock}_close"
            X[f"{stock}"] = np.log(X[close_col]) - np.log(X[open_col])
            X.drop(columns=[open_col, close_col], inplace=True)

        news = pd.read_csv(data_dir / 'news.csv')
        news['Period'] = news['Day'] // time_window
        news['Period'] = news["Year"] * 1000 + news['Period']
        news['Embeddings'] = news['Embeddings'].apply(json.loads)

        start = int(start_ratio * len(X))
        end = int(end_ratio * len(X))

        timestamps = X.Period.iloc[start:end].values
        X = X[stocks].iloc[start:end].values
        common_news = news[news.Stock_symbol.isna()]
        news = news[~news.Stock_symbol.isna()]

        super().__init__(X, timestamps, news, common_news, stocks, *args, **kwargs)
