from .base_dataset import BaseDataset
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import torch


class StocksDataset(BaseDataset):
    def __init__(
        self,
        stocks_list: str,
        stocks_dir: str,
        num_train_stocks: int=300,
        *args, 
        **kwargs) -> None:

        data = []
        min_dates = []
        max_dates = []
        existing = []
        with open(stocks_list, 'r') as f:
            for stock in f.readlines():
                stock = stock.strip()
                stock_data = pd.read_csv(f"{stocks_dir}/{stock}.csv")
                dates = pd.to_datetime(stock_data['date'])
                min_dates.append(dates.min())
                max_dates.append(dates.max())
                existing.append(stock)

        self.num_train_stocks = num_train_stocks
        self.stocks_dir = stocks_dir

        existing = np.array(existing)
        min_dates = np.array(min_dates)
        max_dates = np.array(max_dates)
        idxs = min_dates.argsort()[:num_train_stocks]
        sorted_stocks = existing[idxs]
        min_date = min_dates[idxs][-1]
        max_date = np.min(max_dates[idxs])

        stock = existing[0]
        stock_data = pd.read_csv(f"{stocks_dir}/{stock}.csv")
        dates = pd.to_datetime(stock_data['date'])
        stock_data = stock_data[(min_date <= dates) & (dates <= max_date)]
        observed_dates = pd.to_datetime(stock_data.date)
        start_date = observed_dates.min()

        total = pd.DataFrame(stock_data.date.tolist(), columns=['date'])
        for stock in sorted_stocks:
            stock_data = pd.read_csv(f"{stocks_dir}/{stock}.csv")
            dates = pd.to_datetime(stock_data['date'])
            stock_data = stock_data[(min_date <= dates) & (dates <= max_date)]
            stock_data = stock_data[['date', 'close']].rename(columns={'close': stock})
            total = total.merge(stock_data, how='inner', on='date')

        X = np.log(total.drop(columns='date')).diff(-1).to_numpy()
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)[:-1][::-1]

        pos_dates = (observed_dates - start_date).dt.total_seconds() / (24 * 3600)
        timestamps = pos_dates[::-1].to_numpy()

        self.start_date = start_date
        self.means = scaler.mean_
        self.stds = scaler.scale_

        super().__init__(X_normalized, timestamps, *args, **kwargs)
