from .base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
import torch


class StocksDataset(BaseDataset):
    def get_common_news(
        self,
        timestamp: int, 
        embedding_size: int) -> torch.Tensor:
        """
        Gets general news about market
        Args:
            timestamp (int): observing period
            embedding_size (int): text embedding dimension
        Returns:
            tensor (torch.Tensor): required embeddings
        """

        filtered_news = self.common_news[self.common_news['Period'] == timestamp].copy()
        to_return = filtered_news.Embeddings.tolist()
        return to_return if to_return else torch.zeros(embedding_size, dtype=torch.float32)

    def get_observed_news(
        self,
        timestamp: int,
        embedding_size: int,
        observed_stocks: list) -> torch.Tensor:
        """
        Gets most relevant news for each stock on given period
        Args:
            timestamp (int): observing period
            embedding_size (int): text embedding dimension
            observed_stocks (list): list of needed stocks
        Returns:
            tensor (torch.Tensor): tensor with news embeddings
        """

        filtered_news = self.news[self.news['Period'] == timestamp].copy()
        embeds = (
            filtered_news
            .sort_values('Abs_Score', ascending=False)
            .groupby('Stock_symbol')
            .head(5)
            .groupby('Stock_symbol')
            .agg({"Embeddings": list})
        ).reset_index()

        stocks = embeds['Stock_symbol'].unique()

        tensor = torch.zeros((len(observed_stocks), embedding_size), dtype=torch.float32)
        for i, stock in enumerate(observed_stocks):
            if stock in stocks:
                embedding = filtered_news[filtered_news.Stock_symbol == stock].Embeddings.tolist()
            else:
                embedding = self.get_common_news(timestamp, embedding_size)
            tensor[i] = torch.tensor(
                data=embedding, 
                dtype=torch.float32,
            ).mean(axis=0).clone().detach() # getting average embedding for stock in period
        return tensor
        
    def __init__(
        self,
        data_dir: str,
        start_ratio: float,
        end_ratio: float,
        embedding_size: int=768,
        time_window: int=1,
        *args, 
        **kwargs) -> None:
        """
        Dataset class
        Args:
            data_dir (str): directory with preprocessed data
            start_ratio (float): where dataset starts, using to split train and val
            end_ratio (float): where dataset ends
            embedding_size (int): text embedding dimension
            time_window (int): size of window to observe data
        """

        data_dir = ROOT_PATH / Path(data_dir) / "preprocessed"
        assert data_dir.exists(),(
        f"{data_dir} doesn't exists."
        "Maybe you forgot to run preprocess.py?")

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

        self.news = pd.read_csv(data_dir / 'news_to_observe.csv')
        self.news['Period'] = self.news['Day'] // time_window
        self.news['Period'] = self.news["Year"] * 1000 + self.news['Period']
        self.news['Embeddings'] = self.news['Embeddings'].apply(json.loads)

        self.common_news = pd.read_csv(data_dir / 'unlabeled_news.csv')
        self.common_news['Period'] = self.common_news['Day'] // time_window
        self.common_news['Period'] = self.common_news["Year"] * 1000 + self.common_news['Period']
        self.common_news['Embeddings'] = self.common_news['Embeddings'].apply(json.loads)
        self.common_news = self.common_news.groupby('Period').apply(lambda x: x.nlargest(5, 'Abs_Score')).reset_index(drop=True)

        start = int(start_ratio * len(X))
        end = int(end_ratio * len(X))

        timestamps = X.Period.iloc[start:end].values
        X = X[stocks].iloc[start:end].values
        tensor_news = torch.zeros((len(timestamps), len(stocks), embedding_size), dtype=torch.float32)

        for i, timestamp in tqdm(enumerate(timestamps), total=len(timestamps), desc="Preparing news"):
            tensor_news[i] = self.get_observed_news(
                timestamp=timestamp, 
                embedding_size=embedding_size,
                observed_stocks=stocks,
            ).clone().detach()

        super().__init__(X, tensor_news, stocks, *args, **kwargs)
