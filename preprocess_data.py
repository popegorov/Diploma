from pathlib import Path
from src.utils.io_utils import ROOT_PATH
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from tqdm import tqdm
from typing import Tuple, List

import hydra
import json
import numpy as np
import pandas as pd
import torch
import warnings

warnings.filterwarnings("ignore")


def prepare_text(row):
    """
    Cropping text to required size.
    Args:
        row: row of Frame
    Returns:
        text: text to process
    """
    text = row.Lsa_summary
    if not isinstance(text, str) or len(text) > 1500:
        text = row.Article_title
    if not isinstance(text, str):
        text = ""
    return text

def get_preds_and_embeds(
        texts: list,
        is_labeled: bool=True, 
        batch_size: int=64) -> Tuple[List]:
    """
    Calculates sentiment predictions and embeddings with Finbert.
    Args:
        texts (str): list of texts
        save_dir (Path): saving directory
        is_labeled (bool): text's label indicator
        batch_size (int): size of batch
    Returns:
        scores (list): list of model confidences of predictions
        types (list): list of predicted sentiments
        embeddings (list): list of embeddings
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model_name = "ProsusAI/finbert"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    classification_model = BertForSequenceClassification.from_pretrained(model_name).to(device)
    embedding_model = BertModel.from_pretrained(model_name).to(device)

    classification_model.eval()
    embedding_model.eval()
    scores = []
    types = []
    all_embeddings = []

    label_to_type = {
        'positive': 1.0,
        'neutral': 1e-2,
        'negative': -1.0,
    }
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        
        inputs = tokenizer(
            batch, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            cls_outputs = classification_model(**inputs)
            probs = torch.softmax(cls_outputs.logits, dim=1)
            confidences, preds = torch.max(probs, dim=1)
            
            if is_labeled:
                emb_outputs = embedding_model(**inputs)
                embeddings = emb_outputs.last_hidden_state.mean(dim=1)
        
        id2label = classification_model.config.id2label
        
        for j in range(len(batch)):
            scores.append(confidences[j].item())
            types.append(label_to_type[id2label[preds[j].item()]])
        
        if is_labeled:
            all_embeddings.extend(embeddings.detach().cpu().tolist())
    
    return scores, types, all_embeddings

def preprocess_news(
    path_to_news: str,
    stocks_to_observe: list,
    stock_to_sector_path: str,
    save_dir: Path,
    start_date: pd.Timestamp, 
    end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Preprocesses given news. Cropping needed information in given period.
    Calculates year and day of the year for each observation. Calculates embeddings 
    for every relevant news.
    Args:
        path_to_news (str): path to news data
        stocks_to_observe (list): list of needed stocks
        stock_to_sector_path (str): path to stock to sector dict
        save_dir (Path): path to saving directory
        start_date (pd.Timestamp): start date of observation
        end_date (pd.Timestamp): end date of observation 
    Returns:
        news_to_observe (pd.DataFrame): Frame with labeled news
        unlabeled_news (pd.DataFrame): Frame with general news
    """
    news_data = pd.read_csv(path_to_news)

    print("Cropping data...")
    news_data.Date = pd.to_datetime(news_data.Date).dt.tz_localize(None)
    news_to_observe = news_data[(start_date <= news_data.Date) & (news_data.Date <= end_date)].reset_index(drop=True)

    news_to_observe.drop(columns=["Unnamed: 0"], inplace=True)
    news_to_observe['Year'] = news_to_observe.Date.dt.year
    news_to_observe['Day'] = news_to_observe.Date.dt.dayofyear
    print("Mapping stocks to sector...")
    with open(stock_to_sector_path, "r") as f:
        stock_to_sector = json.load(f)

    news_to_observe["Sector"] = [stock_to_sector.get(x, f'Common') for x in news_to_observe["Stock_symbol"]]

    unlabeled_news = news_to_observe[news_to_observe.Stock_symbol.isna()].reset_index(drop=True)
    news_to_observe = news_to_observe[news_to_observe.Stock_symbol.isin(stocks_to_observe)].reset_index(drop=True)

    print("Labeled data length", len(news_to_observe))
    print("Unlabeled data length", len(unlabeled_news))

    texts_to_process = [prepare_text(news_to_observe.iloc[i]) for i in range(len(news_to_observe))]
    scores, types, embeds = get_preds_and_embeds(texts_to_process, save_dir)

    news_to_observe['Score'] = scores
    news_to_observe['Type'] = types
    news_to_observe['Abs_Score'] = news_to_observe['Type'].abs() * news_to_observe['Score']
    news_to_observe['Embeddings'] = embeds
    news_to_observe['Embeddings'] = news_to_observe['Embeddings'].apply(json.dumps)

    texts_to_process = [prepare_text(unlabeled_news.iloc[i]) for i in range(len(unlabeled_news))]
    scores, types, embeds = get_preds_and_embeds(texts_to_process, False)
    unlabeled_news['Texts'] = texts_to_process
    unlabeled_news['Score'] = scores
    unlabeled_news['Type'] = types
    unlabeled_news['Abs_Score'] = unlabeled_news['Type'].abs() * unlabeled_news['Score']

    news_to_observe = news_to_observe[["Year", "Day", "Abs_Score", "Embeddings", "Sector", "Stock_symbol"]]
    unlabeled_news = unlabeled_news.groupby(['Year', 'Day']).apply(lambda x: x.nlargest(5, 'Abs_Score')).reset_index(drop=True)
    _, _, embeds = get_preds_and_embeds(unlabeled_news['Texts'].tolist())
    unlabeled_news['Embeddings'] = embeds
    unlabeled_news['Embeddings'] = unlabeled_news['Embeddings'].apply(json.dumps)

    unlabeled_news = unlabeled_news[["Year", "Day", "Abs_Score", "Embeddings"]]
    return news_to_observe, unlabeled_news

def preprocess_stocks(
    stocks_list: str, 
    stocks_dir: str,
    num_train_stocks: int):
    """
    Preprocesses required stocks from stoсk list. Choosing needed amount 
    of stocks with the longest history of observation. Calculates year 
    and day of the year for each observation.
    Args:
        stocks_list (str): path to list of required stocks
        stocks_dir (str): directory with stocks data
        num_train_stocks (int): needed amount of stocks
    Returns:
        total (pd.DataFrame): Frame with information about needed stocks
        scaler (StandardScaler): Scaler of stock information
        sorted_stocks (list): list of observed stocks
        start_date (pd.Timestamp): min observed date
        end_date (pd.Timestamp): max observed date 
    """
    min_dates = []
    existing = []
    max_date = pd.to_datetime("2023-12-28 00:00:00")

    print("Reading stocks list...")
    with open(stocks_list, 'r') as f:
        for stock in f.readlines():
            stock = stock.strip()
            stock_data = pd.read_csv(f"{stocks_dir}/{stock}.csv")
            dates = pd.to_datetime(stock_data['date'])
            if dates.max() != max_date:
                continue
            min_dates.append(dates.min())
            existing.append(stock)

    existing = np.array(existing)
    min_dates = np.array(min_dates)

    idxs = min_dates.argsort()[:num_train_stocks]
    sorted_stocks = existing[idxs]
    min_date = min_dates[idxs][-1] 

    print("Building dataset...")
    stock = existing[0]
    stock_data = pd.read_csv(f"{stocks_dir}/{stock}.csv")
    dates = pd.to_datetime(stock_data['date'])
    stock_data = stock_data[(min_date <= dates) & (dates <= max_date)]
    start_date = min_date
    end_date = max_date

    total = pd.DataFrame(stock_data.date.tolist(), columns=['date'])
    for stock in sorted_stocks:
        stock_data = pd.read_csv(f"{stocks_dir}/{stock}.csv")
        dates = pd.to_datetime(stock_data['date'])
        stock_data = stock_data[(min_date <= dates) & (dates <= max_date)]
        stock_data = stock_data[['date', 'open', 'close']].rename(columns={'open': f"{stock}_open", 'close': f"{stock}_close"})
        total = total.merge(stock_data, how='inner', on='date')

    total = total.iloc[::-1]
    total.date = pd.to_datetime(total.date)
    total['Year'] = total.date.dt.year
    total['Day'] = total.date.dt.dayofyear

    return total, None, sorted_stocks, start_date, end_date

@hydra.main(version_base=None, config_path="src/configs", config_name="preprocess")
def main(config):
    """
    Saves all preprocessed data in save directory for the future training and inference.
    Args:
        config (OmegaConf): config
    """
    config = config.vars
    num_train_stocks = config.num_train_stocks
    stocks_dir = config.stocks_dir
    stocks_list = config.stocks_list
    stock_to_sector_path = config.stock_to_sector_path
    save_dir = ROOT_PATH / config.save_dir / 'preprocessed'
    path_to_news = config.path_to_news
    save_dir.mkdir(parents=True, exist_ok=True) 

    X, scaler, sorted_stocks, start_date, end_date = preprocess_stocks(
        stocks_list=stocks_list,
        stocks_dir=stocks_dir,
        num_train_stocks=num_train_stocks,
    )

    news_to_observe, unlabeled_news = preprocess_news(
        path_to_news=path_to_news, 
        stocks_to_observe=sorted_stocks,
        stock_to_sector_path=stock_to_sector_path,
        save_dir=save_dir,
        start_date=start_date, 
        end_date=end_date,
    )
    print("Datasets are saving...")
    news_to_observe.to_csv(save_dir / 'news_to_observe.csv', index=False)
    unlabeled_news.to_csv(save_dir / 'unlabeled_news.csv', index=False)

    to_save = {}
    to_save['start_date'] = str(start_date)
    to_save['end_date'] = str(end_date)

    X.to_csv(save_dir / 'X.csv', index=False)
    if scaler is not None:
        np.save(save_dir / 'means.npy', scaler.mean_)
        np.save(save_dir / 'stds.npy', scaler.scale_)
    with open(save_dir / 'start_date.json', 'w', encoding='utf8') as f:
        json.dump(to_save, f, ensure_ascii=False, indent=4)

    print("Success!")


if __name__ == "__main__":
    main()
