from pathlib import Path
from sklearn.preprocessing import StandardScaler
import hydra
import json
import numpy as np
import pandas as pd


@hydra.main(version_base=None, config_path="src/configs", config_name="preprocess")
def main(config):
    data = []
    min_dates = []
    max_dates = []
    existing = []

    config = config.vars
    num_train_stocks = config.num_train_stocks
    stocks_dir = config.stocks_dir
    stocks_list = config.stocks_list
    save_dir = Path(config.save_dir) / 'preprocessed'
    save_dir.mkdir(parents=True, exist_ok=True) 

    with open(stocks_list, 'r') as f:
        for stock in f.readlines():
            stock = stock.strip()
            stock_data = pd.read_csv(f"{stocks_dir}/{stock}.csv")
            dates = pd.to_datetime(stock_data['date'])
            min_dates.append(dates.min())
            max_dates.append(dates.max())
            existing.append(stock)

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
    X_normalized = scaler.fit_transform(X)[:-1][::-1].copy()

    pos_dates = (observed_dates - start_date).dt.total_seconds() / (24 * 3600)
    timestamps = pos_dates[::-1].to_numpy().copy()

    to_save = {}
    to_save['start_date'] = str(start_date)
    
    np.save(save_dir / 'timestamps.npy', timestamps)
    np.save(save_dir / 'X.npy', X_normalized)
    np.save(save_dir / 'means.npy', scaler.mean_)
    np.save(save_dir / 'stds.npy', scaler.scale_)
    with open(save_dir / 'start_date.json', 'w', encoding='utf8') as f:
        json.dump(to_save, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()