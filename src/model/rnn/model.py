import torch
import torch.nn as nn

class PriceNewsLSTMModel(nn.Module):
    def __init__(
            self, 
            price_hidden_size: int=64, 
            news_hidden_size: int=64, 
            mixed_hidden_size: int=128, 
            news_input_dim: int=768):
        super().__init__()

        self.price_lstm = nn.LSTM(input_size=1, hidden_size=price_hidden_size, batch_first=True)
        self.news_lstm = nn.LSTM(input_size=news_input_dim, hidden_size=news_hidden_size, batch_first=True)

        self.mixed_linear = nn.Sequential(
            nn.Linear(price_hidden_size + news_hidden_size, mixed_hidden_size),
            nn.ReLU(),
            nn.Linear(mixed_hidden_size, 1)
        )

    def forward(
        self,
        observed_data: torch.Tensor,
        observed_news: torch.Tensor
    ):
        """
        Forwards data
        Args:
            observed_data (torch.Tensor): batch
            observed_news (torch.Tensor): side news information
        Returns:
            out (torch.Tensor): predictions
        """
        B, K, L = observed_data.size()
        news_dim = observed_news.size(-1)

        observed_data = observed_data.unsqueeze(-1)  # (B, K, L, 1)
        observed_data = observed_data.reshape(-1, L, 1)
        _, (price_h_n, _) = self.price_lstm(observed_data)
        price_features = price_h_n.squeeze(0)

        observed_news = observed_news.reshape(-1, L+1, news_dim)  # (B * K, L, En)
        _, (news_h_n, _) = self.news_lstm(observed_news)
        news_features = news_h_n.squeeze(0)

        combined = torch.cat([price_features, news_features], dim=-1)
        out = self.mixed_linear(combined)  # (B * K, 1)

        out = out.reshape(B, K)
        return out
