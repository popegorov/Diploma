from hydra.utils import instantiate
from torch import nn
import torch


class BaseLSTMModel(nn.Module):
    def __init__(
        self, 
        lstm_model_config: dict,
        loss_type: str="mse",
        device: str="cpu",
    ) -> None:
        super().__init__()
        self.device = device

        lstm_model_config['_target_'] = 'src.model.PriceNewsLSTMModel'
        self.model = instantiate(lstm_model_config)
        self.criterion = nn.MSELoss() if loss_type == "mse" else nn.L1Loss()

    def forward(
        self,
        observed_data: torch.Tensor,
        observed_news: torch.Tensor,
        **batch) -> dict:
        """
        Model forward
        Args:
            observed_data (torch.Tensor): ground truth data
            observed_news (torch.Tensor): news embeddings
        Returns:
            loss (dict): predictions dict with loss
        """
        to_predict = observed_data[:,:,-1]
        data = observed_data[:,:,:-1]

        output = self.model(data, observed_news)

        loss = self.criterion(to_predict, output)
        observed_data_modified = observed_data.clone()
        observed_data_modified[:, :, -1] = output

        return {"loss": loss, "predicted": observed_data_modified}
        