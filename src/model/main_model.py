from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn

import numpy as np
import torch

class BaseDiffModel(nn.Module):
    def __init__(
        self, 
        target_dim: int, 
        time_embed_dim: int,
        feature_embed_dim: int,
        num_steps: int,
        n_samples: int,
        is_unconditional: int,
        beta_start: float,
        beta_end: float,
        diff_model_config: dict,
        schedule: str="quad",
        target_startegy: str="random",
        device: str="cpu",
    ) -> None:
        super().__init__()
        self.target_dim = target_dim
        self.device = device
        self.n_samples = n_samples # for validation

        self.emb_time_dim = time_embed_dim
        self.emb_feature_dim = feature_embed_dim
        self.is_unconditional = is_unconditional
        self.target_strategy = target_startegy

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if not self.is_unconditional:
            self.emb_total_dim += 1
        
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, 
            embedding_dim=self.emb_feature_dim,
        )

        diff_model_config['_target_'] = 'src.model.DiffusionModel'
        diff_model_config['input_dim'] = 1 if self.is_unconditional else 2
        diff_model_config['side_dim'] = self.emb_total_dim
        self.model = instantiate(diff_model_config)

        self.num_steps = num_steps

        if schedule == "quad":
            self.beta = np.linspace(
                start=np.sqrt(beta_start),
                stop=np.sqrt(beta_end),
                num=self.num_steps,
            )**2
        elif schedule == "linear":
            self.beta = np.linspace(
                start=beta_start,
                stop=beta_end,
                num=self.num_steps
            )
        else:
            assert schedule in ["quad", "linear"], "Schedule should be 'quad' or 'linear'"

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)

        self.alpha_torch = torch.tensor(
            data=self.alpha, 
            dtype=torch.float32, 
            device=self.device,
            )
        self.alpha_torch = self.alpha_torch.unsqueeze(1).unsqueeze(1)


    def time_embedding(
        self, 
        position: torch.Tensor, 
        dim_model: int=128) -> torch.Tensor:
        position_emb = torch.zeros(
            size=[position.shape[0], position.shape[1], dim_model], 
            dtype=torch.float32,
            device=self.device,
            )
        
        position = position.unsqueeze(2)
        div_term = 1 / torch.pow(
            input=torch.tensor([10000.0]),
            exponent=torch.arange(start=0, end=dim_model, step=2).to(self.device) / dim_model,
        )

        table = position * div_term
        position_emb[..., 0::2] = torch.sin(table)
        position_emb[..., 1::2] = torch.cos(table)

        return position_emb

    def get_randmask(self, observed_mask: torch.Tensor) -> torch.Tensor:
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)

        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_test_pattern_mask(
        self, 
        observed_mask: torch.Tensor,
        test_pattern_mask: torch.Tensor) -> torch.Tensor:
        return observed_mask * test_pattern_mask

    def get_side_info(
        self,
        observed_timestamps: torch.Tensor,
        cond_mask: torch.Tensor) -> torch.Tensor:
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(
            position=observed_timestamps,
            dim_model=self.emb_time_dim
            )

        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim, device=self.device)
        ) # K, E

        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0)
        feature_embed = feature_embed.expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1) 
        side_info = side_info.transpose(1, 3) #(B, -1, K, L)

        if not self.is_unconditional:
            side_mask = cond_mask.unsqueeze(1)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def set_input_to_diffmodel(
        self,
        noisy_data: torch.Tensor,
        observed_data: torch.Tensor,
        cond_mask: torch.Tensor) -> torch.Tensor:
        if self.is_unconditional:
            total_input = noisy_data.unsqueeze(1) # (B, 1, K, L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1) # (B, 2, K, L)

        return total_input 

    def impute(
        self,
        observed_data: torch.Tensor,
        cond_mask: torch.Tensor,
        side_info: torch.Tensor,
        n_samples: int) -> torch.Tensor:
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros([B, n_samples, K, L], device=self.device)

        for i in range(n_samples):
            if self.is_unconditional:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (np.sqrt(self.alpha_hat[t]) * noisy_obs 
                        + np.sqrt(self.beta[t]) * noise)

                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)
            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional:
                    diff_input = (cond_mask * noisy_cond_history[t] 
                        + (1.0 - cond_mask) * current_sample)
                    diff_input = diff_input.unsqueeze(1)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)
                
                predicted = self.model(
                    x=diff_input, 
                    cond_info=side_info, 
                    diffusion_step=torch.tensor([t]),
                ).to(self.device)

                coef1 = 1.0 / np.sqrt(self.alpha_hat[t])
                coef2 = (1.0 - self.alpha_hat[t]) / np.sqrt((1.0 - self.alpha[t]))
                current_sample = coef1 * (current_sample - coef2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (1.0 - self.alpha[t-1]) / (1.0 - self.alpha[t]) * self.beta[t]

                    current_sample += np.sqrt(sigma) * noise

            imputed_samples[:, i] = current_sample.detach()
        
        return imputed_samples

    def calc_loss(
        self,
        observed_data: torch.Tensor,
        cond_mask: torch.Tensor,
        observed_mask: torch.Tensor,
        side_info: torch.Tensor,
        is_train: bool,
        set_t: int=-1) -> dict:

        B, K, L = observed_data.shape
        if not is_train:
            t = torch.ones(size=B, dtype=torch.long, device=self.device) * set_t
        else:
            t = torch.randint(0, self.num_steps, [B], device=self.device)
        
        current_alpha = self.alpha_torch[t]
        noise = torch.randn_like(observed_data)
        noisy_data = (torch.sqrt(current_alpha) * observed_data +
                        torch.sqrt(1.0 - current_alpha) * noise)
        
        total_input = self.set_input_to_diffmodel(
            noisy_data=noisy_data, 
            observed_data=observed_data, 
            cond_mask=cond_mask
            )
        
        predicted = self.model(total_input, side_info, t)
        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        denom = target_mask.sum()
        loss = (residual**2).sum() / (denom if denom > 0 else 1.0)

        return {"loss": loss}
        
    def calc_loss_valid(
        self,
        observed_data: torch.Tensor,
        cond_mask: torch.Tensor,
        observed_mask: torch.Tensor,
        side_info: torch.Tensor,
        is_train: bool=False) -> dict:
        
        loss_sum = 0.0

        for t in range(self.num_steps):
            loss = self.calc_loss(
                observed_data=observed_data,
                cond_mask=cond_mask,
                observed_mask=observed_mask,
                side_info=side_info,
                is_train=is_train,
                set_t=t,
            )["loss"]
            loss_sum += loss.detach()

        samples = self.impute(
            observed_data=observed_data,
            cond_mask=cond_mask,
            side_info=side_info,
            n_samples=self.n_samples,
        )

        samples_median = samples.median(dim=1)

        return {"loss": loss_sum / self.num_steps, "predicted": samples_median}

    def forward(
        self,
        is_train: bool,
        observed_data: torch.Tensor,
        observed_masks: torch.Tensor,
        observed_timestamps: torch.Tensor,
        gt_masks: torch.Tensor,
        **batch) -> dict:

        cond_mask = self.get_randmask(observed_masks) if is_train else gt_masks
        side_info = self.get_side_info(observed_timestamps, cond_mask)
        loss = self.calc_loss if is_train else self.calc_loss_valid

        return loss(observed_data, cond_mask, observed_masks, side_info, is_train)

                    