import json
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.utils.io_utils import ROOT_PATH

class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.trainer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.trainer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(config.trainer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            if part == "train":
                continue
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch = self.move_batch_to_device(batch)
        # batch = self.transform_batch(batch)  # transform batch on device -- faster

        batch['is_train'] = False

        outputs = self.model(**batch)
        batch.update(outputs)

        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        self.evaluation_metrics.reset()

        # create Save dir
        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        stocks = dataloader.dataset.observed_stocks
        predicted_series = []
        gt_series = []

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )
                for i in range(len(batch["predicted"])):
                    predicted = batch["predicted"][i].clone()
                    observed_data = batch["observed_data"][i].clone()
                    gt_mask = batch["gt_masks"][i].clone()
                    gt_series.append(observed_data[:,-1])
                    if gt_mask[:,-1].any():
                        predicted_series.append(torch.zeros(gt_mask.shape[1]))
                    else:
                        predicted_series.append(predicted[:,-1])
                
        predicted_series = torch.stack(predicted_series, dim=0) # LxK
        gt_series = torch.stack(gt_series, dim=0) # LxK
        r2s = {}

        timestamps = torch.arange(len(predicted_series))
        if self.save_path is not None:
            for i, stock in enumerate(stocks):
                preds = predicted_series[:,i].detach().cpu().numpy()
                gts = gt_series[:,i].detach().cpu().numpy()
 
                r2s[stock] = 1.0 - (((gts - preds)**2).sum() / ((gts - gts.mean())**2).sum()).item()
                plt.figure()
                plt.title(stock)
                sns.lineplot(x=timestamps, y=preds, label="Predicted")
                sns.lineplot(x=timestamps, y=gts, label="True").set(xlabel="Time", ylabel="Price log difference")
                plt.savefig(f"{self.save_path / stock}.png", dpi=300, bbox_inches='tight')
                plt.close()

        with open(self.save_path / "real_r2s.json", 'w') as f:
            json.dump(r2s, f)

        print("Real mean R^2:", sum(r2s.values()) / len(r2s))

        return self.evaluation_metrics.result()
