from functools import partial
import inspect
from typing import Dict, Any
import logging

import lightning as L
import torch
from torch.optim import Adam, AdamW, RAdam

from core.config import RunConfig
from core.tools import dynamic_import
from core.optimizers import Ranger

_logger = logging.getLogger("SynapseLogger")

class ModelModule(L.LightningModule):
    def __init__(
            self,
            run_cfg: RunConfig,
            model_class: str,
            model_params: Dict[str, Any],
            loss_fn: str, # TODO: merge loss_fn and loss_params into a single dict
            loss_params: Dict[str, Any] | None = None,
            training_step_func: str = "default",
            validation_step_func: str = "default",
            test_step_func: str = "default",
            optimizer: str = "adam",
            start_lr: float = 1e-3,
            lr_scheduler: str | None = None,
            metrics: Dict[str, dict] | None = None,
    ) :
        """
        Initialize LightningModule wrapper.

        Args:
            run_cfg (RunConfig): Configuration for the run, including batch size, epochs, etc.
            model_class (str): The fully qualified class name of the model to be used.
            model_params (dict): Model parameters to be passed to the model.
            loss_fn (str): The fully qualified name of the loss function to be used.
            loss_params (dict): Parameters for the loss function.
            training_step_func (str): Custom training step function. Default is "default".
            validation_step_func (str): Custom validation step function. Default is "default".
            test_step_func (str): Custom test step function. Default is "default".
            optimizer (str): Optimizer to be used.
            start_lr (float): Initial learning rate for the optimizer.
            lr_scheduler (str): Learning rate scheduler to be used.
            metrics (dict): Dictionary of metric names and their corresponding functions.
        """
        super().__init__()
        self.run_cfg = run_cfg

        model_cls = dynamic_import(model_class)

        if not issubclass(model_cls, torch.nn.Module):
            raise TypeError(f"model_class: {model_class} is not the subclass of torch.nn.Module")

        self.model = model_cls(**model_params)

        loss_function = dynamic_import(loss_fn)
        if inspect.isclass(loss_function) and issubclass(loss_function, torch.nn.Module):
            if loss_params:
                self.loss_fn = loss_function(**loss_params)
            else:
                self.loss_fn = loss_function()
        elif inspect.isfunction(loss_function):
            if loss_params:
                self.loss_fn = partial(loss_function, **loss_params)
            else:
                self.loss_fn = loss_function
        else:
            raise TypeError(f"loss_function: {loss_fn} is not a valid loss function or class")

        self._training_step = self._default_training_step
        if training_step_func != "default":
            training_step_fn = dynamic_import(training_step_func)
            if inspect.isfunction(training_step_fn):
                self._training_step = training_step_fn
            else:
                raise TypeError(f"Invalid training step function: {training_step_func}")

        self._validation_step = self._default_validation_step
        if validation_step_func != "default":
            validation_step_fn = dynamic_import(validation_step_func)
            if inspect.isfunction(validation_step_fn):
                self._validation_step = validation_step_fn
            else:
                raise TypeError(f"Invalid validation step function: {validation_step_func}")

        self._test_step = self._default_test_step
        if test_step_func != "default":
            test_step_fn = dynamic_import(test_step_func)
            if inspect.isfunction(test_step_fn):
                self._test_step = test_step_fn
            else:
                raise TypeError(f"Invalid test step function: {test_step_func}")

        self.optimizer = optimizer.lower()
        self.start_lr = start_lr
        self.lr_scheduler = lr_scheduler.lower()

        self.metrics = {}
        if metrics:
            for metric_name, metric_fn_dict in metrics.items():
                metric_fn = dynamic_import(metric_fn_dict['function'])
                if inspect.isclass(metric_fn) and issubclass(metric_fn, torch.nn.Module):
                    if metric_fn_dict.get('params'):
                        self.metrics[metric_name] = metric_fn(**metric_fn_dict['params'])
                    else:
                        self.metrics[metric_name] = metric_fn()
                elif inspect.isfunction(metric_fn):
                    if metric_fn_dict.get('params'):
                        self.metrics[metric_name] = partial(metric_fn, **metric_fn_dict['params'])
                    else:
                        self.metrics[metric_name] = partial(metric_fn)
                else:
                    raise TypeError(f"Invalid metric function: {metric_fn}")

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Universal forward method to call the model.
        """
        return self.model(*args, **kwargs)

    def _shared_default_step(self, batch: Any, stage: str) -> torch.Tensor:
        x, y, w, s = batch
        inputs = [x[k] for k in x.keys()]
        labels = [y[k] for k in y.keys()]
        weight = [w[k] for k in w.keys()][0]

        outputs = self(*inputs)

        loss = self.loss_fn(outputs, *labels, weight)

        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # _logger.info(f"{stage}_loss: {loss}")
        for metric_name, metric_fn in self.metrics.items():
            # FIXME: some metrics need to consider the weight
            metric = metric_fn(outputs, *labels)
            self.log(f"{stage}_{metric_name}", metric, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            # _logger.info(f"{stage}_{metric_name}: {metric}")

        return loss

    def _default_training_step(self, batch):
        return self._shared_default_step(batch, "train")

    def _default_validation_step(self, batch):
        return self._shared_default_step(batch, "val")

    def _default_test_step(self, batch):
        return self._shared_default_step(batch, "test")

    def training_step(self, batch, batch_idx):
        return self._training_step(batch)

    def validation_step(self, batch, batch_idx):
        return self._validation_step(batch)

    def test_step(self, batch, batch_idx):
        return self._test_step(batch)

    # def on_validation_epoch_end(self):
    #     val_loss = self.trainer.callback_metrics["val_loss"].item()

    def configure_optimizers(self):
        # TODO: record LR
        optimizer_type = self.optimizer
        lr_scheduler_type = self.lr_scheduler
        lr = self.start_lr
        # TODO: make optimizer and scheduler parameters configurable, following the same way as loss function loading
        if optimizer_type == "adam":
            optimizer = Adam(self.parameters(), lr=lr)
        elif optimizer_type == "adamw":
            optimizer = AdamW(self.parameters(), lr=lr)
        elif optimizer_type == "radam":
            optimizer = RAdam(self.parameters(), lr=lr)
        elif optimizer_type == "ranger":
            optimizer = Ranger(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        scheduler_dict = {}
        if lr_scheduler_type:
            if lr_scheduler_type == "steps":
                lr_step = round(self.run_cfg.epochs / 3)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=[lr_step, 2 * lr_step], gamma=0.1,
                    last_epoch=-1) # FIXME: last_epoch=-1 means it will start from the beginning, fix after implementing checkpoint loading
                scheduler_dict = {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1
                }
            elif lr_scheduler_type == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.run_cfg.epochs, eta_min=0.0, last_epoch=-1)
                scheduler_dict = {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1
                }
            elif lr_scheduler_type == "one-cycle":
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=lr, total_steps=self.run_cfg.epochs * self.run_cfg.batch_size,
                    pct_start=0.3, anneal_strategy='linear', cycle_momentum=False, last_epoch=-1)
                scheduler_dict = {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }


        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_dict
        }
