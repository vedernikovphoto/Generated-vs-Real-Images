import pytorch_lightning as pl
import torch
import logging
from timm import create_model
from src.config import Config
from src.losses import get_losses
from src.metrics import get_metrics
from utils.train_utils import load_object
from torch.nn import Sequential, Dropout, Linear


logging.basicConfig(level=logging.INFO, format='{message}', style='{')


class MaritimeModule(pl.LightningModule):
    """
    PyTorch Lightning module for the Maritime Images classification.

    Attributes:
        config (Config): Configuration object.
        model (torch.nn.Module): The model to be trained.
        losses (list): List of loss functions.
        train_metrics (torchmetrics.Metric): Train metrics.
        valid_metrics (torchmetrics.Metric): Validation metrics.
        test_metrics (torchmetrics.Metric): Test metrics.
    """
    def __init__(self, config: Config):
        """
        Initializes the MaritimeModule with the specified configuration.

        Args:
            config (Config): Configuration object containing model, optimizer, and scheduler settings.
        """
        super().__init__()
        self._config = config

        self._model = create_model(
            num_classes=1, **self._config.mdl_parameters,
        )
        in_features = self._model.classifier.in_features
        self._model.classifier = Sequential(
            Dropout(p=0.5),
            Linear(in_features, 1),
        )

        self._losses = get_losses(self._config.losses)
        metrics = get_metrics(task='binary')
        self._train_metrics = metrics.clone(prefix='train_')
        self._valid_metrics = metrics.clone(prefix='val_')
        self._test_metrics = metrics.clone(prefix='test_')

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width].

        Returns:
            torch.Tensor: Predicted logits of shape [batch_size, 1].
        """
        return self._model(x)

    def configure_optimizers(self) -> dict:
        """
        Configures the optimizers and learning rate scheduler.

        Returns:
            dict: A dictionary containing the optimizer and scheduler configurations.
        """
        optimizer = load_object(self._config.optimizer)(
            self._model.parameters(),
            **self._config.optimizer_kwargs,
        )
        scheduler = load_object(self._config.scheduler)(optimizer, **self._config.scheduler_kwargs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self._config.monitor_metric,
                'interval': 'epoch',
                'frequency': 1,
            },
        }

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Executes a single training step.

        Args:
            batch (tuple): A tuple containing images and ground truth labels.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        images, gt_labels = batch
        gt_labels = gt_labels.float().unsqueeze(1)
        pr_logits = self(images)
        loss = self._calculate_loss(pr_logits, gt_labels, 'train_')

        pr_probs = torch.sigmoid(pr_logits)
        self._train_metrics.update(pr_probs, gt_labels)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Executes a single validation step.

        Args:
            batch (tuple): A tuple containing images and ground truth labels.
            batch_idx (int): Index of the current batch.
        """
        images, gt_labels = batch
        gt_labels = gt_labels.float().unsqueeze(1)
        pr_logits = self(images)
        pr_probs = torch.sigmoid(pr_logits)
        self._valid_metrics.update(pr_probs, gt_labels)

    def test_step(self, batch, batch_idx):
        """
        Executes a single test step.

        Args:
            batch (tuple): A tuple containing images and ground truth labels.
            batch_idx (int): Index of the current batch.
        """
        images, gt_labels = batch
        gt_labels = gt_labels.float().unsqueeze(1)
        pr_logits = self(images)
        pr_probs = torch.sigmoid(pr_logits)
        self._test_metrics.update(pr_probs, gt_labels)

    def on_train_epoch_start(self):
        """
        Resets training metrics at the start of each epoch.
        """
        self._train_metrics.reset()

    def on_train_epoch_end(self):
        """
        Logs training metrics at the end of each epoch.
        """
        metrics = self._train_metrics.compute()
        self.log_dict(metrics, on_epoch=True)
        logging.info('Training Metrics:')
        for k, v in metrics.items():
            logging.info(f'{k}: {v:.4f}')

    def on_validation_epoch_start(self):
        """
        Resets validation metrics at the start of each epoch.
        """
        self._valid_metrics.reset()

    def on_validation_epoch_end(self):
        """
        Logs validation metrics at the end of each epoch.
        """
        metrics = self._valid_metrics.compute()
        self.log_dict(metrics, on_epoch=True)
        logging.info('Validation Metrics:')
        for k, v in metrics.items():
            logging.info(f'{k}: {v:.4f}')

    def on_test_epoch_start(self):
        """
        Resets test metrics at the start of testing.
        """
        self._test_metrics.reset()

    def on_test_epoch_end(self):
        """
        Logs test metrics at the end of testing.
        """
        metrics = self._test_metrics.compute()
        self.log_dict(metrics, on_epoch=True)
        logging.info('Test Metrics:')
        for k, v in metrics.items():
            logging.info(f'{k}: {v:.4f}')

    def _calculate_loss(
        self,
        pr_logits: torch.Tensor,
        gt_labels: torch.Tensor,
        prefix: str,
    ) -> torch.Tensor:
        """
        Calculates the total loss, including L1 regularization.

        Args:
            pr_logits (torch.Tensor): Predicted logits from the model.
            gt_labels (torch.Tensor): Ground truth labels of shape [batch_size, 1].
            prefix (str): Prefix for logging loss values.

        Returns:
            torch.Tensor: The total computed loss.
        """
        total_loss = 0
        for cur_loss in self._losses:
            loss = cur_loss.loss(pr_logits, gt_labels)
            total_loss += cur_loss.weight * loss
            self.log(f'{prefix}{cur_loss.name}_loss', loss.item())

        l1_lambda = self._config.regularization.l1_lambda
        l1_norm_list = [model_param.abs().sum() for model_param in self._model.parameters()]
        l1_norm = sum(l1_norm_list)
        l1_loss = l1_lambda * l1_norm

        total_loss += l1_loss
        self.log(f'{prefix}l1_loss', l1_loss.item())

        self.log(f'{prefix}total_loss', total_loss.item())
        return total_loss
