from dataclasses import dataclass
from torch import nn
from typing import List

from src.config import LossConfig
from utils.train_utils import load_object


@dataclass
class Loss:
    """
    Represents a loss function with its name, weight, and implementation.

    Attributes:
        name (str): Identifier for the loss function.
        weight (float): Weight applied during loss computation.
        loss (nn.Module): Initialized loss function module.
    """
    name: str
    weight: float
    loss: nn.Module


def get_losses(losses_cfg: List[LossConfig]) -> List[Loss]:
    """
    Creates a list of loss functions from configurations.

    Args:
        losses_cfg (List[LossConfig]): Configurations specifying the loss functions.

    Returns:
        List[Loss]: Initialized loss functions with names and weights.
    """
    losses = []
    for loss_cfg in losses_cfg:
        loss_fn_class = load_object(loss_cfg.loss_fn)
        loss_instance = loss_fn_class(**loss_cfg.loss_kwargs)
        losses.append(
            Loss(
                name=loss_cfg.name,
                weight=loss_cfg.weight,
                loss=loss_instance,
            ),
        )
    return losses
