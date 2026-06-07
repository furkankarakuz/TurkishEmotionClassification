import torch
import numpy as np
from torch import nn
from transformers import Trainer
from typing import Any, Optional, Union
from sklearn.utils.class_weight import compute_class_weight


class WeightedTrainer(Trainer):
    """A ``Trainer`` that applies class weights to the cross-entropy loss.

    This is useful for imbalanced classification: rarer classes are given a larger weight so the model is not dominated by the majority class. The weights are passed in at construction time instead of relying on a global variable, and are moved to the labels' device automatically at loss time.
    """

    def __init__(self, *args: Any, class_weights: Optional[torch.Tensor] = None, **kwargs: Any) -> None:
        """Initialize the trainer.

        Args:
            *args: Positional arguments forwarded to ``transformers.Trainer``.
            class_weights: Optional 1-D tensor of per-class weights with shape ``(num_classes,)``. If ``None``, an unweighted cross-entropy loss is used.
            **kwargs: Keyword arguments forwarded to ``transformers.Trainer``.
        """
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model: nn.Module, inputs: dict[str, torch.Tensor], return_outputs: bool = False, **kwargs: Any) -> Union[torch.Tensor, tuple[torch.Tensor, Any]]:
        """Compute the (optionally class-weighted) cross-entropy loss.

        Args:
            model: The model being trained.
            inputs: Batch dictionary. The ``"labels"`` entry is popped out and the remaining entries are passed to the model's forward call.
            return_outputs: If ``True``, also return the model outputs alongside the loss.
            **kwargs: Extra keyword arguments passed by newer ``Trainer`` versions (e.g. ``num_items_in_batch``); accepted and ignored.

        Returns:
            The scalar loss tensor, or a tuple ``(loss, outputs)`` if ``return_outputs`` is ``True``.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        weight = self.class_weights.to(labels.device) if self.class_weights is not None else None
        loss = nn.CrossEntropyLoss(weight=weight)(outputs.logits, labels)

        return (loss, outputs) if return_outputs else loss


def compute_class_weights(labels: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    """Compute balanced class weights for a weighted loss.

    Uses scikit-learn's "balanced" heuristic, which assigns each class a weight inversely proportional to its frequency, helping the model pay more attention to under-represented classes on imbalanced datasets.

    Args:
        labels: 1-D array of integer-encoded training labels.
        num_classes: Total number of classes.
        device: Device the resulting weight tensor should live on; must match the model's device.

    Returns:
        A float tensor of shape ``(num_classes,)`` holding the per-class weights, placed on ``device``.
    """
    weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=labels)
    return torch.tensor(weights, dtype=torch.float).to(device)
