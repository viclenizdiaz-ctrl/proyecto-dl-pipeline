"""Arquitectura base: MLP de una capa oculta para clasificación multiclase."""
import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """input_dim -> hidden_dim -> n_classes.

    Devuelve logits (sin softmax), que es lo que espera nn.CrossEntropyLoss.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
