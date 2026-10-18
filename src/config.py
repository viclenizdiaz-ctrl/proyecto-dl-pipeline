"""Configuración del entorno: dispositivo, semillas e hiperparámetros."""
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "dataset_1.csv"
OUTPUT_DIR = ROOT / "outputs"


@dataclass
class Config:
    seed: int = 42
    val_size: float = 0.2
    max_features: int = 5000
    hidden_dim: int = 128
    dropout: float = 0.3
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 1e-3


def get_device() -> torch.device:
    """Detecta automáticamente el mejor dispositivo: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Fija las semillas de Python, NumPy y PyTorch para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
