"""Registro de métricas y gráficos."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin interfaz, funciona en servidores y notebooks
import matplotlib.pyplot as plt
import pandas as pd


def save_history(history: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(path, index_label="epoch")


def plot_history(history: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="validación")
    ax1.set(title="Pérdida por época", xlabel="Época", ylabel="CrossEntropy")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="validación")
    ax2.set(title="Accuracy por época", xlabel="Época", ylabel="Accuracy")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
