"""Pipeline de entrenamiento y validación.

Uso:
    python -m src.train
    python -m src.train --epochs 40 --lr 0.0005
"""
import argparse

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score

from src.config import OUTPUT_DIR, Config, get_device, set_seed
from src.data import build_dataloaders
from src.model import MLPClassifier
from src.utils import plot_history, save_history


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Una época de entrenamiento: forward, pérdida, backward y actualización."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()           # 1. reiniciar gradientes acumulados
        logits = model(xb)              # 2. forward pass
        loss = criterion(logits, yb)    # 3. cálculo de la pérdida
        loss.backward()                 # 4. backward pass (autograd)
        optimizer.step()                # 5. actualización de pesos con Adam

        running_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        total += xb.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evalúa en datos no vistos: sin gradientes y con el modelo en modo eval."""
    model.eval()
    running_loss, total = 0.0, 0
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        running_loss += loss.item() * xb.size(0)
        total += xb.size(0)
        y_true.extend(yb.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / total
    metrics = {
        "loss": running_loss / total,
        "accuracy": accuracy,
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }
    return metrics, y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.epochs:
        cfg.epochs = args.epochs
    if args.lr:
        cfg.learning_rate = args.lr
    if args.batch_size:
        cfg.batch_size = args.batch_size

    set_seed(cfg.seed)
    device = get_device()
    print(f"PyTorch {torch.__version__} | dispositivo: {device}")

    data = build_dataloaders(cfg)
    print(f"Train: {data.n_train} | Validación: {data.n_val} | "
          f"Features: {data.input_dim} | Clases: {len(data.classes)}")

    model = MLPClassifier(data.input_dim, cfg.hidden_dim, len(data.classes), cfg.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=data.class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    history = {k: [] for k in
               ["train_loss", "train_acc", "val_loss", "val_acc", "val_f1"]}
    best_val_loss = float("inf")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUTPUT_DIR / "best_model.pt"

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, data.train_loader, criterion, optimizer, device)
        val, _, _ = evaluate(model, data.val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["accuracy"])
        history["val_f1"].append(val["f1_macro"])

        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            torch.save(model.state_dict(), ckpt_path)

        print(f"Época {epoch:02d}/{cfg.epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {val['loss']:.4f} acc {val['accuracy']:.3f} f1 {val['f1_macro']:.3f}")

    save_history(history, OUTPUT_DIR / "history.csv")
    plot_history(history, OUTPUT_DIR / "curves.png")

    # Reporte final con el mejor modelo (menor pérdida de validación).
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    val, y_true, y_pred = evaluate(model, data.val_loader, criterion, device)
    report = classification_report(y_true, y_pred, target_names=data.classes, digits=3)
    (OUTPUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    print(f"\nMejor modelo -> val loss {val['loss']:.4f} | acc {val['accuracy']:.3f} "
          f"| F1 macro {val['f1_macro']:.3f}\n")
    print(report)


if __name__ == "__main__":
    main()
