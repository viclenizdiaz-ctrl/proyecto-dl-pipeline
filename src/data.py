"""Carga del dataset, split train/val y vectorización TF-IDF."""
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from src.config import DATA_PATH, Config


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    input_dim: int
    classes: List[str]
    class_weights: torch.Tensor
    n_train: int
    n_val: int


def build_dataloaders(cfg: Config) -> DataBundle:
    df = pd.read_csv(DATA_PATH).dropna(subset=["titulo", "categoria"])

    classes = sorted(df["categoria"].unique())
    label2idx = {c: i for i, c in enumerate(classes)}
    y = df["categoria"].map(label2idx).to_numpy()

    # Split estratificado: cada clase conserva su proporción en train y val.
    x_tr_txt, x_val_txt, y_tr, y_val = train_test_split(
        df["titulo"].to_numpy(),
        y,
        test_size=cfg.val_size,
        stratify=y,
        random_state=cfg.seed,
    )

    # El vectorizador se ajusta SOLO con train para evitar fuga de información.
    vectorizer = TfidfVectorizer(
        max_features=cfg.max_features,
        ngram_range=(1, 2),
        min_df=2,
        strip_accents="unicode",
        sublinear_tf=True,
    )
    x_tr = vectorizer.fit_transform(x_tr_txt).toarray().astype(np.float32)
    x_val = vectorizer.transform(x_val_txt).toarray().astype(np.float32)

    # Pesos inversamente proporcionales a la frecuencia de cada clase.
    weights = compute_class_weight("balanced", classes=np.arange(len(classes)), y=y_tr)

    train_ds = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr).long())
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val).long())

    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=x_tr.shape[1],
        classes=classes,
        class_weights=torch.tensor(weights, dtype=torch.float32),
        n_train=len(train_ds),
        n_val=len(val_ds),
    )
