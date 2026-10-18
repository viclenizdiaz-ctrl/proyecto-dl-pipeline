"""Genera data/dataset_1.csv a partir del export de Mercado Libre (data/raw/*.xlsx).

Uso:
    python -m src.prepare_dataset data/raw/INTEGRALY_17-10.xlsx
"""
import sys
from pathlib import Path

import pandas as pd

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "dataset_1.csv"

# Categoría de segundo nivel de Mercado Libre -> etiqueta corta
CATEGORIAS = {
    "Hogar, Muebles y Jardín > Iluminación para el Hogar": "iluminacion",
    "Belleza y Cuidado Personal > Artefactos para el Cabello": "artefactos_cabello",
    "Hogar, Muebles y Jardín > Bazar y Cocina": "bazar_cocina",
    "Construcción > Baños y Sanitarios": "banos_sanitarios",
    "Hogar, Muebles y Jardín > Muebles para el Hogar": "muebles",
    "Hogar, Muebles y Jardín > Jardin y Aire Libre": "jardin_aire_libre",
    "Electrodomésticos y Aires Ac. > Pequeños Electrodomésticos": "peq_electrodomesticos",
    "Herramientas > Herramientas Eléctricas": "herramientas_electricas",
}


def main(xlsx_path: str) -> None:
    df = pd.read_excel(xlsx_path, usecols=["Id", "Categoría", "Título"]).dropna()
    df["titulo"] = df["Título"].str.strip().str.replace(r"\s+", " ", regex=True)
    df["id_base"] = df["Id"].astype(str).str.split("-").str[0]
    nivel2 = df["Categoría"].str.split(" > ").str[:2].str.join(" > ")
    df = df[nivel2.isin(CATEGORIAS)].copy()
    df["categoria"] = nivel2[df.index].map(CATEGORIAS)

    # Se descartan títulos que aparecen en más de una clase (etiqueta ambigua)
    n_clases = df.groupby("titulo")["categoria"].nunique()
    df = df[df["titulo"].isin(n_clases[n_clases == 1].index)]

    # Una fila por título (las variantes de color/talle repiten el título)
    df = df.drop_duplicates("titulo")[["id_base", "titulo", "categoria"]]
    df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"{len(df)} títulos guardados en {OUT_PATH}")
    print(df["categoria"].value_counts().to_string())


if __name__ == "__main__":
    main(sys.argv[1])
