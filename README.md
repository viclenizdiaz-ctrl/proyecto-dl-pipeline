# Clasificador de categorías de productos (Mercado Libre) — Pipeline base en PyTorch

Pre-entrega del Proyecto Final: pipeline de entrenamiento y validación con un clasificador base.
El objetivo no es la complejidad del modelo sino la calidad del pipeline.

## Problema y dataset
Dado el **título** de una publicación de Mercado Libre, predecir su **categoría** (clasificación multiclase, 8 clases).

- Fuente: export de publicaciones propias (`INTEGRALY_17-10.xlsx`). El Excel original **no se versiona** (`data/raw/` está en `.gitignore`) porque contiene datos comerciales.
- `data/dataset_1.csv`: 3.556 títulos únicos con columnas `id_base`, `titulo`, `categoria`. Se genera con `src/prepare_dataset.py`.
- Las clases son categorías de segundo nivel de Mercado Libre. Se descartó "Hogar > Baños" por solaparse con "Construcción > Baños y Sanitarios".

| Clase | Títulos |
|---|---|
| iluminacion | 655 |
| artefactos_cabello | 645 |
| bazar_cocina | 637 |
| banos_sanitarios | 454 |
| muebles | 399 |
| jardin_aire_libre | 310 |
| peq_electrodomesticos | 299 |
| herramientas_electricas | 157 |

El desbalance se compensa con pesos por clase en `CrossEntropyLoss` (`compute_class_weight("balanced")`).

## Estructura
```
├── data/
│   ├── raw/                 # Excel original (ignorado en git)
│   └── dataset_1.csv
├── src/
│   ├── config.py            # dispositivo, semillas, hiperparámetros
│   ├── prepare_dataset.py   # Excel -> dataset_1.csv
│   ├── data.py              # split estratificado + TF-IDF + DataLoaders
│   ├── model.py             # MLPClassifier (nn.Module)
│   ├── train.py             # training loop + evaluate()
│   └── utils.py             # historial y gráficos
├── notebooks/
│   └── pipeline_colab.ipynb # notebook ejecutado en Google Colab
├── outputs/                 # curvas e historial de entrenamiento
├── requirements.txt
└── README.md
```

## Cómo ejecutarlo
**Google Colab:** abrí `notebooks/pipeline_colab.ipynb`, guardá `data/dataset_1.csv` en tu Google Drive (`Mi unidad/proyecto-dl-pipeline/data/`), ajustá `DATA_PATH` si hace falta y ejecutá las celdas en orden.

**En local:**
```bash
python -m venv .venv && source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.train                  # parámetros por defecto
python -m src.train --epochs 40 --lr 0.0005
```
(Opcional) regenerar el dataset: `python -m src.prepare_dataset data/raw/INTEGRALY_17-10.xlsx`

## Diseño del pipeline
- **Dispositivo**: detección automática CUDA → MPS → CPU. Semillas fijadas en Python, NumPy y PyTorch (`seed=42`).
- **Entrada**: TF-IDF (hasta 5.000 features, unigramas y bigramas), ajustado **solo con train** para evitar fuga de información.
- **Split**: 80 % train / 20 % validación, estratificado por clase.
- **Modelo**: MLP `n_features → 128 → ReLU → Dropout(0.3) → 8` (devuelve logits).
- **Entrenamiento**: Adam, `CrossEntropyLoss` ponderada, batch de 64, 30 épocas. Ciclo explícito: `zero_grad()` → forward → loss → `backward()` → `step()`.
- **Validación**: `evaluate()` separada, con `model.eval()` y `torch.no_grad()`. Registra loss, Accuracy y F1 macro por época.
- La accuracy de *train* se calcula durante la época con Dropout activo, por lo que es levemente pesimista respecto de la de validación.

## Resultados del experimento inicial

- **Versión de PyTorch**: 2.11.0+cpu
- **Dispositivo usado**: cpu
- **learning_rate**: 1e-3. Es el valor estándar de partida para Adam; con un MLP chico y batches de 64 permitió converger en pocas épocas sin oscilaciones.
- **Épocas**: 30 · **Batch size**: 64 · **Optimizador**: Adam
- **Pérdida de entrenamiento**: 1.9519 → 0.0036
- **Pérdida de validación**: 1.6877 → 0.0738
- **Mejor época (menor val loss)**: 29

**Métricas finales del mejor modelo (712 títulos de validación)**

| Métrica | Valor |
|---|---|
| Val loss | 0.0737 |
| Accuracy | 0.986 |
| F1 macro | 0.979 |

![Curvas de pérdida y accuracy](outputs/curves.png)

### Interpretación de la curva de pérdida
La pérdida bajó de forma sostenida tanto en entrenamiento (de 1.95 a 0.004) como en validación (de 1.69 a 0.074). La caída más fuerte ocurre en las primeras 10 épocas y después ambas curvas se aplanan. La pérdida de validación no aumentó en ningún momento, por lo que no se observa overfitting severo; sí hay una pequeña brecha entre train y validación (0.004 vs 0.074), esperable en un MLP con 5.000 features y ~2.800 ejemplos de entrenamiento. Al inicio la pérdida de validación es menor que la de train porque la de train se promedia durante la época, mientras el modelo todavía está aprendiendo y con Dropout activo.

La accuracy de validación superó 0.96 desde la primera época y se estabilizó en torno a 0.986. Los errores se concentran en las clases con menos datos: `herramientas_electricas` (recall 0.903, 31 ejemplos) y `jardin_aire_libre` (F1 0.952). El resultado es consistente con la sencillez de la tarea: una regresión logística sobre el mismo TF-IDF alcanza ≈0.985, lo que indica que el MLP aprende correctamente pero que el dataset es poco exigente.

## Limitaciones y próximos pasos
- Un mismo producto puede aparecer con títulos casi idénticos; si caen en train y validación, la accuracy se infla un poco.
- Próximos checkpoints: embeddings/secuencias en lugar de TF-IDF, y categorías de nivel más profundo.