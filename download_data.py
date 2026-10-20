from pathlib import Path

import pandas as pd
from datasets import load_dataset

FILAS_POR_CLASE = 5000
SALIDA = Path("data/tickets_soporte.csv")

COLUMNAS = [
    "created_at",
    "category",
    "issue_type_id",
    "product",
    "channel",
    "region",
    "language",
    "customer_age",
    "customer_gender",
    "subscription_type",
    "customer_tenure_months",
    "previous_tickets",
    "customer_segment",
    "priority",
    "issue_complexity_score",
    "auto_resolved",
]

print("Conectando con Hugging Face...")

datos = load_dataset(
    "s2pidape/support-ticket-dataset",
    "with_ai",
    split="train",
    streaming=True,
)

datos = datos.shuffle(seed=42, buffer_size=10_000)

positivos = []
negativos = []

for registro in datos:
    resultado = registro.get("auto_resolved")

    if resultado is None:
        continue

    valor = str(resultado).strip().lower()
    fila = {columna: registro.get(columna) for columna in COLUMNAS}

    if valor in {"yes", "true", "1"} and len(positivos) < FILAS_POR_CLASE:
        fila["auto_resolved"] = "Sí"
        positivos.append(fila)

    elif valor in {"no", "false", "0"} and len(negativos) < FILAS_POR_CLASE:
        fila["auto_resolved"] = "No"
        negativos.append(fila)

    if (
        len(positivos) == FILAS_POR_CLASE
        and len(negativos) == FILAS_POR_CLASE
    ):
        break

df = pd.DataFrame(positivos + negativos)

if len(df) != FILAS_POR_CLASE * 2:
    raise RuntimeError(
        f"No se reunieron suficientes registros. Se obtuvieron {len(df):,}."
    )

df.insert(0, "case_id", range(1, len(df) + 1))
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

SALIDA.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(SALIDA, index=False, encoding="utf-8")

print("\nDataset preparado correctamente")
print(f"Archivo: {SALIDA}")
print(f"Filas: {len(df):,}")
print(f"Variables: {len(df.columns)}")
print(f"Duplicados: {df.duplicated().sum()}")
print(f"Nulos totales: {int(df.isna().sum().sum())}")
print("\nDistribución del resultado:")
print(df["auto_resolved"].value_counts())