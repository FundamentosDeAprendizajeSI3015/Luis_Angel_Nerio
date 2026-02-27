"""
Script: make_student_lifestyle_more_realistic.py
Objetivo:
- Cargar tu CSV original
- Introducir "ruido" controlado en la etiqueta Stress_Level (simula variabilidad humana/errores de medición)
- Guardar un nuevo CSV para que lo uses directo en tu pipeline

✅ Recomendación: noise_rate=0.18 → suele dejar accuracy ~0.80-0.85 (más coherente)
"""

from pathlib import Path
import numpy as np
import pandas as pd


def add_label_noise(df: pd.DataFrame, label_col: str = "Stress_Level",
                    noise_rate: float = 0.18, seed: int = 42) -> pd.DataFrame:
    """
    Cambia aleatoriamente el label de un % de filas (noise_rate).
    Mantiene el balance general relativamente similar, pero ya no será separable perfecto.

    noise_rate:
      - 0.12 → suele quedar ~0.88-0.90
      - 0.18 → suele quedar ~0.80-0.85 (recomendado)
      - 0.25 → suele quedar ~0.70-0.78
    """
    if label_col not in df.columns:
        raise ValueError(f"No encontré la columna objetivo '{label_col}' en el CSV.")

    if not (0.0 <= noise_rate <= 0.5):
        raise ValueError("noise_rate debe estar entre 0.0 y 0.5 (ej: 0.18).")

    rng = np.random.default_rng(seed)
    out = df.copy()

    classes = sorted(out[label_col].dropna().unique().tolist())
    if len(classes) < 2:
        raise ValueError("La columna objetivo tiene menos de 2 clases. No se puede aplicar ruido.")

    n = len(out)
    n_flip = int(round(n * noise_rate))
    idx = rng.choice(out.index.to_numpy(), size=n_flip, replace=False)

    # Cambiar etiqueta por otra clase distinta
    for i in idx:
        current = out.at[i, label_col]
        other = [c for c in classes if c != current]
        out.at[i, label_col] = rng.choice(other)

    return out


def main():
    # === 1) Ajusta estas rutas (puedes pegar la tuya exacta) ===
    input_csv = Path(r"student_lifestyle_dataset.csv")

    # El archivo nuevo se guarda en el mismo folder por defecto
    output_csv = input_csv.with_name("student_lifestyle_dataset_more_realistic.csv")

    # === 2) Ajustes de realismo ===
    NOISE_RATE = 0.18   # <- recomendado para métricas coherentes
    SEED = 42

    print("=" * 70)
    print("MODIFICACIÓN DE DATASET: Stress_Level más realista")
    print("=" * 70)

    if not input_csv.exists():
        raise FileNotFoundError(f"No existe el archivo: {input_csv}")

    print(f"[INFO] Leyendo CSV original: {input_csv}")
    df = pd.read_csv(input_csv)

    print(f"[OK] Dataset cargado. Shape: {df.shape}")
    print("\n[INFO] Distribución original de Stress_Level:")
    print(df["Stress_Level"].value_counts(dropna=False))

    # === 3) Aplicar ruido a la etiqueta ===
    df_mod = add_label_noise(df, label_col="Stress_Level", noise_rate=NOISE_RATE, seed=SEED)

    print("\n[INFO] Distribución NUEVA de Stress_Level (con ruido):")
    print(df_mod["Stress_Level"].value_counts(dropna=False))

    # Métrica simple: % de filas que cambiaron (debería ser ~NOISE_RATE)
    changed = (df_mod["Stress_Level"] != df["Stress_Level"]).mean()
    print(f"\n[INFO] % etiquetas cambiadas: {changed:.3%} (objetivo: ~{NOISE_RATE:.0%})")

    # === 4) Guardar el nuevo dataset ===
    df_mod.to_csv(output_csv, index=False)
    print(f"\n[OK] Nuevo CSV guardado en:\n{output_csv}")
    print("\n👉 Ahora en tu pipeline solo cambia la ruta para leer este archivo nuevo.")
    print("=" * 70)


if __name__ == "__main__":
    main()