import json

from src.ingest import ingest_and_profile
from src.eda import run_basic_eda
from src.visualize import run_all_visuals, save_log_transform_comparison
from src.preprocess import preprocess
from src.transform import apply_transformations
from src.config import RESULTS_DIR
from src.utils import setup_results_output


def main():
    # Configurar redirección de salida a archivo
    log_file, output_writer = setup_results_output(RESULTS_DIR)
    print(f" Guardando resultados en: {log_file}")
    print("=" * 80)
    
    # 1) Ingesta
    out = ingest_and_profile()
    df = out["df"]

    # 2) EDA (resumen)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    eda = run_basic_eda(df)
    with open(RESULTS_DIR / "eda_report.json", "w", encoding="utf-8") as f:
        json.dump(eda.__dict__, f, ensure_ascii=False, indent=2)
    
    # Imprimir resumen de medidas estadísticas
    print("\n" + "=" * 80)
    print("ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    print("=" * 80)
    print("\n Medidas Estadísticas (Media, Mediana, Moda, Desv. Est.):")
    for col, stats in eda.statistical_measures.items():
        print(f"\n  {col}:")
        print(f"    • Media: {stats.get('media', 'N/A'):.4f}")
        print(f"    • Mediana: {stats.get('mediana', 'N/A'):.4f}")
        print(f"    • Moda: {stats.get('moda', 'N/A')}")
        print(f"    • Desv. Estándar: {stats.get('desv_estandar', 'N/A'):.4f}")
    
    print("\n" + "-" * 80)
    print(" Análisis de Outliers (Regla 1.5 × IQR):")
    for col, analysis in eda.outlier_analysis.items():
        num_outliers = analysis.get('cantidad_outliers', 0)
        pct_outliers = analysis.get('porcentaje_outliers', 0)
        print(f"\n  {col}:")
        print(f"    • Cantidad de outliers: {num_outliers} ({pct_outliers:.2f}%)")
        print(f"    • Límite inferior: {analysis.get('limite_inferior', 'N/A'):.4f}")
        print(f"    • Límite superior: {analysis.get('limite_superior', 'N/A'):.4f}")
        if num_outliers > 0:
            print(f"    • Impacto en media por eliminar outliers: {analysis.get('impacto_remocion', {}).get('media', {}).get('cambio_porcentaje', 0):.2f}%")
            print(f"    • Impacto en desv. estándar: {analysis.get('impacto_remocion', {}).get('desv_estandar', {}).get('cambio_porcentaje', 0):.2f}%")

    # 3) Visualizaciones (incluye UMAP)
    run_all_visuals(df)

    # 4) Transformaciones (encoding, log, feature engineering)
    transform_out = apply_transformations(df, out.get("csv_path"))
    save_log_transform_comparison(df, transform_out.base_df, transform_out.log_transform_cols)
    with open(RESULTS_DIR / "transform_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "categorical_columns": transform_out.categorical_columns,
                "engineered_features": transform_out.engineered_features,
                "label_mappings": transform_out.label_mappings,
                "log_skew_report": transform_out.log_skew_report,
                "output_paths": {k: str(v) for k, v in transform_out.output_paths.items()},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 5) Preprocesamiento
    prep = preprocess(df)

    # Guardar mapping estrés
    with open(RESULTS_DIR / "stress_mapping.json", "w", encoding="utf-8") as f:
        json.dump(prep.stress_mapping, f, ensure_ascii=False, indent=2)

    # Proceso completado
    print("\n" + "=" * 80)
    print(" ANÁLISIS COMPLETADO")
    print("\n Archivos generados:")
    print("- reports/results/eda_report.json")
    print("- reports/results/transform_report.json")
    print("- reports/results/stress_mapping.json")
    print("- reports/figures/ (incluye visualizaciones)")
    print("- data/processed/ (datasets transformados)")
    
    print("\n" + "=" * 80)
    print(f" Ejecución completada. Log guardado en: {log_file}")
    
    # Cerrar el archivo de salida si es necesario
    if hasattr(output_writer, 'close'):
        output_writer.close()


if __name__ == "__main__":
    main()
