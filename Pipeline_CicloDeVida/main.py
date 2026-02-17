import json
import joblib

from src.ingest import ingest_and_profile
from src.eda import run_basic_eda
from src.visualize import run_all_visuals, save_log_transform_comparison
from src.preprocess import preprocess, split_for_regression, split_for_classification
from src.transform import apply_transformations
from src.train_regression import train_regression_models, predict_regression_models
from src.train_classif import train_classification_models, predict_classification_models
from src.evaluate import (
    evaluate_regression, evaluate_classification,
    save_regression_pred_plot, save_confusion_matrix_plot
)
from src.config import RESULTS_DIR, MODELS_DIR
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

    # 6) Split
    Xtr_r, Xte_r, ytr_r, yte_r = split_for_regression(prep, test_size=0.2)
    Xtr_c, Xte_c, ytr_c, yte_c = split_for_classification(prep, test_size=0.2)

    # 7) Entrenamiento regresión (GPA)
    reg_models = train_regression_models(Xtr_r, ytr_r)
    reg_preds = predict_regression_models(reg_models, Xte_r)
    reg_metrics = evaluate_regression(yte_r.values, reg_preds)

    # Guardar resultados regresión
    reg_metrics.to_csv(RESULTS_DIR / "metrics_regression.csv", index=False, encoding="utf-8")

    # Guardar plot y_true vs y_pred del mejor modelo (MAE más bajo)
    best_reg_name = reg_metrics.iloc[0]["model"]
    save_regression_pred_plot(yte_r.values, reg_preds[best_reg_name], best_reg_name)

    # 8) Entrenamiento clasificación (Estrés)
    clf_models = train_classification_models(Xtr_c, ytr_c)
    clf_preds = predict_classification_models(clf_models, Xte_c)
    clf_metrics = evaluate_classification(yte_c.values, clf_preds, average="weighted")

    # Guardar resultados clasificación
    clf_metrics.to_csv(RESULTS_DIR / "metrics_classification.csv", index=False, encoding="utf-8")

    # Guardar matriz de confusión del mejor modelo (F1 weighted más alto)
    best_clf_name = clf_metrics.iloc[0]["model"]
    # labels ordenadas por el mapping (0,1,2...)
    inv_map = {v: k for k, v in prep.stress_mapping.items()}
    class_labels = [inv_map[i] for i in sorted(inv_map.keys())]
    save_confusion_matrix_plot(yte_c.values, clf_preds[best_clf_name], best_clf_name, class_labels)

    # 9) Guardar modelos (mejores)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(reg_models[best_reg_name], MODELS_DIR / f"best_reg_{best_reg_name}.joblib")
    joblib.dump(clf_models[best_clf_name], MODELS_DIR / f"best_clf_{best_clf_name}.joblib")
    joblib.dump(prep.scaler, MODELS_DIR / "scaler.joblib")

    # 10) Prints finales
    print("\ PIPELINE COMPLETO EJECUTADO")
    print("\n--- Mejores modelos ---")
    print("Regresión (GPA):", best_reg_name)
    print("Clasificación (Stress):", best_clf_name)

    print("\n--- Métricas Regresión (top) ---")
    print(reg_metrics.head())

    print("\n--- Métricas Clasificación (top) ---")
    print(clf_metrics.head())

    print("\n Archivos generados:")
    print("- reports/results/eda_report.json")
    print("- reports/results/transform_report.json")
    print("- reports/results/stress_mapping.json")
    print("- reports/results/metrics_regression.csv")
    print("- reports/results/metrics_classification.csv")
    print("- reports/figures/ (incluye UMAP y demás)")
    print("- data/processed/ (transformados y dataset_processed.csv)")
    print("- models/ (best_reg_*, best_clf_*, scaler.joblib)")
    
    print("\n" + "=" * 80)
    print(f" Ejecución completada. Log guardado en: {log_file}")
    
    # Cerrar el archivo de salida si es necesario
    if hasattr(output_writer, 'close'):
        output_writer.close()


if __name__ == "__main__":
    main()
