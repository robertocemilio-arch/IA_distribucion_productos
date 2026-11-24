from src.analytics.associations import ejecutar_reglas_asociacion
from src.analytics.clustering import evaluar_y_aplicar_kmeans
from src.analytics.preprocessing import calcular_indice_impulso
from src.db.connection import get_connection


def ejecutar_pipeline_analitico(
    min_support: float = 0.002,
    min_lift: float = 1.0,
    k_range: range = range(2, 7),
    k_fijo: int | None = None,
) -> dict:
    """Ejecuta en orden: asociaciones -> impulso -> estandarización y K-Means."""
    conn = get_connection()
    try:
        df_reglas = ejecutar_reglas_asociacion(
            conn, min_support=min_support, min_lift=min_lift
        )
        df_impulso = calcular_indice_impulso(conn)
        df_clusters, df_eval, k_elegido = evaluar_y_aplicar_kmeans(
            conn, k_range=k_range, k_fijo=k_fijo
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "reglas": df_reglas,
        "impulso": df_impulso,
        "clusters": df_clusters,
        "eval_k": df_eval,
        "k_elegido": k_elegido,
    }


def main() -> None:
    res = ejecutar_pipeline_analitico()
    df_reglas = res["reglas"]
    df_impulso = res["impulso"]
    df_clusters = res["clusters"]
    df_eval = res["eval_k"]
    mejor_k = res["k_elegido"]

    n_intra = int(df_reglas["misma_categoria"].sum()) if not df_reglas.empty else 0
    print("=== CAPA DE ANÁLISIS COMPLETADA ===")
    print(f"Reglas de asociación guardadas: {len(df_reglas)} (Intra-categoría: {n_intra})")

    print("\nEvaluación de K-Means (Coeficiente de Silueta):")
    print(df_eval.to_string(index=False))
    print(f"-> K óptimo seleccionado automáticamente: K = {mejor_k}")

    print("\nPerfil promedio por Cluster (valores reales):")
    perfil = (
        df_clusters.groupby("cluster")
        .agg(
            productos=("producto_id", "count"),
            precio_prom=("precio", "mean"),
            frecuencia_prom=("frecuencia_venta", "mean"),
            impulso_prom=("indice_impulso", "mean"),
        )
        .round(2)
    )
    print(perfil.to_string())

    print("\nTop 5 productos por Índice de Impulso:")
    print(
        df_impulso.sort_values("indice_impulso", ascending=False)[
            ["nombre_producto", "categoria", "indice_impulso"]
        ]
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()