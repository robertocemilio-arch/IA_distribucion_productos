import sqlite3

from src.db.connection import get_connection
from src.optimizer.gondola_setup import HOLGURA_VERTICAL_CM, configurar_gondolas_por_categoria
from src.optimizer.greedy_solver import (
    LAMBDA_AFINIDAD_DEFAULT,
    MAX_FACINGS_DEFAULT,
    THETA_DEFAULT,
    ejecutar_optimizacion_global,
)
from src.optimizer.scoring import ALPHA_DEFAULT, BETA_DEFAULT, calcular_puntajes_y_afinidades
from src.optimizer.visibility import ESTATURA_EDSA_2023_CM, actualizar_pesos_visibilidad


def verificar_restricciones(
    conn: sqlite3.Connection,
    holgura_cm: float = HOLGURA_VERTICAL_CM,
    max_facings: int = MAX_FACINGS_DEFAULT,
) -> dict:
    """Valida integridad física y cobertura del 100% del planograma guardado."""
    total_prod = conn.execute("SELECT COUNT(*) FROM producto;").fetchone()[0]
    total_asig = conn.execute("SELECT COUNT(DISTINCT producto_id) FROM asignacion;").fetchone()[0]

    violaciones_altura = conn.execute(
        """SELECT COUNT(*)
           FROM asignacion a
           JOIN producto p ON a.producto_id = p.producto_id
           JOIN nivel_tipo nt ON a.nivel_tipo_id = nt.nivel_tipo_id
           WHERE (p.alto_cm + ?) > nt.alto_disponible_cm + 1e-6;""",
        (holgura_cm,),
    ).fetchone()[0]

    violaciones_ancho = conn.execute(
        """SELECT COUNT(*) FROM (
               SELECT a.instancia_id, a.nivel_tipo_id,
                      SUM(p.ancho_cm * a.facings) AS ancho_ocupado,
                      tg.ancho_total_cm
               FROM asignacion a
               JOIN producto p ON a.producto_id = p.producto_id
               JOIN gondola_instancia gi ON a.instancia_id = gi.instancia_id
               JOIN tipo_gondola tg ON gi.tipo_id = tg.tipo_id
               GROUP BY a.instancia_id, a.nivel_tipo_id
               HAVING ancho_ocupado > tg.ancho_total_cm + 1e-6
           );"""
    ).fetchone()[0]

    violaciones_facings = conn.execute(
        "SELECT COUNT(*) FROM asignacion WHERE facings < 1 OR facings > ?;",
        (max_facings,),
    ).fetchone()[0]

    return {
        "productos_totales": int(total_prod),
        "productos_ubicados": int(total_asig),
        "cobertura_100": total_prod == total_asig,
        "violaciones_altura": int(violaciones_altura),
        "violaciones_ancho": int(violaciones_ancho),
        "violaciones_facings": int(violaciones_facings),
    }


def ejecutar_pipeline_optimizacion(
    estatura_cm: float = ESTATURA_EDSA_2023_CM,
    alpha: float = ALPHA_DEFAULT,
    beta: float = BETA_DEFAULT,
    theta: float = THETA_DEFAULT,
    lambda_afinidad: float = LAMBDA_AFINIDAD_DEFAULT,
    max_facings: int = MAX_FACINGS_DEFAULT,
    holgura_cm: float = HOLGURA_VERTICAL_CM,
) -> dict:
    """Ejecuta toda la Capa de Optimización y persiste los resultados en SQLite."""
    conn = get_connection()
    try:
        df_visibilidad = actualizar_pesos_visibilidad(conn, estatura_cm=estatura_cm)
        df_gondolas = configurar_gondolas_por_categoria(conn, holgura_cm=holgura_cm)
        df_prod, afinidades_intra = calcular_puntajes_y_afinidades(
            conn, alpha=alpha, beta=beta
        )
        res_opt = ejecutar_optimizacion_global(
            conn=conn,
            df_prod=df_prod,
            afinidades_intra=afinidades_intra,
            holgura_cm=holgura_cm,
            theta=theta,
            lambda_afinidad=lambda_afinidad,
            max_facings=max_facings,
        )
        validacion = verificar_restricciones(
            conn, holgura_cm=holgura_cm, max_facings=max_facings
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "visibilidad": df_visibilidad,
        "gondolas": df_gondolas,
        "afinidades_intra_count": len(afinidades_intra),
        "optimizacion": res_opt,
        "validacion": validacion,
    }


def main() -> None:
    res = ejecutar_pipeline_optimizacion()
    df_vis = res["visibilidad"]
    opt = res["optimizacion"]
    val = res["validacion"]

    print("=== CAPA DE OPTIMIZACIÓN COMPLETADA (EDSA 2023: H = 161.0 cm) ===")
    print("\n1. Pesos de visibilidad calculados (nivel_tipo):")
    print(
        df_vis[["tipo_gondola", "numero_nivel", "altura_media_cm", "peso_visibilidad"]].to_string(
            index=False
        )
    )

    print("\n2. Comparativa de Función Objetivo (Antes vs. Después):")
    print(f"   - Z Línea Base (Antes):      {opt['z_base']:.4f}")
    print(f"   - Z Heurística Greedy:       {opt['z_greedy']:.4f} (+{opt['mejora_greedy_pct']:.2f}%)")
    print(
        f"   - Z Final (Greedy + Swap):   {opt['z_final']:.4f} "
        f"(+{opt['mejora_total_pct']:.2f}% | {opt['movimientos_local_search']} mejoras locales)"
    )
    print(f"   - Reglas intra-categoría aplicadas: {res['afinidades_intra_count']}")

    print("\n3. Resultados por Categoría:")
    print(opt["resumen_categorias"].to_string(index=False))

    print("\n4. Auditoría de Restricciones Físicas:")
    print(f"   - Cobertura del catálogo:  {val['productos_ubicados']}/{val['productos_totales']}")
    print(f"   - Violaciones de altura:   {val['violaciones_altura']}")
    print(f"   - Violaciones de ancho:    {val['violaciones_ancho']}")
    print(f"   - Violaciones de facings:  {val['violaciones_facings']}")


if __name__ == "__main__":
    main()