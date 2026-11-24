import sqlite3
import pandas as pd

ALPHA_DEFAULT = 0.6
BETA_DEFAULT = 0.4


def calcular_puntajes_y_afinidades(
    conn: sqlite3.Connection,
    alpha: float = ALPHA_DEFAULT,
    beta: float = BETA_DEFAULT,
) -> tuple[pd.DataFrame, dict[tuple[int, int], float]]:
    """Calcula s_i normalizado por producto y construye el mapa de afinidades intra-categoría."""
    cols_prod = [row[1] for row in conn.execute("PRAGMA table_info(producto);").fetchall()]
    if "cluster" not in cols_prod:
        raise RuntimeError(
            "La tabla 'producto' no tiene la columna 'cluster'. "
            "Ejecuta primero: python -m src.analytics.run_pipeline"
        )

    df_prod = pd.read_sql_query(
        """SELECT producto_id, nombre_producto, categoria, precio,
                  ancho_cm, alto_cm, frecuencia_venta, indice_impulso, cluster
           FROM producto;""",
        conn,
    )
    df_prod["indice_impulso"] = df_prod["indice_impulso"].fillna(0.0).astype(float)
    df_prod["cluster"] = df_prod["cluster"].fillna(0).astype(int)

    df_reglas = pd.read_sql_query(
        "SELECT antecedente_id, consecuente_id, lift FROM regla_asociacion;", conn
    )

    # 1. Lift promedio por producto (participación como antecedente o consecuente)
    if df_reglas.empty:
        df_prod["lift_promedio"] = 1.0
    else:
        part_ant = df_reglas[["antecedente_id", "lift"]].rename(
            columns={"antecedente_id": "producto_id"}
        )
        part_cons = df_reglas[["consecuente_id", "lift"]].rename(
            columns={"consecuente_id": "producto_id"}
        )
        avg_lift = pd.concat([part_ant, part_cons], ignore_index=True).groupby("producto_id")["lift"].mean()
        df_prod["lift_promedio"] = df_prod["producto_id"].map(avg_lift).fillna(1.0)

    # 2. Normalización Min-Max de lift_promedio a [0, 1]
    min_lift = float(df_prod["lift_promedio"].min())
    max_lift = float(df_prod["lift_promedio"].max())
    if max_lift > min_lift:
        df_prod["lift_norm"] = (df_prod["lift_promedio"] - min_lift) / (max_lift - min_lift)
    else:
        df_prod["lift_norm"] = 0.0

    # 3. Puntaje individual s_i = alpha * indice_impulso + beta * lift_norm
    df_prod["score_si"] = (
        alpha * df_prod["indice_impulso"] + beta * df_prod["lift_norm"]
    ).round(6)

    # 4. Mapa simétrico de afinidades intra-categoría con lift > 1.0
    cat_map = dict(zip(df_prod["producto_id"], df_prod["categoria"]))
    afinidades_intra: dict[tuple[int, int], float] = {}

    for row in df_reglas.itertuples(index=False):
        u, v = int(row.antecedente_id), int(row.consecuente_id)
        lift = float(row.lift)
        if u != v and lift > 1.0 and cat_map.get(u) == cat_map.get(v):
            par = (min(u, v), max(u, v))
            if lift > afinidades_intra.get(par, 0.0):
                afinidades_intra[par] = lift

    return df_prod, afinidades_intra