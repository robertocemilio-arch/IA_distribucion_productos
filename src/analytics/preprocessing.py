import sqlite3
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES_CLUSTERING = ["precio", "frecuencia_venta", "indice_impulso"]


def calcular_indice_impulso(conn: sqlite3.Connection) -> pd.DataFrame:
    """Calcula indice_impulso [0, 1] desde lift como consecuente y tamaño de canasta."""
    df_prod = pd.read_sql_query(
        "SELECT producto_id, nombre_producto, categoria FROM producto;", conn
    )
    df_items = pd.read_sql_query("SELECT ticket_id, producto_id FROM ticket_item;", conn)
    df_reglas = pd.read_sql_query(
        "SELECT consecuente_id, lift FROM regla_asociacion;", conn
    )

    # 1. Promedio de artículos acompañantes por ticket donde aparece el producto
    tam_ticket = df_items.groupby("ticket_id").size().rename("tam_ticket")
    df_items = df_items.merge(tam_ticket, on="ticket_id")
    df_items["acompanantes"] = df_items["tam_ticket"] - 1
    avg_acomp = df_items.groupby("producto_id")["acompanantes"].mean()

    # 2. Lift promedio cuando el producto es impulsado (consecuente)
    avg_lift = df_reglas.groupby("consecuente_id")["lift"].mean()

    df_calc = df_prod.set_index("producto_id")
    df_calc["avg_acomp"] = avg_acomp.reindex(df_calc.index).fillna(0.0)
    df_calc["avg_lift"] = avg_lift.reindex(df_calc.index).fillna(1.0)

    # Normalización Min-Max a [0, 1]
    acomp_norm = (df_calc["avg_acomp"] - df_calc["avg_acomp"].min()) / (
        df_calc["avg_acomp"].max() - df_calc["avg_acomp"].min() + 1e-9
    )
    lift_norm = (df_calc["avg_lift"] - df_calc["avg_lift"].min()) / (
        df_calc["avg_lift"].max() - df_calc["avg_lift"].min() + 1e-9
    )

    df_calc["indice_impulso"] = (0.6 * lift_norm + 0.4 * acomp_norm).round(4)

    updates = [
        (float(val), int(pid))
        for pid, val in df_calc["indice_impulso"].items()
    ]
    conn.executemany(
        "UPDATE producto SET indice_impulso = ? WHERE producto_id = ?;",
        updates,
    )
    return df_calc.reset_index()


def preparar_matriz_escalada(
    conn: sqlite3.Connection,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    """Lee atributos del producto, imputa nulos si existen y aplica Z-score."""
    cols = features or FEATURES_CLUSTERING
    df_prod = pd.read_sql_query(
        """SELECT producto_id, nombre_producto, categoria,
                  precio, frecuencia_venta, indice_impulso
           FROM producto;""",
        conn,
    )

    df_num = df_prod[cols].fillna(0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_num)
    return df_prod, X_scaled, scaler