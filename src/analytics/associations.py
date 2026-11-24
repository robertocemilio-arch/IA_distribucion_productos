import sqlite3
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder


def ejecutar_reglas_asociacion(
    conn: sqlite3.Connection,
    min_support: float = 0.002,
    min_lift: float = 1.0,
) -> pd.DataFrame:
    """Extrae reglas 1->1 con FP-Growth y las persiste en 'regla_asociacion'."""
    query = """
        SELECT ti.ticket_id, p.producto_id, p.nombre_producto, p.categoria
        FROM ticket_item ti
        JOIN producto p ON ti.producto_id = p.producto_id;
    """
    df_items = pd.read_sql_query(query, conn)

    transacciones = df_items.groupby("ticket_id")["nombre_producto"].apply(list).tolist()

    te = TransactionEncoder()
    te_ary = te.fit(transacciones).transform(transacciones)
    df_onehot = pd.DataFrame(te_ary, columns=te.columns_)

    itemsets = fpgrowth(df_onehot, min_support=min_support, use_colnames=True, max_len=2)

    conn.execute("DELETE FROM regla_asociacion;")
    if itemsets.empty:
        return pd.DataFrame(
            columns=[
                "antecedente_id",
                "consecuente_id",
                "soporte",
                "confianza",
                "lift",
                "misma_categoria",
            ]
        )

    reglas = association_rules(itemsets, metric="lift", min_threshold=min_lift)
    if reglas.empty:
        return pd.DataFrame(
            columns=[
                "antecedente_id",
                "consecuente_id",
                "soporte",
                "confianza",
                "lift",
                "misma_categoria",
            ]
        )

    prod_info = conn.execute(
        "SELECT nombre_producto, producto_id, categoria FROM producto;"
    ).fetchall()
    prod_map = {nombre: pid for nombre, pid, _ in prod_info}
    cat_map = {pid: cat for _, pid, cat in prod_info}

    filas_sql = []
    for row in reglas.itertuples(index=False):
        ant = list(row.antecedents)
        cons = list(row.consequents)
        if len(ant) == 1 and len(cons) == 1:
            filas_sql.append((
                prod_map[ant[0]],
                prod_map[cons[0]],
                round(float(row.support), 6),
                round(float(row.confidence), 6),
                round(float(row.lift), 6),
            ))

    conn.executemany(
        """INSERT INTO regla_asociacion
           (antecedente_id, consecuente_id, soporte, confianza, lift)
           VALUES (?, ?, ?, ?, ?);""",
        filas_sql,
    )

    df_reglas = pd.read_sql_query("SELECT * FROM regla_asociacion;", conn)
    df_reglas["misma_categoria"] = (
        df_reglas["antecedente_id"].map(cat_map)
        == df_reglas["consecuente_id"].map(cat_map)
    )
    return df_reglas