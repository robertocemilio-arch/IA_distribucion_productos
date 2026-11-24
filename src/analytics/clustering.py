import sqlite3
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.analytics.preprocessing import preparar_matriz_escalada


def asegurar_columna_cluster(conn: sqlite3.Connection) -> None:
    """Agrega la columna 'cluster' a la tabla 'producto' si aún no existe."""
    cols = [row[1] for row in conn.execute("PRAGMA table_info(producto);").fetchall()]
    if "cluster" not in cols:
        conn.execute("ALTER TABLE producto ADD COLUMN cluster INTEGER DEFAULT NULL;")


def evaluar_y_aplicar_kmeans(
    conn: sqlite3.Connection,
    k_range: range = range(2, 7),
    k_fijo: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Evalúa K por silueta y guarda el K óptimo (o k_fijo si se especifica)."""
    asegurar_columna_cluster(conn)
    df_prod, X_scaled, _ = preparar_matriz_escalada(conn)

    n_muestras = len(df_prod)
    ks_candidatos = set(k_range)
    if k_fijo is not None:
        ks_candidatos.add(k_fijo)

    ks_validos = sorted(k for k in ks_candidatos if 2 <= k < n_muestras)
    if not ks_validos:
        raise ValueError(
            f"No hay suficientes productos ({n_muestras}) para aplicar K-Means."
        )

    resultados_k = []
    modelos = {}
    for k in ks_validos:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = float(silhouette_score(X_scaled, labels))
        resultados_k.append({"K": k, "silueta": round(score, 4)})
        modelos[k] = labels

    df_eval = pd.DataFrame(resultados_k)
    if k_fijo is not None and k_fijo in modelos:
        k_elegido = k_fijo
    else:
        mejor_fila = df_eval.loc[df_eval["silueta"].idxmax()]
        k_elegido = int(mejor_fila["K"])

    df_prod["cluster"] = modelos[k_elegido]

    updates = [
        (int(row.cluster), int(row.producto_id))
        for row in df_prod.itertuples(index=False)
    ]
    conn.executemany(
        "UPDATE producto SET cluster = ? WHERE producto_id = ?;",
        updates,
    )
    return df_prod, df_eval, k_elegido