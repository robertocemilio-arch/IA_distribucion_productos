import math
import sqlite3
import pandas as pd

HOLGURA_VERTICAL_CM = 2.0


def seleccionar_tipo_gondola(
    max_alto_prod: float,
    df_tipos_meta: pd.DataFrame,
    holgura_cm: float = HOLGURA_VERTICAL_CM,
) -> int:
    """Elige el tipo de góndola de mayor capacidad lineal compatible con la altura."""
    alto_req = max_alto_prod + holgura_cm

    # Prioridad 1: que el producto más alto quepa en TODOS los niveles (incluyendo zona caliente)
    compat_todos = df_tipos_meta[df_tipos_meta["min_alto_nivel"] >= alto_req]
    if not compat_todos.empty:
        mejor = compat_todos.sort_values("capacidad_lineal_cm", ascending=False).iloc[0]
        return int(mejor["tipo_id"])

    # Prioridad 2: que quepa al menos en el nivel más alto del tipo
    compat_alguno = df_tipos_meta[df_tipos_meta["max_alto_nivel"] >= alto_req]
    if not compat_alguno.empty:
        mejor = compat_alguno.sort_values("capacidad_lineal_cm", ascending=False).iloc[0]
        return int(mejor["tipo_id"])

    # Respaldo: tipo con mayor altura física disponible
    mejor = df_tipos_meta.sort_values("max_alto_nivel", ascending=False).iloc[0]
    return int(mejor["tipo_id"])


def estimar_instancias_ffd(
    anchos_cm: list[float],
    ancho_nivel_cm: float,
    num_niveles: int,
) -> int:
    """Calcula el mínimo de instancias necesarias mediante First-Fit Decreasing."""
    bins_restantes: list[float] = []
    for ancho in sorted(anchos_cm, reverse=True):
        ubicado = False
        for i, libre in enumerate(bins_restantes):
            if ancho <= libre + 1e-9:
                bins_restantes[i] -= ancho
                ubicado = True
                break
        if not ubicado:
            bins_restantes.append(ancho_nivel_cm - ancho)

    return max(1, math.ceil(len(bins_restantes) / num_niveles))


def crear_instancia_gondola(
    conn: sqlite3.Connection,
    tipo_id: int,
    categoria: str,
) -> int:
    """Inserta una nueva instancia en 'gondola_instancia' y devuelve su instancia_id."""
    cursor = conn.execute(
        "INSERT INTO gondola_instancia (tipo_id, categoria_asignada) VALUES (?, ?);",
        (int(tipo_id), categoria),
    )
    return int(cursor.lastrowid)


def configurar_gondolas_por_categoria(
    conn: sqlite3.Connection,
    holgura_cm: float = HOLGURA_VERTICAL_CM,
) -> pd.DataFrame:
    """Crea automáticamente las instancias necesarias por categoría (Modo A: 100% cobertura)."""
    conn.execute("DELETE FROM asignacion;")
    conn.execute("DELETE FROM gondola_instancia;")

    df_tipos_meta = pd.read_sql_query(
        """SELECT tg.tipo_id, tg.nombre, tg.ancho_total_cm, tg.num_niveles,
                  (tg.ancho_total_cm * tg.num_niveles) AS capacidad_lineal_cm,
                  MIN(nt.alto_disponible_cm) AS min_alto_nivel,
                  MAX(nt.alto_disponible_cm) AS max_alto_nivel
           FROM tipo_gondola tg
           JOIN nivel_tipo nt ON tg.tipo_id = nt.tipo_id
           GROUP BY tg.tipo_id;""",
        conn,
    )
    meta_por_tipo = df_tipos_meta.set_index("tipo_id").to_dict(orient="index")

    df_prod = pd.read_sql_query(
        "SELECT producto_id, categoria, ancho_cm, alto_cm FROM producto;", conn
    )

    resumen = []
    for categoria, grupo in df_prod.groupby("categoria", sort=True):
        max_alto = float(grupo["alto_cm"].max())
        tipo_id = seleccionar_tipo_gondola(max_alto, df_tipos_meta, holgura_cm=holgura_cm)
        info_tipo = meta_por_tipo[tipo_id]

        n_instancias = estimar_instancias_ffd(
            grupo["ancho_cm"].tolist(),
            float(info_tipo["ancho_total_cm"]),
            int(info_tipo["num_niveles"]),
        )

        for _ in range(n_instancias):
            crear_instancia_gondola(conn, tipo_id, str(categoria))

        resumen.append({
            "categoria": str(categoria),
            "tipo_id": tipo_id,
            "tipo_gondola": info_tipo["nombre"],
            "n_productos": int(len(grupo)),
            "max_alto_cm": round(max_alto, 1),
            "ancho_total_req_cm": round(float(grupo["ancho_cm"].sum()), 1),
            "n_instancias": n_instancias,
            "capacidad_total_cm": round(n_instancias * float(info_tipo["capacidad_lineal_cm"]), 1),
        })

    return pd.DataFrame(resumen)