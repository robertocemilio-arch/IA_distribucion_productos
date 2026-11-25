import sqlite3
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.analytics.clustering import evaluar_y_aplicar_kmeans
from src.analytics.preprocessing import preparar_matriz_escalada
from src.analytics.run_pipeline import ejecutar_pipeline_analitico
from src.db.connection import get_connection
from src.optimizer.gondola_setup import HOLGURA_VERTICAL_CM
from src.optimizer.greedy_solver import (
    LAMBDA_AFINIDAD_DEFAULT,
    MAX_FACINGS_DEFAULT,
    THETA_DEFAULT,
    Slot,
    construir_linea_base,
)
from src.optimizer.run_optimizer import (
    ejecutar_pipeline_optimizacion,
    verificar_restricciones,
)
from src.optimizer.scoring import (
    ALPHA_DEFAULT,
    BETA_DEFAULT,
    calcular_puntajes_y_afinidades,
)
from src.optimizer.visibility import ESTATURA_EDSA_2023_CM


def _evaluar_silueta_solo_lectura(
    conn: sqlite3.Connection,
    k_range: range = range(2, 7),
) -> pd.DataFrame:
    """Calcula el coeficiente de silueta para K en [2..6] sin modificar SQLite."""
    df_prod, X_scaled, _ = preparar_matriz_escalada(conn)
    n_muestras = len(df_prod)
    resultados = []
    for k in k_range:
        if 2 <= k < n_muestras:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            score = float(silhouette_score(X_scaled, labels))
            resultados.append({"K": k, "silueta": round(score, 4)})
    return pd.DataFrame(resultados)


def _evaluar_comparativa_actual(
    conn: sqlite3.Connection,
    df_asignacion: pd.DataFrame,
    alpha: float,
    beta: float,
) -> dict:
    """Compara la línea base empírica (Z_base) frente al planograma persistido (Z_final)."""
    validacion = verificar_restricciones(conn)
    if df_asignacion.empty:
        return {
            "z_base": 0.0,
            "z_final": 0.0,
            "mejora_total_pct": 0.0,
            "afinidades_intra_count": 0,
            "resumen_categorias": pd.DataFrame(),
            "validacion": validacion,
        }

    df_prod_score, afinidades_intra = calcular_puntajes_y_afinidades(
        conn, alpha=alpha, beta=beta
    )
    prod_info = df_prod_score.set_index("producto_id").to_dict(orient="index")

    df_slots_db = pd.read_sql_query(
        """SELECT gi.instancia_id, gi.categoria_asignada, nt.nivel_tipo_id,
                  nt.numero_nivel, tg.ancho_total_cm, nt.alto_disponible_cm,
                  nt.peso_visibilidad
           FROM gondola_instancia gi
           JOIN tipo_gondola tg ON gi.tipo_id = tg.tipo_id
           JOIN nivel_tipo nt ON gi.tipo_id = nt.tipo_id
           ORDER BY gi.categoria_asignada, gi.instancia_id, nt.numero_nivel;""",
        conn,
    )

    filas_resumen = []
    total_z_base = 0.0
    total_z_final = 0.0

    for categoria, grupo_prod in df_prod_score.groupby("categoria", sort=True):
        cat_str = str(categoria)
        prod_ids = [int(pid) for pid in grupo_prod["producto_id"].tolist()]
        prod_set = set(prod_ids)

        afinidades_cat = {
            par: lift
            for par, lift in afinidades_intra.items()
            if par[0] in prod_set and par[1] in prod_set
        }

        sub_slots = df_slots_db[df_slots_db["categoria_asignada"] == cat_str]
        slots = [
            Slot(
                slot_id=idx,
                instancia_id=int(row.instancia_id),
                nivel_tipo_id=int(row.nivel_tipo_id),
                numero_nivel=int(row.numero_nivel),
                ancho_total_cm=float(row.ancho_total_cm),
                alto_disponible_cm=float(row.alto_disponible_cm),
                peso_visibilidad=float(row.peso_visibilidad),
            )
            for idx, row in enumerate(sub_slots.itertuples(index=False))
        ]

        z_base = construir_linea_base(
            slots_plantilla=slots,
            prod_ids=prod_ids,
            prod_info=prod_info,
            afinidades_cat=afinidades_cat,
            holgura_cm=HOLGURA_VERTICAL_CM,
            theta=THETA_DEFAULT,
            lambda_afinidad=LAMBDA_AFINIDAD_DEFAULT,
            max_facings=MAX_FACINGS_DEFAULT,
        )

        sub_asig = df_asignacion[df_asignacion["categoria"] == cat_str]
        z_final = float(sub_asig["score_objetivo"].sum())
        mejora_pct = ((z_final - z_base) / z_base * 100.0) if z_base > 0 else 0.0

        ocup_por_slot = (
            sub_asig.groupby(["instancia_id", "nivel_tipo_id"])["ancho_ocupado_cm"]
            .sum()
            .to_dict()
        )
        ocupaciones = [
            (ocup_por_slot.get((s.instancia_id, s.nivel_tipo_id), 0.0) / s.ancho_total_cm) * 100.0
            for s in slots
        ]

        filas_resumen.append({
            "categoria": cat_str,
            "productos": len(prod_ids),
            "instancias": int(sub_slots["instancia_id"].nunique()),
            "Z_base": round(z_base, 4),
            "Z_final": round(z_final, 4),
            "mejora_%": round(mejora_pct, 2),
            "ocupacion_%": round(sum(ocupaciones) / len(ocupaciones), 1) if ocupaciones else 0.0,
            "facings_prom": round(float(sub_asig["facings"].mean()), 2) if not sub_asig.empty else 0.0,
        })

        total_z_base += z_base
        total_z_final += z_final

    mejora_total = ((total_z_final - total_z_base) / total_z_base * 100.0) if total_z_base > 0 else 0.0

    return {
        "z_base": round(total_z_base, 4),
        "z_final": round(total_z_final, 4),
        "mejora_total_pct": round(mejora_total, 2),
        "afinidades_intra_count": len(afinidades_intra),
        "resumen_categorias": pd.DataFrame(filas_resumen),
        "validacion": validacion,
    }


@st.cache_data(show_spinner=False)
def cargar_datos_bd(
    alpha: float = ALPHA_DEFAULT,
    beta: float = BETA_DEFAULT,
) -> dict:
    """Carga todas las vistas necesarias desde SQLite con caché de Streamlit."""
    conn = get_connection()
    try:
        df_prod = pd.read_sql_query(
            """SELECT producto_id, nombre_producto, categoria, precio,
                      ancho_cm, alto_cm, frecuencia_venta, indice_impulso, cluster
               FROM producto
               ORDER BY frecuencia_venta DESC;""",
            conn,
        )

        n_tickets = int(conn.execute("SELECT COUNT(*) FROM ticket;").fetchone()[0])
        n_items = int(conn.execute("SELECT COUNT(*) FROM ticket_item;").fetchone()[0])
        n_instancias = int(conn.execute("SELECT COUNT(*) FROM gondola_instancia;").fetchone()[0])

        df_reglas = pd.read_sql_query(
            """SELECT r.regla_id,
                      p1.nombre_producto AS antecedente,
                      p1.categoria AS categoria_ant,
                      p2.nombre_producto AS consecuente,
                      p2.categoria AS categoria_cons,
                      r.soporte, r.confianza, r.lift
               FROM regla_asociacion r
               JOIN producto p1 ON r.antecedente_id = p1.producto_id
               JOIN producto p2 ON r.consecuente_id = p2.producto_id
               ORDER BY r.lift DESC;""",
            conn,
        )
        if not df_reglas.empty:
            df_reglas["misma_categoria"] = (
                df_reglas["categoria_ant"] == df_reglas["categoria_cons"]
            )
        else:
            df_reglas["misma_categoria"] = pd.Series(dtype=bool)

        df_niveles = pd.read_sql_query(
            """SELECT nt.nivel_tipo_id, nt.tipo_id, tg.nombre AS tipo_gondola,
                      tg.ancho_total_cm, tg.num_niveles, nt.numero_nivel,
                      nt.altura_suelo_cm, nt.alto_disponible_cm, nt.peso_visibilidad
               FROM nivel_tipo nt
               JOIN tipo_gondola tg ON nt.tipo_id = tg.tipo_id
               ORDER BY nt.tipo_id, nt.numero_nivel;""",
            conn,
        )
        df_niveles["altura_media_cm"] = (
            df_niveles["altura_suelo_cm"] + df_niveles["alto_disponible_cm"] / 2.0
        ).round(1)

        df_asignacion = pd.read_sql_query(
            """SELECT a.asignacion_id, a.instancia_id, gi.tipo_id,
                      tg.nombre AS tipo_gondola, tg.ancho_total_cm,
                      gi.categoria_asignada AS categoria,
                      a.nivel_tipo_id, nt.numero_nivel, nt.altura_suelo_cm,
                      nt.alto_disponible_cm, nt.peso_visibilidad,
                      a.producto_id, p.nombre_producto, p.precio,
                      p.ancho_cm, p.alto_cm, p.frecuencia_venta,
                      p.indice_impulso, a.facings, a.cluster, a.score_objetivo
               FROM asignacion a
               JOIN producto p ON a.producto_id = p.producto_id
               JOIN gondola_instancia gi ON a.instancia_id = gi.instancia_id
               JOIN tipo_gondola tg ON gi.tipo_id = tg.tipo_id
               JOIN nivel_tipo nt ON a.nivel_tipo_id = nt.nivel_tipo_id
               ORDER BY gi.categoria_asignada, a.instancia_id, nt.numero_nivel DESC, a.score_objetivo DESC;""",
            conn,
        )
        if not df_asignacion.empty:
            df_asignacion["ancho_ocupado_cm"] = (
                df_asignacion["ancho_cm"] * df_asignacion["facings"]
            ).round(1)

        df_eval_k = _evaluar_silueta_solo_lectura(conn)
        resumen_opt = _evaluar_comparativa_actual(conn, df_asignacion, alpha=alpha, beta=beta)

        k_actual = int(df_prod["cluster"].nunique()) if "cluster" in df_prod.columns else 0
        stats = {
            "n_productos": int(len(df_prod)),
            "n_tickets": n_tickets,
            "n_items": n_items,
            "n_categorias": int(df_prod["categoria"].nunique()),
            "n_instancias": n_instancias,
            "k_actual": k_actual,
        }
    finally:
        conn.close()

    return {
        "productos": df_prod,
        "stats": stats,
        "reglas": df_reglas,
        "niveles": df_niveles,
        "asignacion": df_asignacion,
        "eval_k": df_eval_k,
        "resumen_opt": resumen_opt,
    }


def recalcular_sistema(
    k_fijo: int | None,
    estatura_cm: float = ESTATURA_EDSA_2023_CM,
    alpha: float = ALPHA_DEFAULT,
) -> dict:
    """Actualiza K-Means y re-ejecuta el optimizador con los parámetros elegidos."""
    beta = round(1.0 - alpha, 2)

    conn = get_connection()
    try:
        n_reglas = conn.execute("SELECT COUNT(*) FROM regla_asociacion;").fetchone()[0]
        nulos_impulso = conn.execute(
            "SELECT COUNT(*) FROM producto WHERE indice_impulso IS NULL;"
        ).fetchone()[0]

        if n_reglas == 0 or nulos_impulso > 0:
            conn.close()
            res_an = ejecutar_pipeline_analitico(k_fijo=k_fijo)
            k_elegido = int(res_an["k_elegido"])
        else:
            _, _, k_elegido = evaluar_y_aplicar_kmeans(conn, k_fijo=k_fijo)
            conn.commit()
            conn.close()
    except Exception:
        conn.close()
        raise

    res_opt = ejecutar_pipeline_optimizacion(
        estatura_cm=estatura_cm,
        alpha=alpha,
        beta=beta,
    )
    cargar_datos_bd.clear()
    res_opt["k_elegido"] = k_elegido
    return res_opt