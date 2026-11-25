import pandas as pd
import streamlit as st

from src.ui.charts import (
    grafico_clusters_scatter,
    grafico_comparativa_z,
    grafico_curva_visibilidad,
    grafico_planograma_instancia,
    grafico_productos_por_categoria,
    grafico_reglas_asociacion,
    grafico_silueta_k,
)


def render_tab_datos(df_prod: pd.DataFrame, stats: dict) -> None:
    """Pestaña 1: Resumen transaccional y catálogo filtrable de productos."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Productos únicos", f"{stats['n_productos']}")
    c2.metric("Tickets procesados", f"{stats['n_tickets']:,}")
    c3.metric("Ítems en tickets", f"{stats['n_items']:,}")
    c4.metric("Categorías activas", f"{stats['n_categorias']}")

    st.plotly_chart(grafico_productos_por_categoria(df_prod), use_container_width=True)

    st.subheader("Explorador del Catálogo de Productos")
    f1, f2 = st.columns([1, 2])
    categorias = ["Todas"] + sorted(df_prod["categoria"].unique().tolist())
    cat_sel = f1.selectbox("Filtrar por categoría", categorias, key="filtro_cat_datos")
    busqueda = f2.text_input("Buscar producto por nombre", "", key="busq_prod_datos")

    df_view = df_prod.copy()
    if cat_sel != "Todas":
        df_view = df_view[df_view["categoria"] == cat_sel]
    if busqueda.strip():
        df_view = df_view[
            df_view["nombre_producto"].str.contains(busqueda.strip(), case=False, na=False)
        ]

    st.dataframe(
        df_view[
            [
                "nombre_producto",
                "categoria",
                "precio",
                "ancho_cm",
                "alto_cm",
                "frecuencia_venta",
                "indice_impulso",
                "cluster",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_tab_clustering(
    df_prod: pd.DataFrame,
    df_eval_k: pd.DataFrame,
    k_actual: int,
) -> None:
    """Pestaña 2: Coeficiente de silueta, perfil por cluster y dispersión interactiva."""
    silueta_activa = (
        float(df_eval_k.loc[df_eval_k["K"] == k_actual, "silueta"].iloc[0])
        if not df_eval_k.empty and (df_eval_k["K"] == k_actual).any()
        else 0.0
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Clusters activos (K)", f"{k_actual}")
    c2.metric("Coeficiente de Silueta", f"{silueta_activa:.4f}")
    c3.metric("Productos segmentados", f"{len(df_prod)}")

    if k_actual == 2:
        st.info(
            "Con K = 2 el algoritmo separa los productos de rotación masiva del resto del catálogo. "
            "Puedes probar K = 3 o K = 4 desde la barra lateral para explorar una segmentación más granular."
        )

    col_izq, col_der = st.columns([1.2, 1])
    with col_izq:
        st.plotly_chart(grafico_silueta_k(df_eval_k, k_actual), use_container_width=True)

    with col_der:
        st.subheader("Perfil Promedio por Cluster")
        perfil = (
            df_prod.groupby("cluster", as_index=False)
            .agg(
                productos=("producto_id", "count"),
                precio_prom=("precio", "mean"),
                frecuencia_prom=("frecuencia_venta", "mean"),
                impulso_prom=("indice_impulso", "mean"),
            )
            .round(2)
        )
        st.dataframe(perfil, use_container_width=True, hide_index=True)

    st.divider()
    e1, e2 = st.columns(2)
    opciones_ejes = {
        "Frecuencia de Venta": "frecuencia_venta",
        "Índice de Impulso": "indice_impulso",
        "Precio (Bs.)": "precio",
    }
    eje_x_lbl = e1.selectbox("Eje X", list(opciones_ejes.keys()), index=0)
    eje_y_lbl = e2.selectbox("Eje Y", list(opciones_ejes.keys()), index=1)

    st.plotly_chart(
        grafico_clusters_scatter(df_prod, opciones_ejes[eje_x_lbl], opciones_ejes[eje_y_lbl]),
        use_container_width=True,
    )


def render_tab_asociaciones(df_reglas: pd.DataFrame) -> None:
    """Pestaña 3: Reglas FP-Growth con soporte, confianza, lift y filtro intra-categoría."""
    if df_reglas.empty:
        st.warning("No se encontraron reglas de asociación en la base de datos.")
        return

    n_intra = int(df_reglas["misma_categoria"].sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reglas totales (1 → 1)", f"{len(df_reglas)}")
    c2.metric("Reglas intra-categoría", f"{n_intra}", help=f"{n_intra // 2} pares físicos únicos")
    c3.metric("Lift máximo", f"{df_reglas['lift'].max():.3f}")
    c4.metric("Confianza máxima", f"{df_reglas['confianza'].max():.3f}")

    solo_intra = st.checkbox(
        "Mostrar solo reglas intra-categoría (usadas en el bono de cercanía λ)",
        value=False,
    )
    df_view = df_reglas[df_reglas["misma_categoria"]] if solo_intra else df_reglas

    st.plotly_chart(grafico_reglas_asociacion(df_view), use_container_width=True)

    st.subheader("Detalle de Reglas Extraídas")
    st.dataframe(
        df_view[
            [
                "antecedente",
                "categoria_ant",
                "consecuente",
                "categoria_cons",
                "soporte",
                "confianza",
                "lift",
                "misma_categoria",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_tab_gondola(
    df_asig: pd.DataFrame,
    df_niveles: pd.DataFrame,
    resumen_opt: dict,
    estatura_cm: float,
    ultimo_opt: dict | None = None,
) -> None:
    """Pestaña 4: Métricas Antes vs. Después, Planograma visual y auditoría física."""
    if df_asig.empty:
        st.warning("Aún no hay asignaciones guardadas. Ejecuta el optimizador desde la barra lateral.")
        return

    val = resumen_opt["validacion"]
    total_violaciones = (
        val["violaciones_altura"] + val["violaciones_ancho"] + val["violaciones_facings"]
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Z Línea Base (Antes)", f"{resumen_opt['z_base']:.4f}")
    k2.metric(
        "Z Optimizado (Final)",
        f"{resumen_opt['z_final']:.4f}",
        delta=f"+{resumen_opt['mejora_total_pct']:.2f}%",
    )
    k3.metric("Góndolas instanciadas", f"{df_asig['instancia_id'].nunique()}")
    k4.metric(
        "Auditoría Física",
        f"{val['productos_ubicados']}/{val['productos_totales']}",
        delta="0 violaciones" if total_violaciones == 0 else f"{total_violaciones} alertas",
        delta_color="normal" if total_violaciones == 0 else "inverse",
    )

    if ultimo_opt is not None:
        opt_meta = ultimo_opt["optimizacion"]
        st.caption(
            f"Última ejecución → Greedy puro: Z = {opt_meta['z_greedy']:.4f} "
            f"(+{opt_meta['mejora_greedy_pct']:.2f}%) | "
            f"Búsqueda local (Swap/Relocate): {opt_meta['movimientos_local_search']} mejoras aplicadas "
            f"| Pares afines intra-categoría: {ultimo_opt['afinidades_intra_count']}"
        )

    st.divider()

    sel1, sel2, sel3 = st.columns([1.4, 1.4, 1.2])
    categorias = sorted(df_asig["categoria"].unique().tolist())
    cat_sel = sel1.selectbox("Seleccionar Categoría", categorias, key="cat_planograma")

    df_cat = df_asig[df_asig["categoria"] == cat_sel]
    instancias_meta = (
        df_cat[["instancia_id", "tipo_id", "tipo_gondola"]]
        .drop_duplicates()
        .sort_values("instancia_id")
    )
    opciones_inst = {
        f"Instancia #{int(r.instancia_id)} ({r.tipo_gondola})": (
            int(r.instancia_id),
            int(r.tipo_id),
            str(r.tipo_gondola),
        )
        for r in instancias_meta.itertuples(index=False)
    }
    inst_label = sel2.selectbox("Instancia de Góndola", list(opciones_inst.keys()))
    inst_id, tipo_id, nombre_tipo = opciones_inst[inst_label]

    modo_color = sel3.radio(
        "Colorear productos por",
        options=["cluster", "score_objetivo"],
        format_func=lambda v: "Cluster (K-Means)" if v == "cluster" else "Aporte Z",
        horizontal=True,
    )

    df_inst = df_cat[df_cat["instancia_id"] == inst_id]
    df_niv_tipo = df_niveles[df_niveles["tipo_id"] == tipo_id]

    st.plotly_chart(
        grafico_planograma_instancia(df_inst, df_niv_tipo, color_por=modo_color),
        use_container_width=True,
    )

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(
            grafico_curva_visibilidad(df_niv_tipo, estatura_cm=estatura_cm, nombre_tipo=nombre_tipo),
            use_container_width=True,
        )
    with g2:
        st.plotly_chart(
            grafico_comparativa_z(resumen_opt["resumen_categorias"]),
            use_container_width=True,
        )

    st.subheader(f"Detalle de Productos en {cat_sel} — {inst_label}")
    st.dataframe(
        df_inst[
            [
                "numero_nivel",
                "peso_visibilidad",
                "nombre_producto",
                "cluster",
                "facings",
                "ancho_cm",
                "ancho_ocupado_cm",
                "alto_cm",
                "indice_impulso",
                "score_objetivo",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Ver tabla resumen de optimización por categoría"):
        st.dataframe(
            resumen_opt["resumen_categorias"],
            use_container_width=True,
            hide_index=True,
        )