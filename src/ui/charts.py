import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go

from src.optimizer.visibility import FACTOR_CENTRO_VISUAL, calcular_peso_visibilidad

CLUSTER_COLORS = {
    0: "#2563EB",  # Azul
    1: "#EF4444",  # Rojo coral (destaca cluster de mayor rotación)
    2: "#10B981",  # Esmeralda
    3: "#F59E0B",  # Ámbar
    4: "#8B5CF6",  # Violeta
    5: "#EC4899",  # Magenta
}


def _color_cluster(cluster_id: int) -> str:
    return CLUSTER_COLORS.get(int(cluster_id), "#64748B")


def grafico_productos_por_categoria(df_prod: pd.DataFrame) -> go.Figure:
    """Barras horizontales con la cantidad de productos por categoría."""
    conteo = (
        df_prod.groupby("categoria", as_index=False)
        .agg(productos=("producto_id", "count"))
        .sort_values("productos", ascending=True)
    )
    fig = px.bar(
        conteo,
        x="productos",
        y="categoria",
        orientation="h",
        text="productos",
        color="productos",
        color_continuous_scale="Blues",
        labels={"productos": "Cantidad de productos", "categoria": "Categoría"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        title="Distribución del Catálogo por Categoría",
        coloraxis_showscale=False,
        height=360,
        margin=dict(l=10, r=30, t=45, b=20),
    )
    return fig


def grafico_silueta_k(df_eval_k: pd.DataFrame, k_actual: int) -> go.Figure:
    """Curva del coeficiente de silueta para K en [2..6] resaltando el K activo."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_eval_k["K"],
            y=df_eval_k["silueta"],
            mode="lines+markers+text",
            text=[f"{v:.3f}" for v in df_eval_k["silueta"]],
            textposition="top center",
            line=dict(color="#2563EB", width=2.5),
            marker=dict(size=9, color="#2563EB"),
            name="Coeficiente de Silueta",
        )
    )

    fila_activa = df_eval_k[df_eval_k["K"] == k_actual]
    if not fila_activa.empty:
        fig.add_trace(
            go.Scatter(
                x=fila_activa["K"],
                y=fila_activa["silueta"],
                mode="markers",
                marker=dict(size=15, color="#EF4444", line=dict(color="#FFFFFF", width=2)),
                name=f"K activo (K={k_actual})",
            )
        )

    fig.update_layout(
        title="Evaluación de K-Means por Coeficiente de Silueta",
        xaxis=dict(title="Número de Clusters (K)", dtick=1),
        yaxis=dict(
            title="Silueta",
            range=[0, min(1.0, float(df_eval_k["silueta"].max()) + 0.15)],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=350,
        margin=dict(l=10, r=20, t=55, b=20),
    )
    return fig


def grafico_clusters_scatter(
    df_prod: pd.DataFrame,
    eje_x: str = "frecuencia_venta",
    eje_y: str = "indice_impulso",
) -> go.Figure:
    """Gráfico de dispersión interactivo de productos coloreado por cluster."""
    etiquetas = {
        "frecuencia_venta": "Frecuencia de Venta (tickets)",
        "indice_impulso": "Índice de Impulso [0 - 1]",
        "precio": "Precio (Bs.)",
    }
    df_plot = df_prod.copy()
    df_plot["cluster_label"] = df_plot["cluster"].apply(lambda c: f"Cluster {int(c)}")

    mapa_colores = {
        f"Cluster {c}": _color_cluster(c) for c in sorted(df_plot["cluster"].unique())
    }
    orden_clusters = sorted(mapa_colores.keys())

    fig = px.scatter(
        df_plot,
        x=eje_x,
        y=eje_y,
        color="cluster_label",
        category_orders={"cluster_label": orden_clusters},
        color_discrete_map=mapa_colores,
        size="precio",
        size_max=18,
        hover_name="nombre_producto",
        hover_data={
            "categoria": True,
            "precio": ":.2f",
            "frecuencia_venta": True,
            "indice_impulso": ":.4f",
            "cluster_label": False,
        },
        labels={
            eje_x: etiquetas.get(eje_x, eje_x),
            eje_y: etiquetas.get(eje_y, eje_y),
            "cluster_label": "Segmento",
        },
    )
    fig.update_layout(
        title=f"Segmentación K-Means: {etiquetas.get(eje_x, eje_x)} vs. {etiquetas.get(eje_y, eje_y)}",
        height=440,
        margin=dict(l=10, r=20, t=50, b=20),
    )
    return fig


def grafico_reglas_asociacion(df_reglas: pd.DataFrame) -> go.Figure:
    """Dispersión Soporte vs. Confianza coloreada por Lift."""
    df_plot = df_reglas.copy()
    df_plot["regla"] = df_plot["antecedente"] + " → " + df_plot["consecuente"]
    df_plot["tipo_regla"] = df_plot["misma_categoria"].map(
        {True: "Intra-categoría (misma góndola)", False: "Inter-categoría"}
    )

    fig = px.scatter(
        df_plot,
        x="soporte",
        y="confianza",
        color="lift",
        symbol="tipo_regla",
        size="lift",
        size_max=18,
        color_continuous_scale="Viridis",
        hover_name="regla",
        hover_data={
            "categoria_ant": True,
            "categoria_cons": True,
            "soporte": ":.4f",
            "confianza": ":.3f",
            "lift": ":.3f",
            "tipo_regla": False,
        },
        labels={
            "soporte": "Soporte",
            "confianza": "Confianza",
            "lift": "Lift",
            "tipo_regla": "Alcance",
        },
    )
    fig.update_layout(
        title="Mapa de Reglas de Asociación (FP-Growth)",
        height=410,
        margin=dict(l=10, r=20, t=50, b=20),
    )
    return fig


def grafico_curva_visibilidad(
    df_niveles_tipo: pd.DataFrame,
    estatura_cm: float,
    nombre_tipo: str,
) -> go.Figure:
    """Campana gaussiana asimétrica EDSA 2023 con los estantes reales superpuestos."""
    alturas = np.linspace(10.0, 190.0, 181)
    pesos = [
        calcular_peso_visibilidad(
            altura_suelo_cm=float(h),
            alto_disponible_cm=0.0,
            estatura_cm=estatura_cm,
        )
        for h in alturas
    ]
    h_opt = FACTOR_CENTRO_VISUAL * estatura_cm

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=alturas,
            y=pesos,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(37, 99, 235, 0.12)",
            line=dict(color="#2563EB", width=2.5),
            name="Curva EDSA 2023",
            hovertemplate="Altura: %{x:.0f} cm<br>Visibilidad w_j: %{y:.3f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_niveles_tipo["altura_media_cm"],
            y=df_niveles_tipo["peso_visibilidad"],
            mode="markers+text",
            text=[f"N{int(n)}" for n in df_niveles_tipo["numero_nivel"]],
            textposition="top center",
            marker=dict(size=12, color="#EF4444", line=dict(color="#FFFFFF", width=2)),
            name="Niveles de góndola",
            hovertemplate=(
                "<b>Nivel %{text}</b><br>"
                "Centro: %{x:.1f} cm<br>"
                "Peso w_j: %{y:.4f}<extra></extra>"
            ),
        )
    )

    fig.add_vline(
        x=h_opt,
        line_dash="dash",
        line_color="#10B981",
        annotation_text=f"h_opt = {h_opt:.1f} cm",
        annotation_position="top right",
    )

    fig.update_layout(
        title=f"Perfil de Visibilidad EDSA 2023 (H={estatura_cm:.1f} cm) · {nombre_tipo}",
        xaxis=dict(title="Altura desde el suelo (cm)", range=[10, 190]),
        yaxis=dict(title="Peso de visibilidad (w_j)", range=[0, 1.12]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=370,
        margin=dict(l=10, r=20, t=55, b=20),
    )
    return fig


def grafico_comparativa_z(df_resumen_cat: pd.DataFrame) -> go.Figure:
    """Barras agrupadas comparando Z_base (Antes) vs. Z_final (Después) por categoría."""
    df_ord = df_resumen_cat.sort_values("Z_final", ascending=True)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=df_ord["categoria"],
            x=df_ord["Z_base"],
            name="Línea Base (Antes)",
            orientation="h",
            marker_color="#94A3B8",
            hovertemplate="<b>%{y}</b><br>Z Base: %{x:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=df_ord["categoria"],
            x=df_ord["Z_final"],
            name="Optimizado (Después)",
            orientation="h",
            marker_color="#2563EB",
            text=[f"+{m:.1f}%" for m in df_ord["mejora_%"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Z Final: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Impacto de la Optimización por Categoría (Z Base vs. Z Final)",
        barmode="group",
        xaxis_title="Valor de Función Objetivo (Z)",
        yaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=370,
        margin=dict(l=10, r=45, t=55, b=20),
    )
    return fig


def grafico_planograma_instancia(
    df_inst: pd.DataFrame,
    df_niveles_tipo: pd.DataFrame,
    color_por: str = "cluster",
) -> go.Figure:
    """Dibuja el planograma físico a escala (0 a ancho_total_cm) por nivel de góndola."""
    df_niv = df_niveles_tipo.sort_values("numero_nivel", ascending=True)
    ancho_total = float(df_niv["ancho_total_cm"].iloc[0])

    etiquetas_niveles = []
    mapa_etiqueta = {}
    for row in df_niv.itertuples(index=False):
        techo = row.altura_suelo_cm + row.alto_disponible_cm
        label = (
            f"Nivel {int(row.numero_nivel)} · "
            f"{row.altura_suelo_cm:.0f}-{techo:.0f} cm (w={row.peso_visibilidad:.2f})"
        )
        etiquetas_niveles.append(label)
        mapa_etiqueta[int(row.nivel_tipo_id)] = label

    max_score = float(df_inst["score_objetivo"].max()) if not df_inst.empty else 1.0
    min_score = float(df_inst["score_objetivo"].min()) if not df_inst.empty else 0.0
    rango_score = max(max_score - min_score, 1e-9)

    fig = go.Figure()
    clusters_presentes = set()

    for niv_row in df_niv.itertuples(index=False):
        nid = int(niv_row.nivel_tipo_id)
        label_niv = mapa_etiqueta[nid]
        sub = df_inst[df_inst["nivel_tipo_id"] == nid].sort_values(
            ["cluster", "score_objetivo"], ascending=[False, False]
        )

        x_cursor = 0.0
        for row in sub.itertuples(index=False):
            ancho_bloque = float(row.ancho_cm) * int(row.facings)
            cluster_id = int(row.cluster)
            clusters_presentes.add(cluster_id)

            if color_por == "score_objetivo":
                norm_s = max(0.0, min(1.0, (float(row.score_objetivo) - min_score) / rango_score))
                color_bar = pc.sample_colorscale("YlOrRd", [0.15 + 0.85 * norm_s])[0]
                text_color = "#1E293B" if norm_s < 0.55 else "#FFFFFF"
            else:
                color_bar = _color_cluster(cluster_id)
                text_color = "#FFFFFF"

            nombre = str(row.nombre_producto)
            if ancho_bloque >= 15.0:
                nombre_corto = nombre if len(nombre) <= 13 else nombre[:12] + "…"
                texto = f"{nombre_corto}<br>×{int(row.facings)}"
            elif ancho_bloque >= 9.0:
                texto = f"×{int(row.facings)}"
            else:
                texto = ""

            hover_html = (
                f"<b>{nombre}</b><br>"
                f"Cluster: {cluster_id}<br>"
                f"Facings: {int(row.facings)} ({row.ancho_cm:.1f} cm c/u → {ancho_bloque:.1f} cm)<br>"
                f"Alto envase: {row.alto_cm:.1f} cm (estante: {row.alto_disponible_cm:.0f} cm)<br>"
                f"Precio: Bs. {row.precio:.2f} | Impulso: {row.indice_impulso:.3f}<br>"
                f"Aporte Z: {row.score_objetivo:.4f}<extra></extra>"
            )

            fig.add_trace(
                go.Bar(
                    orientation="h",
                    y=[label_niv],
                    x=[ancho_bloque],
                    base=[x_cursor],
                    text=[texto],
                    textposition="inside",
                    insidetextanchor="middle",
                    textfont=dict(color=text_color, size=11),
                    marker=dict(color=color_bar, line=dict(color="#FFFFFF", width=1.8)),
                    hovertemplate=hover_html,
                    showlegend=False,
                )
            )
            x_cursor += ancho_bloque

        libre = max(0.0, ancho_total - x_cursor)
        if libre > 0.5:
            fig.add_trace(
                go.Bar(
                    orientation="h",
                    y=[label_niv],
                    x=[libre],
                    base=[x_cursor],
                    marker=dict(
                        color="rgba(148, 163, 184, 0.22)",
                        line=dict(color="rgba(148, 163, 184, 0.5)", width=1),
                    ),
                    hovertemplate=f"<b>Espacio libre</b><br>{libre:.1f} cm disponibles<extra></extra>",
                    showlegend=False,
                )
            )

    if color_por == "cluster":
        for c in sorted(clusters_presentes):
            fig.add_trace(
                go.Bar(
                    orientation="h",
                    y=[None],
                    x=[None],
                    name=f"Cluster {c}",
                    marker_color=_color_cluster(c),
                    showlegend=True,
                )
            )
        fig.add_trace(
            go.Bar(
                orientation="h",
                y=[None],
                x=[None],
                name="Espacio libre",
                marker_color="rgba(148, 163, 184, 0.35)",
                showlegend=True,
            )
        )

    fig.update_layout(
        title="Planograma Visual a Escala (ancho real = ancho_cm × facings)",
        barmode="overlay",
        xaxis=dict(
            title="Posición horizontal en la góndola (cm)",
            range=[0, ancho_total],
            dtick=10,
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.25)",
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=etiquetas_niveles,
            title="",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=max(360, len(etiquetas_niveles) * 92 + 90),
        margin=dict(l=10, r=20, t=60, b=30),
    )
    return fig