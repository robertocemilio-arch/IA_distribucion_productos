import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from src.optimizer.greedy_solver import (
    LAMBDA_AFINIDAD_DEFAULT,
    MAX_FACINGS_DEFAULT,
    THETA_DEFAULT,
)
from src.optimizer.scoring import ALPHA_DEFAULT
from src.optimizer.visibility import ESTATURA_EDSA_2023_CM, FACTOR_CENTRO_VISUAL
from src.ui.data_loader import cargar_datos_bd, recalcular_sistema
from src.ui.views import (
    render_tab_asociaciones,
    render_tab_clustering,
    render_tab_datos,
    render_tab_gondola,
)


def inicializar_estado() -> None:
    """Inicializa parámetros persistentes de la sesión en Streamlit."""
    st.session_state.setdefault("k_opcion", "Automático (Silueta)")
    st.session_state.setdefault("estatura_cm", ESTATURA_EDSA_2023_CM)
    st.session_state.setdefault("alpha", ALPHA_DEFAULT)
    st.session_state.setdefault("ultimo_opt", None)


def render_sidebar() -> None:
    """Panel de control acotado (K, estatura EDSA 2023 y balance α/β)."""
    st.sidebar.title("Panel de Simulación")
    st.sidebar.caption("Ajusta los parámetros clave y presiona el botón para recalcular.")

    with st.sidebar.form("form_parametros"):
        opciones_k = ["Automático (Silueta)", 2, 3, 4, 5, 6]
        idx_k = opciones_k.index(st.session_state["k_opcion"])
        k_opcion = st.selectbox(
            "Segmentación K-Means (K)",
            options=opciones_k,
            index=idx_k,
            help="Automático elige el K de mayor silueta (K=2); los valores fijos permiten explorar segmentos más finos.",
        )

        estatura_cm = st.slider(
            "Estatura promedio H (cm)",
            min_value=145.0,
            max_value=185.0,
            value=float(st.session_state["estatura_cm"]),
            step=0.5,
            help="Referencia EDSA 2023 (INE Bolivia): 161.0 cm. Determina el centro visual óptimo h_opt = 0.80 × H.",
        )

        alpha = st.slider(
            "Peso del índice de impulso (α)",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state["alpha"]),
            step=0.05,
            help="El peso del lift promedio normalizado se ajusta automáticamente como β = 1 - α.",
        )

        submitted = st.form_submit_button(
            "Recalcular y Optimizar",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        k_fijo = None if k_opcion == "Automático (Silueta)" else int(k_opcion)
        with st.spinner("Actualizando clusters y optimizando góndolas..."):
            res_opt = recalcular_sistema(
                k_fijo=k_fijo,
                estatura_cm=estatura_cm,
                alpha=alpha,
            )
        st.session_state["k_opcion"] = k_opcion
        st.session_state["estatura_cm"] = estatura_cm
        st.session_state["alpha"] = alpha
        st.session_state["ultimo_opt"] = res_opt
        st.sidebar.success(
            f"Listo: K={res_opt['k_elegido']} | Mejora ΔZ = +{res_opt['optimizacion']['mejora_total_pct']:.2f}%"
        )

    alpha_act = float(st.session_state["alpha"])
    beta_act = round(1.0 - alpha_act, 2)
    h_act = float(st.session_state["estatura_cm"])
    h_opt = FACTOR_CENTRO_VISUAL * h_act

    st.sidebar.divider()
    st.sidebar.markdown("**Parámetros activos del modelo**")
    st.sidebar.markdown(
        f"- **Estatura ($H$):** `{h_act:.1f} cm` ($h_{{opt}} = {h_opt:.1f}$ cm)\n"
        f"- **Puntaje ($s_i$):** $\\alpha = {alpha_act:.2f}$, $\\beta = {beta_act:.2f}$\n"
        f"- **Elasticidad facings:** $\\theta = {THETA_DEFAULT:.2f}$ (máx. {MAX_FACINGS_DEFAULT})\n"
        f"- **Afinidad intra-cat.:** $\\lambda = {LAMBDA_AFINIDAD_DEFAULT:.2f}$"
    )


def main() -> None:
    st.set_page_config(
        page_title="Optimización de Planogramas Retail",
        page_icon="🛒",
        layout="wide",
    )
    inicializar_estado()
    render_sidebar()

    st.title("Sistema Inteligente de Distribución de Productos en Góndolas")
    st.caption(
        "Aprendizaje No Supervisado (K-Means + FP-Growth) y Optimización Espacial SSAP "
        "con perfil antropométrico EDSA 2023 (INE Bolivia)"
    )

    alpha_act = float(st.session_state["alpha"])
    beta_act = round(1.0 - alpha_act, 2)
    datos = cargar_datos_bd(alpha=alpha_act, beta=beta_act)

    tab_datos, tab_clusters, tab_reglas, tab_gondola = st.tabs([
        "1. Datos y Catálogo",
        "2. Clustering (K-Means)",
        "3. Reglas de Asociación",
        "4. Góndola y Planograma",
    ])

    with tab_datos:
        render_tab_datos(datos["productos"], datos["stats"])

    with tab_clusters:
        render_tab_clustering(
            datos["productos"],
            datos["eval_k"],
            datos["stats"]["k_actual"],
        )

    with tab_reglas:
        render_tab_asociaciones(datos["reglas"])

    with tab_gondola:
        render_tab_gondola(
            df_asig=datos["asignacion"],
            df_niveles=datos["niveles"],
            resumen_opt=datos["resumen_opt"],
            estatura_cm=float(st.session_state["estatura_cm"]),
            ultimo_opt=st.session_state["ultimo_opt"],
        )


if __name__ == "__main__":
    main()