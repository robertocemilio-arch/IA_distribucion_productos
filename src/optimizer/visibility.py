import math
import sqlite3
import pandas as pd

# Parámetros antropométricos por defecto (referencia EDSA 2023 - INE Bolivia)
ESTATURA_EDSA_2023_CM = 161.0
FACTOR_CENTRO_VISUAL = 0.80  # ~128.8 cm (zona óptima ojos-pecho)
W_MIN = 0.20
SIGMA_ARRIBA = 22.0          # Penalización rápida por encima de la línea visual
SIGMA_ABAJO = 42.0           # Caída más gradual hacia zona de manos/suelo


def calcular_peso_visibilidad(
    altura_suelo_cm: float,
    alto_disponible_cm: float,
    estatura_cm: float = ESTATURA_EDSA_2023_CM,
    factor_centro_visual: float = FACTOR_CENTRO_VISUAL,
    w_min: float = W_MIN,
    sigma_arriba: float = SIGMA_ARRIBA,
    sigma_abajo: float = SIGMA_ABAJO,
) -> float:
    """Calcula w_j en [w_min, 1.0] mediante una campana gaussiana asimétrica."""
    h_nivel = altura_suelo_cm + (alto_disponible_cm / 2.0)
    h_opt = factor_centro_visual * estatura_cm
    sigma = sigma_arriba if h_nivel > h_opt else sigma_abajo

    w_j = w_min + (1.0 - w_min) * math.exp(-((h_nivel - h_opt) ** 2) / (2.0 * sigma**2))
    return round(float(w_j), 4)


def actualizar_pesos_visibilidad(
    conn: sqlite3.Connection,
    estatura_cm: float = ESTATURA_EDSA_2023_CM,
    factor_centro_visual: float = FACTOR_CENTRO_VISUAL,
    w_min: float = W_MIN,
    sigma_arriba: float = SIGMA_ARRIBA,
    sigma_abajo: float = SIGMA_ABAJO,
) -> pd.DataFrame:
    """Calcula y persiste 'peso_visibilidad' para todos los niveles en 'nivel_tipo'."""
    df_niveles = pd.read_sql_query(
        """SELECT nt.nivel_tipo_id, nt.tipo_id, tg.nombre AS tipo_gondola,
                  nt.numero_nivel, nt.altura_suelo_cm, nt.alto_disponible_cm
           FROM nivel_tipo nt
           JOIN tipo_gondola tg ON nt.tipo_id = tg.tipo_id
           ORDER BY nt.tipo_id, nt.numero_nivel;""",
        conn,
    )

    df_niveles["altura_media_cm"] = (
        df_niveles["altura_suelo_cm"] + df_niveles["alto_disponible_cm"] / 2.0
    ).round(1)

    df_niveles["peso_visibilidad"] = [
        calcular_peso_visibilidad(
            row.altura_suelo_cm,
            row.alto_disponible_cm,
            estatura_cm=estatura_cm,
            factor_centro_visual=factor_centro_visual,
            w_min=w_min,
            sigma_arriba=sigma_arriba,
            sigma_abajo=sigma_abajo,
        )
        for row in df_niveles.itertuples(index=False)
    ]

    updates = [
        (float(row.peso_visibilidad), int(row.nivel_tipo_id))
        for row in df_niveles.itertuples(index=False)
    ]
    conn.executemany(
        "UPDATE nivel_tipo SET peso_visibilidad = ? WHERE nivel_tipo_id = ?;",
        updates,
    )
    return df_niveles