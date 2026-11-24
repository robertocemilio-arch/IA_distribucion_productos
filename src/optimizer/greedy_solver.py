from dataclasses import dataclass, field
import sqlite3
import pandas as pd

from src.optimizer.gondola_setup import HOLGURA_VERTICAL_CM, crear_instancia_gondola

THETA_DEFAULT = 0.30
LAMBDA_AFINIDAD_DEFAULT = 0.15
MAX_FACINGS_DEFAULT = 4
COHESION_CLUSTER_BONUS = 0.02


@dataclass
class Slot:
    """Representa un estante físico específico (instancia_id + nivel_tipo_id)."""
    slot_id: int
    instancia_id: int
    nivel_tipo_id: int
    numero_nivel: int
    ancho_total_cm: float
    alto_disponible_cm: float
    peso_visibilidad: float
    productos: dict[int, int] = field(default_factory=dict)  # producto_id -> facings

    def ancho_usado(self, prod_info: dict[int, dict]) -> float:
        return sum(prod_info[pid]["ancho_cm"] * f for pid, f in self.productos.items())

    def ancho_libre(self, prod_info: dict[int, dict]) -> float:
        return self.ancho_total_cm - self.ancho_usado(prod_info)

    def cabe_altura(self, alto_cm: float, holgura_cm: float) -> bool:
        return (alto_cm + holgura_cm) <= self.alto_disponible_cm + 1e-9

    def clonar_vacio(self) -> "Slot":
        return Slot(
            slot_id=self.slot_id,
            instancia_id=self.instancia_id,
            nivel_tipo_id=self.nivel_tipo_id,
            numero_nivel=self.numero_nivel,
            ancho_total_cm=self.ancho_total_cm,
            alto_disponible_cm=self.alto_disponible_cm,
            peso_visibilidad=self.peso_visibilidad,
        )


def factor_proximidad(slot_a: Slot, slot_b: Slot) -> float:
    """Devuelve 1.0 si comparten estante, 0.5 si son niveles adyacentes en la misma góndola."""
    if slot_a.instancia_id != slot_b.instancia_id:
        return 0.0
    dist = abs(slot_a.numero_nivel - slot_b.numero_nivel)
    if dist == 0:
        return 1.0
    if dist == 1:
        return 0.5
    return 0.0


def evaluar_categoria(
    slots: list[Slot],
    prod_info: dict[int, dict],
    afinidades_cat: dict[tuple[int, int], float],
    theta: float = THETA_DEFAULT,
    lambda_afinidad: float = LAMBDA_AFINIDAD_DEFAULT,
) -> tuple[float, dict[int, float]]:
    """Evalúa la función objetivo Z y el aporte individual prorrateado por producto."""
    aportes: dict[int, float] = {}
    ubicacion: dict[int, Slot] = {}

    for slot in slots:
        for pid, facings in slot.productos.items():
            ubicacion[pid] = slot
            aportes[pid] = (
                slot.peso_visibilidad
                * prod_info[pid]["score_si"]
                * (facings ** theta)
            )

    # Bono de afinidad intra-categoría prorrateado 50/50 para que sum(aportes) == Z
    for (u, v), lift in afinidades_cat.items():
        if u in ubicacion and v in ubicacion:
            y_uv = factor_proximidad(ubicacion[u], ubicacion[v])
            if y_uv > 0.0:
                bono = lambda_afinidad * y_uv * (lift - 1.0)
                aportes[u] += bono / 2.0
                aportes[v] += bono / 2.0

    return float(sum(aportes.values())), aportes


def optimizar_facings_slot(
    slot: Slot,
    prod_info: dict[int, dict],
    theta: float = THETA_DEFAULT,
    max_facings: int = MAX_FACINGS_DEFAULT,
) -> None:
    """Asigna facings marginales por mayor ganancia ΔZ/cm con rendimientos decrecientes."""
    for pid in list(slot.productos.keys()):
        slot.productos[pid] = 1

    libre = slot.ancho_libre(prod_info)
    while True:
        mejor_pid = None
        mejor_criterio = (-1.0, -1.0)

        for pid, f in slot.productos.items():
            if f >= max_facings:
                continue
            ancho = prod_info[pid]["ancho_cm"]
            if ancho <= libre + 1e-9:
                delta_z = (
                    slot.peso_visibilidad
                    * prod_info[pid]["score_si"]
                    * (((f + 1) ** theta) - (f ** theta))
                )
                criterio = (delta_z / ancho, prod_info[pid]["frecuencia_venta"] / ancho)
                if criterio > mejor_criterio:
                    mejor_criterio = criterio
                    mejor_pid = pid

        if mejor_pid is None:
            break

        slot.productos[mejor_pid] += 1
        libre -= prod_info[mejor_pid]["ancho_cm"]


def rellenar_facings_ingenuo(
    slot: Slot,
    prod_info: dict[int, dict],
    max_facings: int = MAX_FACINGS_DEFAULT,
) -> None:
    """Reparte ancho libre en orden alfabético para una comparación justa con la línea base."""
    libre = slot.ancho_libre(prod_info)
    orden_alfabetico = sorted(
        slot.productos.keys(), key=lambda pid: prod_info[pid]["nombre_producto"]
    )
    progreso = True
    while progreso:
        progreso = False
        for pid in orden_alfabetico:
            f = slot.productos[pid]
            ancho = prod_info[pid]["ancho_cm"]
            if f < max_facings and ancho <= libre + 1e-9:
                slot.productos[pid] += 1
                libre -= ancho
                progreso = True


def construir_linea_base(
    slots_plantilla: list[Slot],
    prod_ids: list[int],
    prod_info: dict[int, dict],
    afinidades_cat: dict[tuple[int, int], float],
    holgura_cm: float,
    theta: float,
    lambda_afinidad: float,
    max_facings: int,
) -> float:
    """Simula una distribución empírica sin analítica (alfabética de abajo hacia arriba)."""
    slots_base = [s.clonar_vacio() for s in slots_plantilla]
    slots_base.sort(key=lambda s: (s.instancia_id, s.numero_nivel))

    orden_alf = sorted(prod_ids, key=lambda pid: prod_info[pid]["nombre_producto"])
    for pid in orden_alf:
        info = prod_info[pid]
        ubicado = False
        for slot in slots_base:
            if slot.cabe_altura(info["alto_cm"], holgura_cm) and info["ancho_cm"] <= slot.ancho_libre(prod_info) + 1e-9:
                slot.productos[pid] = 1
                ubicado = True
                break
        # Respaldo ante fragmentación extrema en orden alfabético
        if not ubicado:
            candidatos = [s for s in slots_base if s.cabe_altura(info["alto_cm"], holgura_cm)]
            if candidatos:
                elegido = max(candidatos, key=lambda s: s.ancho_libre(prod_info))
                if info["ancho_cm"] <= elegido.ancho_libre(prod_info) + 1e-9:
                    elegido.productos[pid] = 1

    for slot in slots_base:
        rellenar_facings_ingenuo(slot, prod_info, max_facings=max_facings)

    z_base, _ = evaluar_categoria(
        slots_base, prod_info, afinidades_cat, theta=theta, lambda_afinidad=lambda_afinidad
    )
    return z_base


def _cargar_niveles_de_tipo(conn: sqlite3.Connection, tipo_id: int) -> list[dict]:
    df = pd.read_sql_query(
        """SELECT nt.nivel_tipo_id, nt.numero_nivel, tg.ancho_total_cm,
                  nt.alto_disponible_cm, nt.peso_visibilidad
           FROM nivel_tipo nt
           JOIN tipo_gondola tg ON nt.tipo_id = tg.tipo_id
           WHERE nt.tipo_id = ?
           ORDER BY nt.numero_nivel;""",
        conn,
        params=(int(tipo_id),),
    )
    return df.to_dict(orient="records")


def asignar_greedy_categoria(
    conn: sqlite3.Connection,
    categoria: str,
    tipo_id: int,
    slots: list[Slot],
    prod_ids: list[int],
    prod_info: dict[int, dict],
    afinidades_cat: dict[tuple[int, int], float],
    holgura_cm: float,
    lambda_afinidad: float,
    cohesion_bonus: float = COHESION_CLUSTER_BONUS,
) -> None:
    """Fase 1: Asignación greedy ordenada por bloque de cluster y s_i descendente."""
    # Prioridad de clusters según s_i promedio (y frecuencia promedio como desempate)
    grupos_cluster: dict[int, list[int]] = {}
    for pid in prod_ids:
        grupos_cluster.setdefault(prod_info[pid]["cluster"], []).append(pid)

    clusters_ordenados = sorted(
        grupos_cluster.keys(),
        key=lambda c: (
            sum(prod_info[p]["score_si"] for p in grupos_cluster[c]) / len(grupos_cluster[c]),
            sum(prod_info[p]["frecuencia_venta"] for p in grupos_cluster[c]) / len(grupos_cluster[c]),
        ),
        reverse=True,
    )
    rank_cluster = {c: idx for idx, c in enumerate(clusters_ordenados)}

    orden_productos = sorted(
        prod_ids,
        key=lambda pid: (
            rank_cluster[prod_info[pid]["cluster"]],
            -prod_info[pid]["score_si"],
            -prod_info[pid]["frecuencia_venta"],
            -prod_info[pid]["ancho_cm"],
        ),
    )

    ubicacion: dict[int, Slot] = {}
    niveles_tipo_cache: list[dict] | None = None

    for pid in orden_productos:
        info = prod_info[pid]
        factibles = [
            s for s in slots
            if s.cabe_altura(info["alto_cm"], holgura_cm)
            and info["ancho_cm"] <= s.ancho_libre(prod_info) + 1e-9
        ]

        # Si hay fragmentación de ancho por orden de prioridad, expandir dinámicamente 1 instancia
        if not factibles:
            nueva_inst_id = crear_instancia_gondola(conn, tipo_id, categoria)
            if niveles_tipo_cache is None:
                niveles_tipo_cache = _cargar_niveles_de_tipo(conn, tipo_id)
            next_slot_id = max((s.slot_id for s in slots), default=-1) + 1
            for niv in niveles_tipo_cache:
                slots.append(
                    Slot(
                        slot_id=next_slot_id,
                        instancia_id=nueva_inst_id,
                        nivel_tipo_id=int(niv["nivel_tipo_id"]),
                        numero_nivel=int(niv["numero_nivel"]),
                        ancho_total_cm=float(niv["ancho_total_cm"]),
                        alto_disponible_cm=float(niv["alto_disponible_cm"]),
                        peso_visibilidad=float(niv["peso_visibilidad"]),
                    )
                )
                next_slot_id += 1
            factibles = [
                s for s in slots
                if s.cabe_altura(info["alto_cm"], holgura_cm)
                and info["ancho_cm"] <= s.ancho_libre(prod_info) + 1e-9
            ]

        mejor_slot = None
        mejor_puntaje = (-1.0, -1.0, -1.0)

        for slot in factibles:
            val_vis = slot.peso_visibilidad * info["score_si"]

            # Ganancia marginal por afinidad intra-categoría con productos ya ubicados
            val_afin = 0.0
            for otro_pid, otro_slot in ubicacion.items():
                par = (min(pid, otro_pid), max(pid, otro_pid))
                if par in afinidades_cat:
                    y_val = factor_proximidad(slot, otro_slot)
                    if y_val > 0.0:
                        val_afin += lambda_afinidad * y_val * (afinidades_cat[par] - 1.0)

            # Bono suave de cohesión para mantener el mismo cluster en niveles contiguos
            val_cluster = 0.0
            mismo_cluster_en_slot = any(
                prod_info[p]["cluster"] == info["cluster"] for p in slot.productos
            )
            if mismo_cluster_en_slot:
                val_cluster = cohesion_bonus
            else:
                en_adyacente = any(
                    factor_proximidad(slot, otro_slot) == 0.5
                    and prod_info[otro_pid]["cluster"] == info["cluster"]
                    for otro_pid, otro_slot in ubicacion.items()
                )
                if en_adyacente:
                    val_cluster = cohesion_bonus * 0.5

            criterio = (
                val_vis + val_afin + val_cluster,
                slot.peso_visibilidad,
                slot.ancho_libre(prod_info),
            )
            if criterio > mejor_puntaje:
                mejor_puntaje = criterio
                mejor_slot = slot

        mejor_slot.productos[pid] = 1
        ubicacion[pid] = mejor_slot


def busqueda_local_swap_relocate(
    slots: list[Slot],
    prod_info: dict[int, dict],
    afinidades_cat: dict[tuple[int, int], float],
    holgura_cm: float,
    theta: float,
    lambda_afinidad: float,
    max_facings: int,
    max_iter: int = 50,
) -> tuple[float, dict[int, float], int]:
    """Fase 3: Refinamiento por búsqueda local (intercambios 1-a-1 y relocalizaciones)."""
    z_actual, aportes_actuales = evaluar_categoria(
        slots, prod_info, afinidades_cat, theta=theta, lambda_afinidad=lambda_afinidad
    )
    movimientos = 0

    for _ in range(max_iter):
        mejor_z = z_actual
        mejor_mov = None

        # Vecindario 1: Swap (intercambio 1-a-1 entre estantes distintos)
        for i in range(len(slots)):
            slot_a = slots[i]
            if not slot_a.productos:
                continue
            for j in range(i + 1, len(slots)):
                slot_b = slots[j]
                if not slot_b.productos:
                    continue

                base_ancho_a = sum(prod_info[p]["ancho_cm"] for p in slot_a.productos)
                base_ancho_b = sum(prod_info[p]["ancho_cm"] for p in slot_b.productos)

                backup_a = dict(slot_a.productos)
                backup_b = dict(slot_b.productos)

                for u in list(backup_a.keys()):
                    info_u = prod_info[u]
                    if not slot_b.cabe_altura(info_u["alto_cm"], holgura_cm):
                        continue
                    for v in list(backup_b.keys()):
                        info_v = prod_info[v]
                        if not slot_a.cabe_altura(info_v["alto_cm"], holgura_cm):
                            continue

                        nuevo_ancho_a = base_ancho_a - info_u["ancho_cm"] + info_v["ancho_cm"]
                        nuevo_ancho_b = base_ancho_b - info_v["ancho_cm"] + info_u["ancho_cm"]
                        if (
                            nuevo_ancho_a > slot_a.ancho_total_cm + 1e-9
                            or nuevo_ancho_b > slot_b.ancho_total_cm + 1e-9
                        ):
                            continue

                        # Probar intercambio con re-optimización de facings
                        slot_a.productos = {p: 1 for p in backup_a if p != u}
                        slot_a.productos[v] = 1
                        slot_b.productos = {p: 1 for p in backup_b if p != v}
                        slot_b.productos[u] = 1

                        optimizar_facings_slot(slot_a, prod_info, theta=theta, max_facings=max_facings)
                        optimizar_facings_slot(slot_b, prod_info, theta=theta, max_facings=max_facings)

                        z_cand, _ = evaluar_categoria(
                            slots, prod_info, afinidades_cat, theta=theta, lambda_afinidad=lambda_afinidad
                        )
                        if z_cand > mejor_z + 1e-6:
                            mejor_z = z_cand
                            mejor_mov = ("swap", i, j, u, v)

                        slot_a.productos = dict(backup_a)
                        slot_b.productos = dict(backup_b)

        # Vecindario 2: Relocate (mover un producto a otro estante con espacio disponible)
        for i, slot_a in enumerate(slots):
            if not slot_a.productos:
                continue
            backup_a = dict(slot_a.productos)
            for j, slot_b in enumerate(slots):
                if i == j:
                    continue
                backup_b = dict(slot_b.productos)
                base_ancho_b = sum(prod_info[p]["ancho_cm"] for p in backup_b)

                for u in list(backup_a.keys()):
                    info_u = prod_info[u]
                    if not slot_b.cabe_altura(info_u["alto_cm"], holgura_cm):
                        continue
                    if base_ancho_b + info_u["ancho_cm"] > slot_b.ancho_total_cm + 1e-9:
                        continue

                    slot_a.productos = {p: 1 for p in backup_a if p != u}
                    slot_b.productos = {p: 1 for p in backup_b}
                    slot_b.productos[u] = 1

                    optimizar_facings_slot(slot_a, prod_info, theta=theta, max_facings=max_facings)
                    optimizar_facings_slot(slot_b, prod_info, theta=theta, max_facings=max_facings)

                    z_cand, _ = evaluar_categoria(
                        slots, prod_info, afinidades_cat, theta=theta, lambda_afinidad=lambda_afinidad
                    )
                    if z_cand > mejor_z + 1e-6:
                        mejor_z = z_cand
                        mejor_mov = ("relocate", i, j, u, None)

                    slot_a.productos = dict(backup_a)
                    slot_b.productos = dict(backup_b)

        if mejor_mov is None:
            break

        tipo_mov, i, j, u, v = mejor_mov
        slot_a, slot_b = slots[i], slots[j]
        if tipo_mov == "swap":
            del slot_a.productos[u]
            del slot_b.productos[v]
            slot_a.productos[v] = 1
            slot_b.productos[u] = 1
        else:
            del slot_a.productos[u]
            slot_b.productos[u] = 1

        optimizar_facings_slot(slot_a, prod_info, theta=theta, max_facings=max_facings)
        optimizar_facings_slot(slot_b, prod_info, theta=theta, max_facings=max_facings)

        z_actual, aportes_actuales = evaluar_categoria(
            slots, prod_info, afinidades_cat, theta=theta, lambda_afinidad=lambda_afinidad
        )
        movimientos += 1

    return z_actual, aportes_actuales, movimientos


def ejecutar_optimizacion_global(
    conn: sqlite3.Connection,
    df_prod: pd.DataFrame,
    afinidades_intra: dict[tuple[int, int], float],
    holgura_cm: float = HOLGURA_VERTICAL_CM,
    theta: float = THETA_DEFAULT,
    lambda_afinidad: float = LAMBDA_AFINIDAD_DEFAULT,
    max_facings: int = MAX_FACINGS_DEFAULT,
) -> dict:
    """Resuelve todas las categorías (Base vs Greedy vs Greedy+Swap) y persiste en 'asignacion'."""
    prod_info = df_prod.set_index("producto_id").to_dict(orient="index")

    df_slots_db = pd.read_sql_query(
        """SELECT gi.instancia_id, gi.tipo_id, gi.categoria_asignada,
                  nt.nivel_tipo_id, nt.numero_nivel, tg.ancho_total_cm,
                  nt.alto_disponible_cm, nt.peso_visibilidad
           FROM gondola_instancia gi
           JOIN tipo_gondola tg ON gi.tipo_id = tg.tipo_id
           JOIN nivel_tipo nt ON gi.tipo_id = nt.tipo_id
           ORDER BY gi.categoria_asignada, gi.instancia_id, nt.numero_nivel;""",
        conn,
    )

    conn.execute("DELETE FROM asignacion;")

    filas_sql = []
    metricas_categorias = []
    total_z_base = 0.0
    total_z_greedy = 0.0
    total_z_final = 0.0
    total_movimientos = 0

    for categoria, grupo_prod in df_prod.groupby("categoria", sort=True):
        cat_str = str(categoria)
        prod_ids = [int(pid) for pid in grupo_prod["producto_id"].tolist()]
        prod_set = set(prod_ids)

        afinidades_cat = {
            par: lift for par, lift in afinidades_intra.items()
            if par[0] in prod_set and par[1] in prod_set
        }

        sub_slots = df_slots_db[df_slots_db["categoria_asignada"] == cat_str]
        tipo_id = int(sub_slots["tipo_id"].iloc[0])

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

        # Fase 1: Heurística Greedy con cohesión de cluster (puede crear instancia extra si hay fragmentación)
        asignar_greedy_categoria(
            conn=conn,
            categoria=cat_str,
            tipo_id=tipo_id,
            slots=slots,
            prod_ids=prod_ids,
            prod_info=prod_info,
            afinidades_cat=afinidades_cat,
            holgura_cm=holgura_cm,
            lambda_afinidad=lambda_afinidad,
        )

        # Línea base evaluada sobre el mismo conjunto final de instancias
        z_base = construir_linea_base(
            slots_plantilla=slots,
            prod_ids=prod_ids,
            prod_info=prod_info,
            afinidades_cat=afinidades_cat,
            holgura_cm=holgura_cm,
            theta=theta,
            lambda_afinidad=lambda_afinidad,
            max_facings=max_facings,
        )

        # Fase 2: Asignación marginal de facings (Greedy)
        for slot in slots:
            optimizar_facings_slot(slot, prod_info, theta=theta, max_facings=max_facings)

        z_greedy, _ = evaluar_categoria(
            slots, prod_info, afinidades_cat, theta=theta, lambda_afinidad=lambda_afinidad
        )

        # Fase 3: Búsqueda local (Swap + Relocate)
        z_final, aportes_finales, movs = busqueda_local_swap_relocate(
            slots=slots,
            prod_info=prod_info,
            afinidades_cat=afinidades_cat,
            holgura_cm=holgura_cm,
            theta=theta,
            lambda_afinidad=lambda_afinidad,
            max_facings=max_facings,
        )

        # Preparar inserción en SQLite
        ocupaciones = []
        facings_cat = []
        for slot in slots:
            ocupaciones.append((slot.ancho_usado(prod_info) / slot.ancho_total_cm) * 100.0)
            for pid, facings in slot.productos.items():
                facings_cat.append(facings)
                filas_sql.append((
                    slot.instancia_id,
                    slot.nivel_tipo_id,
                    pid,
                    int(facings),
                    int(prod_info[pid]["cluster"]),
                    round(float(aportes_finales[pid]), 6),
                ))

        mejora_pct = ((z_final - z_base) / z_base * 100.0) if z_base > 0 else 0.0
        n_instancias_final = len({s.instancia_id for s in slots})

        metricas_categorias.append({
            "categoria": cat_str,
            "productos": len(prod_ids),
            "instancias": n_instancias_final,
            "Z_base": round(z_base, 4),
            "Z_greedy": round(z_greedy, 4),
            "Z_final": round(z_final, 4),
            "mejora_%": round(mejora_pct, 2),
            "movs_swap": movs,
            "ocupacion_%": round(sum(ocupaciones) / len(ocupaciones), 1),
            "facings_prom": round(sum(facings_cat) / len(facings_cat), 2),
        })

        total_z_base += z_base
        total_z_greedy += z_greedy
        total_z_final += z_final
        total_movimientos += movs

    conn.executemany(
        """INSERT INTO asignacion
           (instancia_id, nivel_tipo_id, producto_id, facings, cluster, score_objetivo)
           VALUES (?, ?, ?, ?, ?, ?);""",
        filas_sql,
    )

    mejora_greedy_pct = ((total_z_greedy - total_z_base) / total_z_base * 100.0) if total_z_base > 0 else 0.0
    mejora_total_pct = ((total_z_final - total_z_base) / total_z_base * 100.0) if total_z_base > 0 else 0.0

    return {
        "z_base": round(total_z_base, 4),
        "z_greedy": round(total_z_greedy, 4),
        "z_final": round(total_z_final, 4),
        "mejora_greedy_pct": round(mejora_greedy_pct, 2),
        "mejora_total_pct": round(mejora_total_pct, 2),
        "movimientos_local_search": total_movimientos,
        "resumen_categorias": pd.DataFrame(metricas_categorias),
        "total_asignaciones": len(filas_sql),
    }