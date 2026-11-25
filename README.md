# Sistema Inteligente de Agrupamiento de Productos y Optimización de Góndolas

Prototipo de agente inteligente enfocado en **Aprendizaje No Supervisado** y **Optimización Combinatoria (SSAP)** para analizar transacciones históricas de supermercado, segmentar productos, descubrir patrones de compra conjunta y generar automáticamente **planogramas físicos optimizados a escala**, incorporando un modelo antropométrico de visibilidad basado en la **EDSA 2023 (INE Bolivia)**.

> **Proyecto Académico — Inteligencia Artificial**  
> Universidad Mayor de "San Simón" (UMSS) · Facultad de Ciencias y Tecnología  
> Ingeniería de Sistemas e Informática · **Grupo 23**

---

## Tabla de Contenidos

1. [Descripción General](#descripción-general)
2. [Arquitectura y Estructura del Proyecto](#arquitectura-y-estructura-del-proyecto)
3. [Fundamento Matemático y Técnico](#fundamento-matemático-y-técnico)
4. [Requisitos e Instalación](#requisitos-e-instalación)
5. [Guía de Ejecución](#guía-de-ejecución)
6. [Resultados Experimentales](#resultados-experimentales)
7. [Dashboard Interactivo](#dashboard-interactivo)
8. [Equipo de Desarrollo](#equipo-de-desarrollo)

---

## Descripción General

Tradicionalmente, la distribución de productos en las góndolas de un supermercado se realiza bajo criterios empíricos que no aprovechan los patrones ocultos en los tickets de compra ni la ergonomía visual del cliente. Este sistema integra cuatro etapas automatizadas sobre **Python** y **SQLite**:

- **Ingesta y persistencia relacional:** Carga del historial transaccional (`Groceries_dataset.csv`) en un esquema SQLite de 8 tablas con integridad referencial, categorías, dimensiones físicas (`ancho_cm`, `alto_cm`) y catálogo de mobiliario.
- **Minería de datos y segmentación:** Extracción de reglas de asociación (*FP-Growth*) para cuantificar *soporte*, *confianza* y *lift*; cálculo del índice de compra impulsiva; estandarización *Z-score* y agrupamiento con **K-Means** validado por **Coeficiente de Silueta**.
- **Optimización espacial (SSAP):** Cálculo antropométrico de visibilidad por estante, selección automática del tipo y cantidad de góndolas por categoría, y asignación mediante una estrategia híbrida (**Heurística Greedy + Facings marginales + Búsqueda Local *Swap/Relocate***) con auditoría física integrada.
- **Visualización interactiva:** Dashboard web modular construido con **Streamlit** y **Plotly** para inspeccionar clusters, reglas de asociación, curvas ergonómicas y planogramas a escala real en centímetros, con simulación de parámetros en vivo.

---

## Arquitectura y Estructura del Proyecto

El sistema adopta una arquitectura modular por capas desacopladas:

```text
IA_distribucion_productos/
├── data/
│   ├── Groceries_dataset.csv       # Dataset transaccional de entrada
│   └── planograma_retail.db        # Base de datos SQLite generada (8 tablas)
├── src/
│   ├── db/                         # Capa de Datos
│   │   ├── __init__.py
│   │   ├── init_db.py              # Creación del esquema, tipos de góndola y carga CSV
│   │   └── connection.py           # Conexión centralizada a SQLite (PRAGMA foreign_keys = ON)
│   ├── analytics/                  # Capa de Preprocesamiento y Análisis
│   │   ├── __init__.py
│   │   ├── associations.py         # Minería de reglas 1-a-1 con FP-Growth (mlxtend)
│   │   ├── preprocessing.py        # Índice de impulso y normalización Z-score (StandardScaler)
│   │   ├── clustering.py           # K-Means y selección automática de K por silueta
│   │   └── run_pipeline.py         # Orquestador ejecutable de la capa analítica
│   ├── optimizer/                  # Capa de Optimización Espacial (SSAP)
│   │   ├── __init__.py
│   │   ├── visibility.py           # Pesos de visibilidad w_j (EDSA 2023, H = 161.0 cm)
│   │   ├── gondola_setup.py        # Selección de tipo de góndola y creación automática de instancias
│   │   ├── scoring.py              # Puntaje individual s_i y mapa de afinidades intra-categoría
│   │   ├── greedy_solver.py        # Línea base, Greedy + Facings y Búsqueda Local Swap/Relocate
│   │   └── run_optimizer.py        # Orquestador de optimización y auditoría de restricciones
│   ├── ui/                         # Módulos de la Capa de Presentación
│   │   ├── __init__.py
│   │   ├── data_loader.py          # Lectura cacheada de SQLite y recálculo bajo demanda
│   │   ├── charts.py               # Gráficos interactivos Plotly (Planograma, EDSA, K-Means, Reglas)
│   │   └── views.py                # Renderizado de las 4 pestañas del dashboard
│   └── dashboard.py                # Punto de entrada de la aplicación Streamlit
├── requirements.txt                # Dependencias de Python
├── .gitignore
└── README.md
```

### Esquema Relacional (`planograma_retail.db`)

| Tabla | Propósito |
| :--- | :--- |
| `producto` | Catálogo de 167 productos con categoría, precio, dimensiones (`ancho_cm`, `alto_cm`), frecuencia, `indice_impulso` y `cluster`. |
| `ticket` | Registro único de transacciones agrupadas por cliente y fecha (14,963 tickets). |
| `ticket_item` | Detalle de productos adquiridos en cada ticket (38,006 ítems). |
| `regla_asociacion` | Reglas $A \to B$ extraídas con FP-Growth (`soporte`, `confianza`, `lift`). |
| `tipo_gondola` | Catálogo físico de mobiliario de $100\text{ cm}$ de ancho (`Gran_Volumen`: 3 niveles, `Mixta`: 4 niveles, `Estandar`: 5 niveles). |
| `nivel_tipo` | Altura desde el suelo, espacio vertical útil y `peso_visibilidad` ($w_j$) por nivel. |
| `gondola_instancia` | Instancias físicas creadas automáticamente para cubrir el 100% de cada categoría. |
| `asignacion` | Resultado final del planograma (`instancia_id`, `nivel_tipo_id`, `producto_id`, `facings`, `cluster`, `score_objetivo`). |

---

## Fundamento Matemático y Técnico

### 1. Preprocesamiento y Aprendizaje No Supervisado
- **Normalización *Z-score*:** Las variables `precio`, `frecuencia_venta` e `indice_impulso` se estandarizan mediante `StandardScaler` ($x' = \frac{x - \mu}{\sigma}$) para equilibrar distancias euclidianas.
- **K-Means + Silueta:** Se evalúa $K \in \{2, 3, 4, 5, 6\}$ seleccionando automáticamente el $K$ que maximiza el coeficiente de silueta (con opción manual desde el dashboard para explorar segmentaciones más finas).
- **Reglas de Asociación (FP-Growth):** Extrae reglas de co-ocurrencia evaluando soporte, confianza y lift. Las reglas entre productos de la misma categoría alimentan el bono de proximidad física del optimizador.

### 2. Visibilidad Antropométrica ($w_j$) — EDSA 2023 (INE Bolivia)
En lugar de pesos fijos arbitrarios, la visibilidad de cada estante $j$ se calcula mediante una **campana gaussiana asimétrica** parametrizada por la estatura adulta promedio $H = 161.0\text{ cm}$ (referencia EDSA 2023, configurable en el dashboard):

$$h_{\text{opt}} = 0.80 \cdot H \approx 128.8\text{ cm}, \qquad h_j = \text{altura\_suelo}_j + \frac{\text{alto\_disponible}_j}{2}$$

$$w_j = w_{\min} + (1 - w_{\min}) \cdot \exp\left(-\frac{(h_j - h_{\text{opt}})^2}{2\sigma^2}\right)$$

con $w_{\min} = 0.20$, $\sigma = 22.0\text{ cm}$ si $h_j > h_{\text{opt}}$ (penalización rápida sobre la línea de ojos) y $\sigma = 42.0\text{ cm}$ si $h_j \le h_{\text{opt}}$ (caída gradual hacia la zona de manos y suelo).

### 3. Función Objetivo Extendida (SSAP) y Heurística Híbrida
Cada producto $i$ recibe un puntaje normalizado combinando su índice de impulso y su lift promedio normalizado Min-Max:

$$s_i = \alpha \cdot \text{indice\_impulso}_i + \beta \cdot \text{lift\_norm}_i \qquad (\alpha = 0.60,\; \beta = 0.40)$$

El optimizador maximiza el valor comercial ponderado por visibilidad, elasticidad de *facings* con rendimientos decrecientes ($\theta = 0.30$, tope de $4$ *facings*) y bono de cercanía intra-categoría ($\lambda = 0.15$):

$$\max Z = \sum_{i} \sum_{j} x_{ij} \cdot w_j \cdot s_i \cdot f_{ij}^{\theta} \;+\; \lambda \sum_{(i, k) \in R_{\text{intra}}} y_{ik} \cdot (\text{lift}_{ik} - 1)$$

sujeto a cobertura total del catálogo, holgura vertical ergonómica ($\text{alto}_i + 2.0\text{ cm} \le \text{alto\_disponible}_j$) y límite horizontal por estante ($\sum_i \text{ancho}_i \cdot f_{ij} \le 100\text{ cm}$).

El solver opera en tres fases comparadas contra una **línea base empírica** (orden alfabético desde el nivel inferior):
1. **Fase 1 (Greedy con cohesión de cluster):** Asigna productos priorizando bloques de cluster y $s_i$ descendente hacia los estantes de mayor ganancia marginal.
2. **Fase 2 (Facings marginales):** Distribuye el espacio horizontal libre otorgando $+1$ *facing* al producto con mayor ganancia marginal $\Delta Z / \text{ancho\_cm}$.
3. **Fase 3 (Búsqueda Local *Swap* y *Relocate*):** Explora intercambios 1-a-1 y reubicaciones entre estantes recalculando *facings* hasta alcanzar un óptimo local.

---

## Requisitos e Instalación

- **Python** 3.10 o superior (probado en Python 3.13).
- **Dependencias principales:** `pandas`, `numpy`, `scikit-learn`, `mlxtend`, `streamlit`, `plotly`.

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd IA_distribucion_productos
```

### 2. Crear y activar el entorno virtual

En **Windows (PowerShell)**:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

*(Si PowerShell bloquea la ejecución de scripts en tu equipo, ejecuta primero `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`).*

En **Linux / macOS**:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Guía de Ejecución

Con el entorno virtual `(.venv)` activo y ubicado en la carpeta raíz `IA_distribucion_productos/`, ejecuta el pipeline secuencialmente:

### Paso 1: Inicializar la base de datos desde el CSV

```bash
python -m src.db.init_db
```

Crea `data/planograma_retail.db`, carga los 167 productos, los tickets históricos y los 3 tipos de góndola con sus 12 niveles físicos.

### Paso 2: Ejecutar la Capa de Análisis (FP-Growth + K-Means)

```bash
python -m src.analytics.run_pipeline
```

Genera las reglas de asociación, calcula el `indice_impulso`, normaliza variables y persiste el `cluster` óptimo en SQLite.

### Paso 3: Ejecutar la Capa de Optimización (EDSA 2023 + Greedy + Swap/Relocate)

```bash
python -m src.optimizer.run_optimizer
```

Calcula los pesos $w_j$, instancia las góndolas por categoría, ejecuta la optimización híbrida, audita las restricciones físicas y guarda el planograma final en `asignacion`.

### Paso 4: Iniciar el Dashboard Interactivo

```bash
streamlit run src/dashboard.py
```

La aplicación se abrirá automáticamente en `http://localhost:8501`.

---

## Resultados Experimentales

Con la configuración por defecto ($H = 161.0\text{ cm}$, $\alpha = 0.60$, $\beta = 0.40$, $\theta = 0.30$, $\lambda = 0.15$):

### Análisis No Supervisado
- **Reglas de asociación (FP-Growth):** `36` reglas generadas (`6` dirigidas intra-categoría, equivalentes a `3` pares físicos únicos).
- **Clustering (K-Means):** El óptimo automático por silueta es $K = 2$ ($\text{silueta} = 0.5924$), separando `15` productos de rotación masiva (`frecuencia_prom = 1032.67`, `impulso_prom = 0.32`) de los `152` artículos restantes (`frecuencia_prom = 153.12`, `impulso_prom = 0.14`). El dashboard permite simular manualmente $K \in \{2, 3, 4, 5, 6\}$.

### Comparativa de Optimización (Antes vs. Después)

| Escenario | Valor Objetivo ($Z$) | Mejora frente a Línea Base ($\Delta Z\%$) | Detalle |
| :--- | :---: | :---: | :--- |
| **Línea Base (Antes)** | `8.1717` | — | Reposición alfabética de abajo hacia arriba |
| **Heurística Greedy + Facings** | `15.2196` | **+86.25%** | Prioridad por cluster, $s_i$ y ganancia marginal |
| **Final (Greedy + Swap/Relocate)** | `16.5663` | **+102.73%** | `73` mejoras locales aplicadas |

### Auditoría de Restricciones Físicas
- **Cobertura del catálogo:** `167 / 167` productos ubicados (`100%`) en `10` instancias de góndola (`9` categorías).
- **Violaciones de altura (incluyendo holgura de $2\text{ cm}$):** `0`
- **Violaciones de ancho horizontal ($\le 100\text{ cm}$):** `0`
- **Violaciones de límite de *facings* ($1 \le f \le 4$):** `0`

---

## Dashboard Interactivo

La interfaz en **Streamlit** dispone de un **Panel de Simulación** lateral para ajustar $K$, la estatura promedio $H$ (EDSA 2023) y el balance $\alpha / \beta$ con recálculo bajo demanda, junto con cuatro vistas:

1. **Datos y Catálogo:** KPIs globales, distribución por categoría y explorador filtrable de productos.
2. **Clustering (K-Means):** Curva del Coeficiente de Silueta, tabla de perfil por cluster y gráfico de dispersión configurable.
3. **Reglas de Asociación:** Mapa interactivo *Soporte vs. Confianza* coloreado por *Lift* y filtro de reglas intra-categoría.
4. **Góndola y Planograma:** KPIs comparativos ($Z_{\text{base}}$ vs. $Z_{\text{final}}$), **planograma visual a escala real** (coloreable por cluster o aporte $Z$), curva ergonómica EDSA 2023 y auditoría física en vivo.

---

## Equipo de Desarrollo

**Grupo 23 — Inteligencia Artificial (UMSS, Cochabamba - Bolivia)**

- Almendras Ampuero Mauricio
- Burgoa Quiroga Beymar Hanzel
- Emilio Alejo Roberto Carlos
- Espinoza Rios Roger Rudy
- Felipe Rivera Leyton Ludwin
- Gomez Alanes Alan Mauricio
- Herbas Guzmán Nicole Jhuliana
- Padilla Luizaga Carlos Randal

**Docente:** Ing. Patricia Rodríguez Bilbao