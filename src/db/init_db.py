import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

# --- Configuración de rutas ---
BASE_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = BASE_DIR / "data" / "Groceries_dataset.csv"
DB_PATH = BASE_DIR / "data" / "planograma_retail.db"

# Semilla fija para que precios y dimensiones estimadas sean reproducibles
RNG = np.random.default_rng(42)

# Perfiles por categoría: (palabras_clave, rango_precio, rango_ancho_cm, rango_alto_cm)
PERFILES_CATEGORIA = {
    "Bebidas": (
        ["beer", "wine", "soda", "water", "juice", "liquor", "brandy", "whisky", "rum", "prosecco", "tea", "coffee", "cocoa", "beverages"],
        (8.0, 45.0), (7.0, 11.0), (20.0, 34.0),
    ),
    "Lacteos_y_Huevos": (
        ["milk", "yogurt", "cheese", "butter", "cream", "curd", "eggs", "margarine"],
        (6.0, 28.0), (7.0, 14.0), (10.0, 24.0),
    ),
    "Panaderia_y_Cereales": (
        ["bread", "rolls", "buns", "pastry", "cake", "waffles", "zwieback", "cereals", "flour", "pasta", "rice", "oat"],
        (5.0, 25.0), (10.0, 22.0), (14.0, 32.0),
    ),
    "Snacks_y_Dulces": (
        ["chocolate", "candy", "snack", "chips", "popcorn", "chewing gum", "salty", "sweet", "ice cream", "pudding", "jam", "honey", "sugar"],
        (4.0, 22.0), (6.0, 16.0), (10.0, 24.0),
    ),
    "Frutas_y_Verduras": (
        ["fruit", "vegetable", "berries", "citrus", "grapes", "onions", "herbs", "potato", "salad"],
        (5.0, 18.0), (12.0, 20.0), (10.0, 18.0),
    ),
    "Carnes_y_Embutidos": (
        ["meat", "beef", "pork", "chicken", "turkey", "sausage", "ham", "frankfurter", "fish", "liver"],
        (15.0, 55.0), (12.0, 20.0), (8.0, 16.0),
    ),
    "Limpieza_y_Hogar": (
        ["cleaner", "detergent", "soap", "softener", "hygiene", "napkins", "towels", "bags", "candles", "dishes", "light bulbs", "cookware", "bathroom", "kitchen"],
        (10.0, 48.0), (10.0, 24.0), (18.0, 38.0),
    ),
    "Conservas_y_Abarrotes": (
        ["canned", "frozen", "oil", "vinegar", "sauces", "spices", "salt", "mustard", "ketchup", "mayonnaise", "soups", "pickled", "baking"],
        (6.0, 30.0), (7.0, 12.0), (10.0, 22.0),
    ),
}


def clasificar_y_dimensionar(nombre_producto: str) -> tuple[str, float, float, float]:
    """Asigna categoría, precio, ancho_cm y alto_cm estimados según el nombre."""
    nombre_lower = nombre_producto.lower()
    for categoria, (keywords, r_precio, r_ancho, r_alto) in PERFILES_CATEGORIA.items():
        if any(kw in nombre_lower for kw in keywords):
            precio = round(float(RNG.uniform(*r_precio)), 2)
            ancho = round(float(RNG.uniform(*r_ancho)), 1)
            alto = round(float(RNG.uniform(*r_alto)), 1)
            return categoria, precio, ancho, alto

    # Perfil por defecto para artículos misceláneos
    return (
        "Otros_Abarrotes",
        round(float(RNG.uniform(7.0, 28.0)), 2),
        round(float(RNG.uniform(8.0, 15.0)), 1),
        round(float(RNG.uniform(12.0, 24.0)), 1),
    )


def crear_esquema(conn: sqlite3.Connection) -> None:
    """Crea todas las tablas del esquema relacional en SQLite."""
    conn.executescript("""
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS producto (
            producto_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre_producto   TEXT NOT NULL UNIQUE,
            categoria         TEXT NOT NULL,
            precio            REAL NOT NULL CHECK (precio > 0),
            ancho_cm          REAL NOT NULL CHECK (ancho_cm > 0),
            alto_cm           REAL NOT NULL CHECK (alto_cm > 0),
            frecuencia_venta  INTEGER NOT NULL DEFAULT 0,
            indice_impulso    REAL DEFAULT NULL
        );

        CREATE TABLE IF NOT EXISTS ticket (
            ticket_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            member_number     INTEGER NOT NULL,
            fecha             TEXT NOT NULL,
            UNIQUE(member_number, fecha)
        );

        CREATE TABLE IF NOT EXISTS ticket_item (
            ticket_id         INTEGER NOT NULL,
            producto_id       INTEGER NOT NULL,
            cantidad          INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (ticket_id, producto_id),
            FOREIGN KEY (ticket_id) REFERENCES ticket(ticket_id) ON DELETE CASCADE,
            FOREIGN KEY (producto_id) REFERENCES producto(producto_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS regla_asociacion (
            regla_id          INTEGER PRIMARY KEY AUTOINCREMENT,
            antecedente_id    INTEGER NOT NULL,
            consecuente_id    INTEGER NOT NULL,
            soporte           REAL NOT NULL,
            confianza         REAL NOT NULL,
            lift              REAL NOT NULL,
            FOREIGN KEY (antecedente_id) REFERENCES producto(producto_id),
            FOREIGN KEY (consecuente_id) REFERENCES producto(producto_id)
        );

        CREATE TABLE IF NOT EXISTS tipo_gondola (
            tipo_id           INTEGER PRIMARY KEY,
            nombre            TEXT NOT NULL UNIQUE,
            ancho_total_cm    REAL NOT NULL CHECK (ancho_total_cm > 0),
            num_niveles       INTEGER NOT NULL CHECK (num_niveles > 0)
        );

        CREATE TABLE IF NOT EXISTS nivel_tipo (
            nivel_tipo_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            tipo_id             INTEGER NOT NULL,
            numero_nivel        INTEGER NOT NULL,
            alto_disponible_cm  REAL NOT NULL CHECK (alto_disponible_cm > 0),
            altura_suelo_cm     REAL NOT NULL CHECK (altura_suelo_cm >= 0),
            peso_visibilidad    REAL DEFAULT NULL,
            UNIQUE(tipo_id, numero_nivel),
            FOREIGN KEY (tipo_id) REFERENCES tipo_gondola(tipo_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS gondola_instancia (
            instancia_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            tipo_id             INTEGER NOT NULL,
            categoria_asignada  TEXT DEFAULT NULL,
            FOREIGN KEY (tipo_id) REFERENCES tipo_gondola(tipo_id)
        );

        CREATE TABLE IF NOT EXISTS asignacion (
            asignacion_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            instancia_id      INTEGER NOT NULL,
            nivel_tipo_id     INTEGER NOT NULL,
            producto_id       INTEGER NOT NULL,
            facings           INTEGER NOT NULL DEFAULT 1 CHECK (facings >= 1),
            cluster           INTEGER DEFAULT NULL,
            score_objetivo    REAL DEFAULT NULL,
            FOREIGN KEY (instancia_id) REFERENCES gondola_instancia(instancia_id) ON DELETE CASCADE,
            FOREIGN KEY (nivel_tipo_id) REFERENCES nivel_tipo(nivel_tipo_id),
            FOREIGN KEY (producto_id) REFERENCES producto(producto_id)
        );
    """)


def poblar_tipos_gondola(conn: sqlite3.Connection) -> None:
    """Inserta los 3 tipos de góndolas y sus niveles físicos."""
    tipos = [
        (1, "Gondola_Gran_Volumen", 100.0, 3),
        (2, "Gondola_Mixta", 100.0, 4),
        (3, "Gondola_Estandar", 100.0, 5),
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO tipo_gondola (tipo_id, nombre, ancho_total_cm, num_niveles) VALUES (?, ?, ?, ?);",
        tipos,
    )

    # (tipo_id, numero_nivel, alto_disponible_cm, altura_suelo_cm)
    niveles = [
        # Tipo 1: 3 niveles (Gran Volumen)
        (1, 1, 55.0, 15.0),
        (1, 2, 50.0, 75.0),
        (1, 3, 50.0, 130.0),
        # Tipo 2: 4 niveles (Mixta)
        (2, 1, 42.0, 15.0),
        (2, 2, 38.0, 60.0),
        (2, 3, 38.0, 102.0),
        (2, 4, 36.0, 144.0),
        # Tipo 3: 5 niveles (Estándar)
        (3, 1, 34.0, 15.0),
        (3, 2, 30.0, 52.0),
        (3, 3, 30.0, 85.0),
        (3, 4, 30.0, 118.0),
        (3, 5, 29.0, 151.0),
    ]
    conn.executemany(
        """INSERT OR IGNORE INTO nivel_tipo
           (tipo_id, numero_nivel, alto_disponible_cm, altura_suelo_cm, peso_visibilidad)
           VALUES (?, ?, ?, ?, NULL);""",
        niveles,
    )


def cargar_dataset_csv(conn: sqlite3.Connection, csv_path: Path) -> None:
    """Carga productos, tickets y ticket_items desde el CSV transaccional."""
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo CSV en: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["itemDescription"] = df["itemDescription"].astype(str).str.strip()

    # 1. Insertar productos únicos con frecuencia y dimensiones estimadas
    frecuencias = df["itemDescription"].value_counts()
    productos_rows = []
    for nombre, freq in frecuencias.items():
        categoria, precio, ancho, alto = clasificar_y_dimensionar(nombre)
        productos_rows.append((nombre, categoria, precio, ancho, alto, int(freq)))

    conn.executemany(
        """INSERT OR IGNORE INTO producto
           (nombre_producto, categoria, precio, ancho_cm, alto_cm, frecuencia_venta)
           VALUES (?, ?, ?, ?, ?, ?);""",
        productos_rows,
    )

    # Mapa nombre -> producto_id
    prod_map = dict(
        conn.execute("SELECT nombre_producto, producto_id FROM producto;").fetchall()
    )

    # 2. Insertar tickets únicos por (Member_number, Date)
    tickets_df = df[["Member_number", "Date"]].drop_duplicates()
    conn.executemany(
        "INSERT OR IGNORE INTO ticket (member_number, fecha) VALUES (?, ?);",
        list(tickets_df.itertuples(index=False, name=None)),
    )

    # Mapa (member_number, fecha) -> ticket_id
    ticket_map = {
        (m, f): tid
        for tid, m, f in conn.execute("SELECT ticket_id, member_number, fecha FROM ticket;").fetchall()
    }

    # 3. Insertar ticket_item agrupando duplicados dentro del mismo ticket
    items_agg = (
        df.groupby(["Member_number", "Date", "itemDescription"])
        .size()
        .reset_index(name="cantidad")
    )
    ticket_items_rows = [
        (ticket_map[(row.Member_number, row.Date)], prod_map[row.itemDescription], int(row.cantidad))
        for row in items_agg.itertuples(index=False)
    ]

    conn.executemany(
        "INSERT OR IGNORE INTO ticket_item (ticket_id, producto_id, cantidad) VALUES (?, ?, ?);",
        ticket_items_rows,
    )


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()  # Reinicio limpio en script de un solo uso

    with sqlite3.connect(DB_PATH) as conn:
        crear_esquema(conn)
        poblar_tipos_gondola(conn)
        cargar_dataset_csv(conn, CSV_PATH)
        conn.commit()

        # Resumen de verificación
        n_prod = conn.execute("SELECT COUNT(*) FROM producto;").fetchone()[0]
        n_tick = conn.execute("SELECT COUNT(*) FROM ticket;").fetchone()[0]
        n_item = conn.execute("SELECT COUNT(*) FROM ticket_item;").fetchone()[0]
        n_niv = conn.execute("SELECT COUNT(*) FROM nivel_tipo;").fetchone()[0]

    print(f"Base de datos creada exitosamente en: {DB_PATH}")
    print(f"  - Productos cargados: {n_prod}")
    print(f"  - Tickets generados:  {n_tick}")
    print(f"  - Ítems de tickets:   {n_item}")
    print(f"  - Niveles de góndola: {n_niv} (en 3 tipos)")


if __name__ == "__main__":
    main()