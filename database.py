import hashlib
import secrets
import sqlite3
from pathlib import Path

# Always use the database next to this file, whichever folder the server is started from
DB_NAME = str(Path(__file__).parent / "agritech.db")

# Demo account created by seed_data(), so the sample farm can be explored.
# Change or remove this before putting the app on the internet.
DEMO_EMAIL = "kiprop@agrimanager.local"
DEMO_PASSWORD = "demo1234"


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000
    ).hex()
    return f"{salt}${digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    if not stored_hash or "$" not in stored_hash:
        return False
    salt, digest = stored_hash.split("$", 1)
    check = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000
    ).hex()
    return secrets.compare_digest(check, digest)


def get_connection():
    """Establishes a SQLite connection with foreign keys and row-by-name access enabled."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _ensure_users_columns(cursor):
    """Safely adds newer account columns if upgrading an existing users table."""
    cursor.execute("PRAGMA table_info(users)")
    existing_cols = {row["name"] for row in cursor.fetchall()}

    migrations = {
        "email": "ALTER TABLE users ADD COLUMN email TEXT",
        "phone_number": "ALTER TABLE users ADD COLUMN phone_number TEXT",
        "password_hash": "ALTER TABLE users ADD COLUMN password_hash TEXT",
        "created_at": "ALTER TABLE users ADD COLUMN created_at TEXT",
    }

    for col_name, alter_sql in migrations.items():
        if col_name not in existing_cols:
            cursor.execute(alter_sql)


def init_db():
    """Initializes the database schema across operations, financials, AI chat, and harvest inventory."""
    conn = get_connection()
    try:
        cursor = conn.cursor()

        # 1. Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT,
                phone_number TEXT,
                location TEXT DEFAULT 'Kenya',
                password_hash TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        _ensure_users_columns(cursor)

        # 2. Plots table (Linked to users)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS plots (
                plot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                plot_name TEXT NOT NULL,
                size_acres REAL,
                soil_type TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
            )
        """)

        # 3. Crop Cycles table (Linked to plots)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS crop_cycles (
                cycle_id INTEGER PRIMARY KEY AUTOINCREMENT,
                plot_id INTEGER NOT NULL,
                crop_name TEXT NOT NULL,
                planting_date TEXT,
                status TEXT DEFAULT 'Active',
                FOREIGN KEY (plot_id) REFERENCES plots (plot_id) ON DELETE CASCADE
            )
        """)

        # 4. Farm Logs table (Linked to crop cycles)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS farm_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_id INTEGER NOT NULL,
                log_type TEXT NOT NULL,
                description TEXT NOT NULL,
                log_date TEXT NOT NULL,
                FOREIGN KEY (cycle_id) REFERENCES crop_cycles (cycle_id) ON DELETE CASCADE
            )
        """)

        # 5. Financial Ledger table (Linked to crop cycles)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS financial_ledger (
                transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_id INTEGER,
                transaction_type TEXT NOT NULL, -- 'Expense' or 'Income'
                category TEXT NOT NULL,         -- e.g., 'Seeds', 'Fertilizer', 'Labor', 'Harvest Sale'
                amount REAL NOT NULL,
                transaction_date TEXT NOT NULL,
                notes TEXT,
                FOREIGN KEY (cycle_id) REFERENCES crop_cycles (cycle_id)
            )
        """)

        # 6. Chat History table (Linked to users for conversational memory)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                role TEXT NOT NULL, -- 'user' or 'model'
                message TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        """)

        # 7. Harvest Inventory table (Linked to crop cycles)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS harvest_inventory (
                harvest_id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_id INTEGER NOT NULL,
                harvest_date TEXT NOT NULL,
                quantity_kg REAL NOT NULL,
                remaining_kg REAL NOT NULL,
                storage_location TEXT DEFAULT 'Main Store',
                notes TEXT,
                FOREIGN KEY (cycle_id) REFERENCES crop_cycles (cycle_id)
            )
        """)

        # 8. Login sessions (one row per logged-in browser)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
        session_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL DEFAULT (DATETIME('now')),
        created_at TEXT NOT NULL DEFAULT (DATETIME('now')),
        updated_at TEXT NOT NULL DEFAULT (DATETIME('now')),
        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
    """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
        message_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (DATETIME('now')),
        FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
    """)

        # The first version of chat_messages named its id column "messages_id";
        # every query uses "message_id", so rename it in databases created back then.
        cursor.execute("PRAGMA table_info(chat_messages)")
        if "messages_id" in {row["name"] for row in cursor.fetchall()}:
            cursor.execute("ALTER TABLE chat_messages RENAME COLUMN messages_id TO message_id")

        # Older databases seeded the demo farmer with no email or password,
        # so nobody could log in to see the sample farm. Give it the demo login.
        cursor.execute("""
            UPDATE users
            SET email = ?, password_hash = ?
            WHERE name = 'Kiprop Koech' AND password_hash IS NULL
              AND (email IS NULL OR email = ?)
        """, (DEMO_EMAIL, hash_password(DEMO_PASSWORD), DEMO_EMAIL))

        conn.commit()
        print("Database schema initialized successfully.")
    finally:
        conn.close()


def seed_data():
    """Checks if the users table is empty, and if so, inserts sample Kenyan farmer data."""
    conn = get_connection()
    try:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM users;")
        user_count = cursor.fetchone()[0]

        if user_count == 0:
            print("Users table is empty. Seeding initial data...")

            # 1. Insert 1 sample Kenyan farmer
            cursor.execute("""
                INSERT INTO users (name, email, phone_number, location, password_hash)
                VALUES (?, ?, ?, ?, ?)
            """, ("Kiprop Koech", DEMO_EMAIL, "+254712345678", "Eldoret, Uasin Gishu", hash_password(DEMO_PASSWORD)))
            user_id = cursor.lastrowid

            # 2. Insert 2 plots
            cursor.execute("""
                INSERT INTO plots (user_id, plot_name, size_acres, soil_type)
                VALUES (?, ?, ?, ?)
            """, (user_id, "Lower Valley Maize Block", 3.5, "Well-drained Loam"))
            plot1_id = cursor.lastrowid

            cursor.execute("""
                INSERT INTO plots (user_id, plot_name, size_acres, soil_type)
                VALUES (?, ?, ?, ?)
            """, (user_id, "Greenhouse Section A", 0.5, "Volcanic Red Clay"))
            plot2_id = cursor.lastrowid

            # 3. Insert 2 active crop cycles (Maize and Tomatoes)
            cursor.execute("""
                INSERT INTO crop_cycles (plot_id, crop_name, planting_date, status)
                VALUES (?, ?, ?, ?)
            """, (plot1_id, "Maize (H614)", "2026-03-15", "Active"))
            cycle1_id = cursor.lastrowid

            cursor.execute("""
                INSERT INTO crop_cycles (plot_id, crop_name, planting_date, status)
                VALUES (?, ?, ?, ?)
            """, (plot2_id, "Tomatoes (Anna F1)", "2026-04-01", "Active"))
            cycle2_id = cursor.lastrowid

            # 4. Insert 4 realistic soil input and farm log entries
            logs = [
                (cycle1_id, "Soil Input", "Applied 50kg bag of DAP fertilizer during planting.", "2026-03-15"),
                (cycle1_id, "Soil Input", "Top-dressed with CAN fertilizer after weeding.", "2026-05-10"),
                (cycle2_id, "Soil Input", "Added well-composted manure and basal NPK 17:17:17 fertilizer.", "2026-04-01"),
                (cycle2_id, "Management", "Completed first round of copper-based fungicide spray and trellis staking.", "2026-04-20"),
            ]

            cursor.executemany("""
                INSERT INTO farm_logs (cycle_id, log_type, description, log_date)
                VALUES (?, ?, ?, ?)
            """, logs)

            conn.commit()
            print("Sample data seeded successfully!")
        else:
            print("Database already contains user records. Skipping seeding.")
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    seed_data()