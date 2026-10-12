from pathlib import Path
import sqlite3

# Locate the project folder from this script's location.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_ROOT / "database" / "mri_processing.db"

DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

with sqlite3.connect(DATABASE_PATH) as connection:
    connection.execute("PRAGMA foreign_keys = ON")

    connection.executescript("""
        CREATE TABLE IF NOT EXISTS images (
            image_id INTEGER PRIMARY KEY,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL UNIQUE,
            width INTEGER NOT NULL CHECK (width > 0),
            height INTEGER NOT NULL CHECK (height > 0),
            dtype TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS processing_results (
            result_id INTEGER PRIMARY KEY,
            image_id INTEGER NOT NULL,
            method TEXT NOT NULL,
            parameters_json TEXT NOT NULL DEFAULT '{}',
            output_path TEXT NOT NULL UNIQUE,
            psnr REAL,
            ssim REAL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (image_id) REFERENCES images(image_id)
        );
    """)

    tables = connection.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name
    """).fetchall()

print(f"Database: {DATABASE_PATH}")
print("Tables:", ", ".join(row[0] for row in tables))