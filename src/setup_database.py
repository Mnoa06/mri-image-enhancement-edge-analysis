from pathlib import Path
import sqlite3

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
            width INTEGER NOT NULL CHECK(width > 0),
            height INTEGER NOT NULL CHECK(height > 0),
            dtype TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS processing_results (
            result_id INTEGER PRIMARY KEY,
            image_id INTEGER NOT NULL REFERENCES images(image_id),
            method TEXT NOT NULL,
            parameters_json TEXT NOT NULL DEFAULT '{}',
            output_path TEXT NOT NULL UNIQUE,
            psnr REAL,
            ssim REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)

print("Database ready:", DATABASE_PATH)