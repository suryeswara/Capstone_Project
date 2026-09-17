"""
MedVerify AI — Database Schema Migration & Sync Script

Adds new columns (population_analysis, applicability_score, population_match_type)
to existing SQLite database tables if missing.
"""

import os
import sys
import sqlite3

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
db_paths = [
    os.path.join(PROJECT_ROOT, "medverify_dev.db"),
    os.path.join(PROJECT_ROOT, "medverify-ai-backend", "medverify_dev.db"),
]

print("=" * 80)
print("MEDVERIFY AI — DATABASE SCHEMA SYNC")
print("=" * 80)

for DB_PATH in db_paths:
    print(f"\nChecking: {DB_PATH}")
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Verifications table updates
        cursor.execute("PRAGMA table_info(verifications)")
        columns = [col[1] for col in cursor.fetchall()]
        if "population_analysis" not in columns:
            print("  + Adding column 'population_analysis' to 'verifications' table...")
            cursor.execute("ALTER TABLE verifications ADD COLUMN population_analysis JSON")
        if "faithfulness_score" not in columns:
            print("  + Adding column 'faithfulness_score' to 'verifications' table...")
            cursor.execute("ALTER TABLE verifications ADD COLUMN faithfulness_score FLOAT")
        if "system_confidence" not in columns:
            print("  + Adding column 'system_confidence' to 'verifications' table...")
            cursor.execute("ALTER TABLE verifications ADD COLUMN system_confidence FLOAT")

        # Evidence citations table updates
        cursor.execute("PRAGMA table_info(evidence_citations)")
        col_evidence = [col[1] for col in cursor.fetchall()]
        cols_to_add = [
            ("applicability_score", "FLOAT DEFAULT 1.0"),
            ("final_weight", "FLOAT DEFAULT 0.5"),
            ("pmid", "VARCHAR(50)"),
            ("doi", "VARCHAR(100)"),
            ("similarity", "FLOAT DEFAULT 0.0"),
            ("p_age", "FLOAT DEFAULT 0.5"),
            ("p_sex", "FLOAT DEFAULT 0.5"),
            ("p_condition", "FLOAT DEFAULT 0.5"),
            ("p_region", "FLOAT DEFAULT 0.5"),
            ("population_match_type", "VARCHAR(50) DEFAULT 'UNKNOWN'"),
        ]
        for col_name, col_type in cols_to_add:
            if col_name not in col_evidence:
                print(f"  + Adding column '{col_name}' to 'evidence_citations' table...")
                cursor.execute(f"ALTER TABLE evidence_citations ADD COLUMN {col_name} {col_type}")

        # Model versions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                id VARCHAR(50) PRIMARY KEY,
                component VARCHAR(100) NOT NULL,
                model_name VARCHAR(150) NOT NULL,
                version VARCHAR(50) NOT NULL,
                hash_signature VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()
        print("  [SUCCESS] Schema synchronized cleanly!")

print("\nALL DATABASES SYNCHRONIZED CLEANLY!")

