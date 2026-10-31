"""
Tiny additive migration for databases created before the pipeline rewrite.

``Base.metadata.create_all`` never alters existing tables, so new columns are
added here (idempotent) and, on PostgreSQL, the new enum value is registered.
"""

import logging

from sqlalchemy import inspect, text

from app.db.session import Base

logger = logging.getLogger(__name__)


def ensure_schema(engine) -> None:
    insp = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            existing = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in existing:
                    continue
                ddl_type = col.type.compile(dialect=engine.dialect)
                conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN {col.name} {ddl_type}'))
                logger.info("Added column %s.%s", table.name, col.name)

    if engine.dialect.name == "postgresql":
        try:
            with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                conn.execute(text("ALTER TYPE verificationstatusenum ADD VALUE IF NOT EXISTS 'REJECTED_NON_MEDICAL'"))
        except Exception as exc:  # enum may not exist yet on a fresh database
            logger.debug("Enum update skipped: %s", exc)
