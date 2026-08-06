import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import settings

# Absolute path to shared SQLite DB file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SQLITE_DB_PATH = os.path.normpath(os.path.join(PROJECT_ROOT, "medverify_dev.db"))
SQLITE_URL = f"sqlite:///{SQLITE_DB_PATH}"

DATABASE_URL = settings.DATABASE_URL
use_sqlite_env = os.getenv("USE_SQLITE_FALLBACK", "false").lower() == "true"

def _get_engine():
    if use_sqlite_env:
        logging.info(f"Using SQLite fallback database at: {SQLITE_DB_PATH}")
        return create_engine(SQLITE_URL, connect_args={"check_same_thread": False})

    try:
        test_engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        # Test connection
        with test_engine.connect() as conn:
            pass
        logging.info("Connected successfully to PostgreSQL database.")
        return test_engine
    except Exception as e:
        logging.info(f"PostgreSQL connection unavailable ({type(e).__name__}). Using local SQLite fallback at: {SQLITE_DB_PATH}")
        return create_engine(SQLITE_URL, connect_args={"check_same_thread": False})

engine = _get_engine()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
