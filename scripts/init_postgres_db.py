"""
MedVerify AI -- PostgreSQL Database Initialization & Seed Script

Run this script to initialize all PostgreSQL tables and seed default data:
1. Tests connection to your PostgreSQL server
2. Creates tables: users, disease_categories, claims, verifications, evidence_citations
3. Seeds initial disease categories (Diabetes, Cardiovascular Disease, Vaccination)
4. Seeds user account (suryeswar.reddy@gmail.com / Reddy@24)
"""

import os
import sys
import uuid
import logging
from sqlalchemy import create_engine

# Add backend app to Python path
sys.path.insert(0, os.path.join(os.getcwd(), "medverify-ai-backend"))

from app.config import settings
from app.db.session import Base
from app.db.models import UserModel, DiseaseCategoryModel
from app.core.security import hash_password

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

print("=" * 80)
print("MEDVERIFY AI -- POSTGRESQL DATABASE INITIALIZATION")
print("=" * 80)

postgres_url = settings.DATABASE_URL
print(f"Target Database URL: {postgres_url}")

try:
    engine = create_engine(postgres_url, pool_pre_ping=True)
    with engine.connect() as conn:
        print("[SUCCESS] Connected to PostgreSQL server!")

    print("\n1. Creating database schema tables...")
    Base.metadata.create_all(bind=engine)
    print("   [OK] Tables created: users, disease_categories, claims, verifications, evidence_citations.")

    # Seed initial data
    from sqlalchemy.orm import sessionmaker
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    print("\n2. Seeding Phase 1 Disease Categories...")
    categories = [
        ("diabetes", "Diabetes", "biobert-base-cased-v1.2-finetuned"),
        ("cardiovascular", "Cardiovascular Disease", "biobert-base-cased-v1.2-finetuned"),
        ("vaccination", "Vaccination", "biobert-base-cased-v1.2-finetuned"),
    ]
    for cid, cname, cver in categories:
        existing = db.query(DiseaseCategoryModel).filter(DiseaseCategoryModel.id == cid).first()
        if not existing:
            cat = DiseaseCategoryModel(id=cid, name=cname, active_phase=1, classifier_model_version=cver)
            db.add(cat)
            print(f"   + Added disease category: {cname}")

    print("\n3. Seeding User Account (suryeswar.reddy@gmail.com)...")
    email = "suryeswar.reddy@gmail.com"
    pwd = "Reddy@24"
    user = db.query(UserModel).filter(UserModel.email == email).first()
    if user:
        user.hashed_password = hash_password(pwd)
        user.full_name = "Suryeswar Reddy"
        user.role = "researcher"
        print(f"   [OK] Updated password for existing user '{email}'")
    else:
        user_id = f"usr-{uuid.uuid4().hex[:8]}"
        user = UserModel(
            id=user_id,
            email=email,
            hashed_password=hash_password(pwd),
            full_name="Suryeswar Reddy",
            role="researcher"
        )
        db.add(user)
        print(f"   + Added user '{email}' (ID: {user_id})")

    db.commit()
    db.close()

    print("\n================================================================================")
    print("POSTGRESQL DATABASE INITIALIZATION COMPLETE! SUCCESS ✅")
    print("================================================================================")

except Exception as e:
    print(f"\n[ERROR] Could not connect to PostgreSQL: {e}")
    print("\nPlease verify:")
    print(" 1. PostgreSQL server service is running on port 5432")
    print(" 2. Database 'medverify_db' and user 'medverify' have been created")
    print(" 3. Credentials in medverify-ai-backend/.env match your PostgreSQL setup")
