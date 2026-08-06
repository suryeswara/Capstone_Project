"""
MedVerify AI -- Seed User Account Script

Registers/seeds the user account:
Email: suryeswar.reddy@gmail.com
Password: Reddy@24
Full Name: Suryeswar Reddy
Role: researcher
"""

import os
import sys
import uuid
import logging

# Add backend app to Python path
sys.path.insert(0, os.path.join(os.getcwd(), "medverify-ai-backend"))

from app.db.session import engine, Base, SessionLocal
from app.db.models import UserModel
from app.core.security import hash_password

# Ensure DB tables exist
Base.metadata.create_all(bind=engine)

db = SessionLocal()

email = "suryeswar.reddy@gmail.com".strip().lower()
password = "Reddy@24"
full_name = "Suryeswar Reddy"
role = "researcher"

try:
    existing = db.query(UserModel).filter(UserModel.email == email).first()
    if existing:
        print(f"[INFO] User '{email}' already exists in database (ID: {existing.id}). Updating password...")
        existing.hashed_password = hash_password(password)
        existing.full_name = full_name
        existing.role = role
        db.commit()
        print(f"[SUCCESS] Updated user account '{email}' in database!")
    else:
        user_id = f"usr-{uuid.uuid4().hex[:8]}"
        user = UserModel(
            id=user_id,
            email=email,
            hashed_password=hash_password(password),
            full_name=full_name,
            role=role,
        )
        db.add(user)
        db.commit()
        print(f"[SUCCESS] Registered user account '{email}' in database (ID: {user_id})!")

finally:
    db.close()
