import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.middleware import RequestIDMiddleware
from app.api.health import router as health_router
from app.api.verifications import router as verifications_router
from app.api.auth import router as auth_router
from app.db.session import engine, Base
import app.db.models # Ensure models are registered

# Safely initialize database tables on startup
try:
    Base.metadata.create_all(bind=engine)
    logging.info("Database tables initialized successfully.")
except Exception as e:
    logging.warning(f"Database initialization warning: {e}. FastAPI starting with lazy connection.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Request ID & Logging Middleware
app.add_middleware(RequestIDMiddleware)

# CORS Middleware (allows medverify-ai-frontend on port 5173 / localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(health_router)
app.include_router(verifications_router)
app.include_router(auth_router)

@app.get("/")
def root():
    return {
        "message": "Welcome to MedVerify AI Backend Gateway",
        "docs": "/docs",
        "health": "/health",
        "verifications_api": "/api/claims"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
