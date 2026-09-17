import os
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.middleware import RequestIDMiddleware
from app.middleware.rate_limiter import RateLimiterMiddleware
from app.middleware.input_sanitizer import InputSanitizerMiddleware
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

# Stage 12: Rate Limiter Middleware
app.add_middleware(RateLimiterMiddleware)

# Stage 12: Input Sanitizer Middleware
app.add_middleware(InputSanitizerMiddleware)

# Request ID & Logging Middleware
app.add_middleware(RequestIDMiddleware)

# CORS Middleware — Stage 12 hardened
# In production, set ALLOWED_ORIGINS env var to restrict origins
allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://localhost:80"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
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

# Stage 12: Error sanitization — prevent stack traces from leaking to clients
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions and return a safe error response."""
    logging.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred. Please try again later.",
            "support": "If this persists, please contact support.",
        },
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
