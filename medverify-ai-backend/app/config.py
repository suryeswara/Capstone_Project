# pyrefly: ignore [missing-import]
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "MedVerify AI Backend"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"
    
    # PostgreSQL Configuration
    POSTGRES_USER: str = "medverify"
    POSTGRES_PASSWORD: str = "medverify_secret"
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: str = "5432"
    POSTGRES_DB: str = "medverify_db"
    
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
