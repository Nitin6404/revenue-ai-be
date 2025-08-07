"""Configuration settings for the Power BI Destroyer application."""
import os
from typing import Dict, Any, Optional
from pydantic import BaseSettings, Field, PostgresDsn, validator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    APP_NAME: str = "Power BI Destroyer"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # API
    API_PREFIX: str = "/api"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # CORS
    BACKEND_CORS_ORIGINS: list[str] = ["*"]
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./test.db")
    
    # LangChain
    LANGCHAIN_TRACING: bool = os.getenv("LANGCHAIN_TRACING", "False").lower() == "true"
    LANGCHAIN_ENDPOINT: Optional[str] = os.getenv("LANGCHAIN_ENDPOINT")
    LANGCHAIN_API_KEY: Optional[str] = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT: Optional[str] = os.getenv("LANGCHAIN_PROJECT", "power-bi-destroyer")
    
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo-16k")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
    
    # LLM Settings
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "4096"))
    TOP_P: float = float(os.getenv("TOP_P", "0.9"))
    FREQUENCY_PENALTY: float = float(os.getenv("FREQUENCY_PENALTY", "0.0"))
    PRESENCE_PENALTY: float = float(os.getenv("PRESENCE_PENALTY", "0.0"))
    
    # Vector Store (if using)
    VECTOR_STORE_URL: Optional[str] = os.getenv("VECTOR_STORE_URL")
    
    # Redis (for caching)
    REDIS_URL: Optional[str] = os.getenv("REDIS_URL")
    
    # Rate limiting
    RATE_LIMIT: str = os.getenv("RATE_LIMIT", "100/minute")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv(
        "LOG_FORMAT",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Validation for database URL
    @validator("DATABASE_URL", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        return PostgresDsn.build(
            scheme="postgresql",
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            host=os.getenv("POSTGRES_SERVER"),
            path=f"/{os.getenv('POSTGRES_DB') or ''}",
        )
    
    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create settings instance
settings = Settings()

# Configure logging
import logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Log configuration at startup
logger.info(f"Loaded configuration for {settings.APP_NAME} v{settings.APP_VERSION}")
logger.debug(f"Debug mode: {settings.DEBUG}")
logger.debug(f"Database URL: {settings.DATABASE_URL}")
logger.debug(f"OpenAI Model: {settings.OPENAI_MODEL}")
