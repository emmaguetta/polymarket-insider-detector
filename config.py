"""
Configuration management for Polymarket Insider Trading Detector
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Anthropic API
    anthropic_api_key: str

    # Database
    database_url: str = "sqlite:///polymarket_insider.db"

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
