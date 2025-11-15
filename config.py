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

    # Polymarket API
    gamma_api_base_url: str = "https://gamma-api.polymarket.com"

    # Prefiltering Thresholds
    timing_threshold_hours: int = 24
    volume_anomaly_threshold: float = 2.0
    win_rate_threshold: float = 0.75
    min_trades_for_analysis: int = 5

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
