"""
Database models for Polymarket Insider Trading Detector
Using SQLAlchemy ORM
"""
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Float,
    Integer,
    DateTime,
    Boolean,
    Text,
    JSON,
    ForeignKey
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
from config import settings

Base = declarative_base()


class Market(Base):
    """Market information from Polymarket"""
    __tablename__ = 'markets'

    id = Column(String, primary_key=True)  # condition_id
    question = Column(Text, nullable=False)
    description = Column(Text)
    category = Column(String)
    end_date = Column(DateTime)
    resolved = Column(Boolean, default=False)
    outcome = Column(String)
    volume = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    trades = relationship("Trade", back_populates="market")
    suspicious_transactions = relationship("SuspiciousTransaction", back_populates="market")


class Wallet(Base):
    """Wallet/trader information"""
    __tablename__ = 'wallets'

    address = Column(String, primary_key=True)
    first_seen = Column(DateTime)
    last_seen = Column(DateTime)
    total_trades = Column(Integer, default=0)
    total_volume = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    unique_markets = Column(Integer, default=0)
    
    # Flags
    is_suspicious = Column(Boolean, default=False)
    is_new_account = Column(Boolean, default=True)
    
    # Relationships
    trades = relationship("Trade", back_populates="wallet")
    suspicious_transactions = relationship("SuspiciousTransaction", back_populates="wallet")


class Trade(Base):
    """Individual trade/transaction"""
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(String, ForeignKey('markets.id'))
    wallet_address = Column(String, ForeignKey('wallets.address'))
    
    timestamp = Column(DateTime, nullable=False)
    side = Column(String)  # BUY/SELL
    outcome = Column(String)  # YES/NO
    size = Column(Float)  # Trade size in USD
    price = Column(Float)  # Price at execution
    
    # Transaction details
    tx_hash = Column(String)
    token_id = Column(String)
    
    # Relationships
    market = relationship("Market", back_populates="trades")
    wallet = relationship("Wallet", back_populates="trades")
    
    created_at = Column(DateTime, default=datetime.utcnow)


class SuspiciousTransaction(Base):
    """Flagged suspicious transactions from prefiltering"""
    __tablename__ = 'suspicious_transactions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(Integer, ForeignKey('trades.id'))
    market_id = Column(String, ForeignKey('markets.id'))
    wallet_address = Column(String, ForeignKey('wallets.address'))
    
    # Suspicion scores from detectors
    timing_score = Column(Float, default=0.0)
    volume_score = Column(Float, default=0.0)
    gains_score = Column(Float, default=0.0)
    network_score = Column(Float, default=0.0)
    total_score = Column(Float, default=0.0)
    
    # Flags from detectors
    flags = Column(JSON)  # List of detection flags
    
    # LLM Analysis
    llm_analyzed = Column(Boolean, default=False)
    llm_suspicion_score = Column(Float)
    llm_reasoning = Column(Text)
    llm_analysis_date = Column(DateTime)
    
    # Enriched context (stored as JSON)
    context = Column(JSON)
    
    # Timestamps
    detected_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    market = relationship("Market", back_populates="suspicious_transactions")
    wallet = relationship("Wallet", back_populates="suspicious_transactions")


class KnownInsiderCase(Base):
    """Known/documented insider trading cases for LLM examples"""
    __tablename__ = 'known_insider_cases'

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_name = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    
    market_id = Column(String)
    wallet_address = Column(String)
    
    # Case details
    trade_amount = Column(Float)
    timing_details = Column(Text)
    outcome_details = Column(Text)
    investigation_status = Column(String)
    
    # Full case context for LLM
    full_context = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)


# Database initialization
def init_db(database_url: str = None):
    """Initialize database and create tables"""
    url = database_url or settings.database_url
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    return engine


def get_session(engine):
    """Create a database session"""
    Session = sessionmaker(bind=engine)
    return Session()
