"""SQLAlchemy models and simple versioned migrations."""
from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Strategy(Base):
    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True, index=True)
    user_prompt = Column(String)
    generated_python_code = Column(Text)
    timestamp = Column(String)
    data_source = Column(String, default="yfinance")
    risk_profile_json = Column(Text, default="{}")
    is_multi_stock = Column(Boolean, default=False)
    tickers_json = Column(Text, default="[]")
    # Full run metadata
    start_date = Column(String, default="")
    end_date = Column(String, default="")
    position_size_pct = Column(Float, default=100.0)
    screening_prompt = Column(Text, default="")
    screening_code = Column(Text, default="")
    best_metrics_json = Column(Text, default="{}")
    trades_json = Column(Text, default="[]")
    explanation = Column(Text, default="")

    iterations = relationship("BacktestIteration", back_populates="strategy")


class BacktestIteration(Base):
    __tablename__ = "backtest_iterations"

    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"))
    iteration_number = Column(Integer)
    config_json = Column(Text)
    cagr = Column(Float)
    max_drawdown = Column(Float)
    avg_win = Column(Float)
    avg_loss = Column(Float)
    win_rate = Column(Float)
    expectancy = Column(Float)
    opt_explanation = Column(Text, default="")
    metrics_json = Column(Text, default="{}")

    strategy = relationship("Strategy", back_populates="iterations")


Base.metadata.create_all(bind=engine)

# Schema version table + ordered column migrations
_MIGRATIONS = [
    ("strategies", "data_source", "TEXT DEFAULT 'yfinance'"),
    ("strategies", "risk_profile_json", "TEXT DEFAULT '{}'"),
    ("strategies", "is_multi_stock", "INTEGER DEFAULT 0"),
    ("strategies", "tickers_json", "TEXT DEFAULT '[]'"),
    ("strategies", "start_date", "TEXT DEFAULT ''"),
    ("strategies", "end_date", "TEXT DEFAULT ''"),
    ("strategies", "position_size_pct", "REAL DEFAULT 100.0"),
    ("strategies", "screening_prompt", "TEXT DEFAULT ''"),
    ("strategies", "screening_code", "TEXT DEFAULT ''"),
    ("strategies", "best_metrics_json", "TEXT DEFAULT '{}'"),
    ("strategies", "trades_json", "TEXT DEFAULT '[]'"),
    ("strategies", "explanation", "TEXT DEFAULT ''"),
    ("backtest_iterations", "opt_explanation", "TEXT DEFAULT ''"),
    ("backtest_iterations", "metrics_json", "TEXT DEFAULT '{}'"),
]


def _table_columns(conn, table: str) -> set:
    rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
    # PRAGMA: cid, name, type, notnull, dflt_value, pk
    return {r[1] for r in rows}


def migrate() -> None:
    with engine.connect() as conn:
        conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS schema_migrations "
                "(id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL)"
            )
        )
        conn.commit()
        for table, col, col_def in _MIGRATIONS:
            mig_name = f"{table}.{col}"
            exists = conn.execute(
                text("SELECT 1 FROM schema_migrations WHERE name = :n"),
                {"n": mig_name},
            ).fetchone()
            if exists:
                continue
            cols = _table_columns(conn, table)
            if col not in cols:
                try:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {col_def}"))
                    conn.commit()
                    print(f"[db] Added column {table}.{col}")
                except Exception as e:
                    print(f"[db] skip {mig_name}: {e}")
            try:
                conn.execute(
                    text("INSERT OR IGNORE INTO schema_migrations(name) VALUES (:n)"),
                    {"n": mig_name},
                )
                conn.commit()
            except Exception:
                pass


try:
    migrate()
except Exception as e:
    print(f"[db] migration warning: {e}")
