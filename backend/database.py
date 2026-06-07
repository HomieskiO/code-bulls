from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, ForeignKey, Text, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Strategy(Base):
    __tablename__ = "strategies"

    id                   = Column(Integer, primary_key=True, index=True)
    user_prompt          = Column(String)
    generated_python_code = Column(Text)
    timestamp            = Column(String)
    data_source          = Column(String, default="yfinance")
    risk_profile_json    = Column(Text,   default="{}")
    is_multi_stock       = Column(Boolean, default=False)
    tickers_json         = Column(Text,   default="[]")

    iterations = relationship("BacktestIteration", back_populates="strategy")


class BacktestIteration(Base):
    __tablename__ = "backtest_iterations"

    id               = Column(Integer, primary_key=True, index=True)
    strategy_id      = Column(Integer, ForeignKey("strategies.id"))
    iteration_number = Column(Integer)
    config_json      = Column(Text)
    cagr             = Column(Float)
    max_drawdown     = Column(Float)
    avg_win          = Column(Float)
    avg_loss         = Column(Float)
    win_rate         = Column(Float)
    expectancy       = Column(Float)
    opt_explanation  = Column(Text, default="")

    strategy = relationship("Strategy", back_populates="iterations")


Base.metadata.create_all(bind=engine)


def _migrate():
    """Add new columns to existing DB without losing data (SQLite ALTER TABLE)."""
    with engine.connect() as conn:
        existing = {
            row[1]
            for row in conn.execute(
                engine.dialect.get_columns.__func__(  # type: ignore[attr-defined]
                    engine.dialect, conn, "strategies"
                )
            )
        } if False else set()

        safe_alters = [
            ("strategies",         "data_source",       "TEXT DEFAULT 'yfinance'"),
            ("strategies",         "risk_profile_json",  "TEXT DEFAULT '{}'"),
            ("strategies",         "is_multi_stock",     "INTEGER DEFAULT 0"),
            ("strategies",         "tickers_json",       "TEXT DEFAULT '[]'"),
            ("backtest_iterations", "opt_explanation",   "TEXT DEFAULT ''"),
        ]
        for table, col, col_def in safe_alters:
            try:
                conn.execute(
                    engine.dialect.statement_compiler(  # type: ignore[attr-defined]
                        engine.dialect, None
                    ).visit_alter_table_add_column
                )
            except Exception:
                pass
            try:
                conn.exec_driver_sql(
                    f"ALTER TABLE {table} ADD COLUMN {col} {col_def}"
                )
                conn.commit()
                print(f"[db] Added column {table}.{col}")
            except Exception:
                pass  # column already exists


try:
    _migrate()
except Exception:
    pass  # best-effort migration