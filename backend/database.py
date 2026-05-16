from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

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

    strategy = relationship("Strategy", back_populates="iterations")

Base.metadata.create_all(bind=engine)
