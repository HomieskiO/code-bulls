from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
import json
from datetime import datetime

from graph import app as langgraph_app
from database import SessionLocal, Strategy, BacktestIteration

app = FastAPI()

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class BacktestRequest(BaseModel):
    prompt: str

@app.post("/api/backtest")
def run_backtest_endpoint(request: BacktestRequest, db: Session = Depends(get_db)):
    """
    Receives a natural language trading strategy, runs it through the optimization loop,
    and saves the results to the database.
    """
    inputs = {"strategy_prompt": request.prompt}
    final_state = langgraph_app.invoke(inputs)

    if final_state.get("error"):
        return {"error": final_state["error"]}

    # 1. Create the main Strategy record
    strategy_record = Strategy(
        user_prompt=request.prompt,
        generated_python_code=final_state['generated_code'],
        timestamp=datetime.now().isoformat()
    )
    db.add(strategy_record)
    db.commit()
    db.refresh(strategy_record)

    # 2. Create records for each backtest iteration
    for iteration_result in final_state['all_iteration_results']:
        metrics = iteration_result['metrics']
        iteration_record = BacktestIteration(
            strategy_id=strategy_record.id,
            iteration_number=iteration_result['iteration'],
            config_json=json.dumps(iteration_result['config']),
            cagr=metrics.get('cagr'),
            max_drawdown=metrics.get('max_drawdown'),
            avg_win=metrics.get('avg_win'),
            avg_loss=metrics.get('avg_loss'),
            win_rate=metrics.get('win_rate'),
            expectancy=metrics.get('expectancy')
        )
        db.add(iteration_record)
    
    db.commit()

    return {
        "strategy_id": strategy_record.id,
        "best_configuration": final_state['best_config_so_far'],
        "all_iterations": final_state['all_iteration_results']
    }

@app.get("/api/history")
def get_history(db: Session = Depends(get_db)):
    """
    Retrieves all past strategy runs and their results.
    """
    strategies = db.query(Strategy).all()
    return strategies
