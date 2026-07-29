"""Single-ticker LangGraph workflow — staged generate / adapt / optimize."""
from __future__ import annotations

import json
import re
from typing import List, TypedDict

from langgraph.graph import END, StateGraph

from backtest.cerebro_builder import config_score
from backtest.runner import run_single_backtest
from codegen.extract import (
    ensure_stake_pct_in_config,
    extract_code_and_config,
    inject_stake_pct_param,
    inject_standard_strategy_params,
    normalize_python_source,
    position_size_pct,
    position_sizing_instructions,
    sanitize_bt_code,
    validate_python_syntax,
)
from fixtures import fixtures_enabled, load_single_fixture
from llm import call_llm, is_ready, not_ready_message
from prompts import (
    adapt_strategy_prompt,
    optimize_prompt,
    repair_strategy_prompt,
    strategy_code_prompt,
)


class GraphState(TypedDict, total=False):
    strategy_prompt: str
    adapt_instruction: str
    start_date: str
    end_date: str
    position_size_pct: float
    use_fixtures: bool
    generated_code: str
    current_iteration_number: int
    current_config: dict
    all_iteration_results: List[dict]
    best_config_so_far: dict
    error: str
    generate_only: bool


def _parse_single_strategy(resp: str, pct: float):
    code, config = extract_code_and_config(resp)
    code = normalize_python_source(code)
    code = sanitize_bt_code(code)
    code = inject_stake_pct_param(code, pct)
    code = inject_standard_strategy_params(code, multi=False)
    validate_python_syntax(code)
    config = ensure_stake_pct_in_config(config, pct)
    config.setdefault("stop_loss", None)
    config.setdefault("take_profit", None)
    return code, config


def _generate(state: GraphState) -> GraphState:
    print("--- Node: generate_strategy_code ---")
    try:
        pct = position_size_pct(state, 100.0)
        if fixtures_enabled(bool(state.get("use_fixtures"))):
            print("  [fixtures] Loading pregenerated single-stock EMA crossover strategy")
            code, config = load_single_fixture(pct)
            print(f"  Fixture strategy ({len(code)} chars), config={config}")
            return {
                **state,
                "generated_code": code,
                "current_config": config,
                "current_iteration_number": 1,
                "all_iteration_results": [],
                "best_config_so_far": {},
                "error": None,
            }

        if not is_ready():
            return {**state, "error": not_ready_message()}

        stake_frac = round(pct / 100.0, 4)
        sizing = position_sizing_instructions(pct)
        prompt = strategy_code_prompt(state["strategy_prompt"], sizing, stake_frac)

        text = call_llm(prompt)
        try:
            code, config = _parse_single_strategy(text, pct)
        except Exception as err:
            print(f"  First code-gen parse failed ({err}); repairing …")
            repair = repair_strategy_prompt(
                error=str(err)[:400],
                prev=text[:1500],
                user_prompt=state["strategy_prompt"],
                position_sizing=sizing,
                stake_frac=stake_frac,
                multi=False,
            )
            code, config = _parse_single_strategy(call_llm(repair), pct)

        print(f"  Generated strategy ({len(code)} chars), config={config}")
        return {
            **state,
            "generated_code": code,
            "current_config": config,
            "current_iteration_number": 1,
            "all_iteration_results": [],
            "best_config_so_far": {},
            "error": None,
        }
    except Exception as e:
        print(f"ERROR in generate_strategy_code: {e}")
        return {**state, "error": f"Failed to generate/parse LLM response: {e}"}


def _adapt_strategy(state: GraphState) -> GraphState:
    print("--- Node: adapt_strategy_code ---")
    try:
        pct = position_size_pct(state, 100.0)
        instruction = (state.get("adapt_instruction") or "").strip()
        if not instruction:
            return {**state, "error": "Adapt instruction is empty."}
        if not (state.get("generated_code") or "").strip():
            return {**state, "error": "No generated code to adapt."}

        if fixtures_enabled(bool(state.get("use_fixtures"))):
            code, config = load_single_fixture(pct)
            return {
                **state,
                "generated_code": code,
                "current_config": config,
                "current_iteration_number": 1,
                "all_iteration_results": [],
                "best_config_so_far": {},
                "error": None,
            }

        if not is_ready():
            return {**state, "error": not_ready_message()}

        stake_frac = round(pct / 100.0, 4)
        sizing = position_sizing_instructions(pct)
        prompt = adapt_strategy_prompt(
            user_prompt=state.get("strategy_prompt") or "",
            adapt_instruction=instruction,
            prev_code=state["generated_code"],
            prev_config=state.get("current_config") or {},
            position_sizing=sizing,
            stake_frac=stake_frac,
            multi=False,
        )
        text = call_llm(prompt)
        try:
            code, config = _parse_single_strategy(text, pct)
        except Exception as err:
            print(f"  Adapt parse failed ({err}); repairing …")
            repair = repair_strategy_prompt(
                error=str(err)[:400],
                prev=text[:1500],
                user_prompt=f"{state.get('strategy_prompt')}\nAdapt: {instruction}",
                position_sizing=sizing,
                stake_frac=stake_frac,
                multi=False,
            )
            code, config = _parse_single_strategy(call_llm(repair), pct)

        prior = state.get("strategy_prompt") or ""
        merged_prompt = f"{prior}\n[Adapt] {instruction}".strip()
        print(f"  Adapted strategy ({len(code)} chars), config={config}")
        return {
            **state,
            "strategy_prompt": merged_prompt,
            "generated_code": code,
            "current_config": config,
            "current_iteration_number": 1,
            "all_iteration_results": [],
            "best_config_so_far": {},
            "error": None,
        }
    except Exception as e:
        print(f"ERROR in adapt_strategy_code: {e}")
        return {**state, "error": f"Failed to adapt strategy code: {e}"}


def _run(state: GraphState) -> GraphState:
    iteration = state["current_iteration_number"]
    print(f"--- Node: run_backtest  (iteration {iteration}) ---")
    try:
        pct = position_size_pct(state, 100.0)
        metrics = run_single_backtest(
            strategy_code=state["generated_code"],
            config=state["current_config"],
            strategy_prompt=state["strategy_prompt"],
            start_date=state.get("start_date") or "2010-01-01",
            end_date=state.get("end_date") or "2017-11-10",
            position_size_pct_val=pct,
        )
        print(f"  Metrics: cagr={metrics.get('cagr')} trades={metrics.get('total_trades')}")
        config = ensure_stake_pct_in_config(state["current_config"], pct)
        all_results = state.get("all_iteration_results", []) + [
            {"iteration": iteration, "config": config, "metrics": metrics}
        ]
        best = state.get("best_config_so_far") or {}
        if not best or config_score(metrics) > config_score(best.get("metrics", {})):
            best = {"config": config, "metrics": metrics}
        return {
            **state,
            "all_iteration_results": all_results,
            "best_config_so_far": best,
            "error": None,
        }
    except Exception as e:
        print(f"ERROR in run_backtest: {e}")
        return {**state, "error": str(e)}


def _optimize(state: GraphState) -> GraphState:
    iteration = state["current_iteration_number"]
    print(f"--- Node: optimize_strategy  (was iteration {iteration}) ---")
    if fixtures_enabled(bool(state.get("use_fixtures"))):
        print("  [fixtures] Skipping optimize (single iteration only)")
        return {**state, "current_iteration_number": 3}
    if not is_ready():
        return {**state, "error": not_ready_message()}
    prev = state["all_iteration_results"][-1]
    valid_keys = list(prev["config"].keys())
    slim_metrics = {
        k: v
        for k, v in prev["metrics"].items()
        if k not in ("portfolio_values", "trades")
    }
    prompt = optimize_prompt(
        user_prompt=state["strategy_prompt"],
        code=state["generated_code"],
        prev_iter=prev["iteration"],
        prev_config=prev["config"],
        prev_metrics=slim_metrics,
        valid_keys=valid_keys,
    )
    try:
        text = call_llm(prompt)
        m = re.search(r"```json\n(.*?)```", text, re.DOTALL)
        if not m:
            raise ValueError("Optimisation response missing ```json block.")
        raw = json.loads(m.group(1).strip())
        new_config = {k: raw.get(k, prev["config"][k]) for k in valid_keys}
        if "stake_pct" in prev["config"]:
            new_config["stake_pct"] = prev["config"]["stake_pct"]
        else:
            new_config = ensure_stake_pct_in_config(
                new_config, position_size_pct(state, 100.0)
            )
        print(f"  Optimised config: {new_config}")
        return {
            **state,
            "current_config": new_config,
            "current_iteration_number": iteration + 1,
        }
    except Exception as e:
        print(f"ERROR in optimize_strategy: {e}")
        return {**state, "error": f"Failed to generate/parse new config: {e}"}


def _after_gen(state: GraphState) -> str:
    if state.get("error"):
        return END
    if state.get("generate_only"):
        return END
    return "run_backtest"


def _after_run(state: GraphState) -> str:
    if state.get("error"):
        return END
    if fixtures_enabled(bool(state.get("use_fixtures"))):
        print("  [fixtures] Done after 1 backtest iteration")
        return END
    if state["current_iteration_number"] >= 3:
        return END
    return "optimize_strategy"


workflow = StateGraph(GraphState)
workflow.add_node("generate_strategy_code", _generate)
workflow.add_node("run_backtest", _run)
workflow.add_node("optimize_strategy", _optimize)
workflow.set_entry_point("generate_strategy_code")
workflow.add_conditional_edges(
    "generate_strategy_code", _after_gen, {"run_backtest": "run_backtest", END: END}
)
workflow.add_conditional_edges(
    "run_backtest", _after_run, {"optimize_strategy": "optimize_strategy", END: END}
)
workflow.add_edge("optimize_strategy", "run_backtest")
app = workflow.compile()


def run_single_generate(inputs: dict) -> dict:
    return app.invoke({**inputs, "generate_only": True})


def run_single_adapt(state: dict, adapt_instruction: str) -> dict:
    return _adapt_strategy({**state, "adapt_instruction": adapt_instruction})


def run_single_optimize(state: dict) -> dict:
    if state.get("error"):
        return state
    if not state.get("generated_code"):
        return {**state, "error": "No generated code to optimize."}

    st: GraphState = {
        **state,
        "current_iteration_number": 1,
        "all_iteration_results": [],
        "best_config_so_far": {},
        "error": None,
        "generate_only": False,
    }
    st = _run(st)
    if st.get("error"):
        return st
    while True:
        nxt = _after_run(st)
        if nxt == END or nxt is None:
            break
        if nxt == "optimize_strategy":
            st = _optimize(st)
            if st.get("error"):
                return st
            st = _run(st)
            if st.get("error"):
                return st
        else:
            break
    return st
