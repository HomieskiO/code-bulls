"""Screened multi-ticker workflow — staged generate / adapt / optimize."""
from __future__ import annotations

import json
import re
from typing import List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from backtest.cerebro_builder import config_score
from backtest.runner import run_multi_backtest_core
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
from fixtures import fixtures_enabled, load_multi_fixture
from llm import call_llm, is_ready, not_ready_message
from prompts import (
    adapt_strategy_prompt,
    multi_strategy_code_prompt,
    optimize_prompt,
    repair_strategy_prompt,
)
from screening.codegen import (
    SCREENING_CODE_PROMPT,
    SCREENING_REPAIR_PROMPT,
    extract_screening_code,
    fixture_screening_code,
)
from screening.engine import MAX_UNIQUE_TICKERS, run_generated_screener



class ScreenGraphState(TypedDict, total=False):
    strategy_prompt: str
    screening_prompt: str
    adapt_instruction: str
    start_date: str
    end_date: str
    position_size_pct: float
    use_fixtures: bool
    generated_code: str
    screening_code: str
    screening_params: dict  # legacy / unused for LLM path; kept for draft compat
    screening_dict: dict
    screening_csv_path: str
    current_iteration_number: int
    current_config: dict
    all_iteration_results: List[dict]
    best_config_so_far: dict
    error: str
    # When True, stop after code generation (no backtest/optimize)
    generate_only: bool
    # When True, skip generate and only run optimize loop
    optimize_only: bool


def _parse_multi_strategy(resp: str, pct: float):
    code, config = extract_code_and_config(resp)
    code = normalize_python_source(code)
    code = sanitize_bt_code(code)
    code = inject_stake_pct_param(code, pct)
    code = inject_standard_strategy_params(code, multi=True)
    validate_python_syntax(code)
    config = ensure_stake_pct_in_config(config, pct)
    config.setdefault("stop_loss", None)
    config.setdefault("take_profit", None)
    config.pop("screening", None)
    return code, config


def _gen_screening(state: ScreenGraphState) -> ScreenGraphState:
    """LLM (or fixture) generates Python that writes the screened-stocks CSV."""
    print("--- Node: generate_screening_code ---")
    try:
        user_rule = (state.get("screening_prompt") or "").strip()
        if not user_rule:
            return {**state, "error": "Screening prompt is empty."}

        start = state.get("start_date") or "2010-01-01"
        end = state.get("end_date") or "2017-11-10"

        if fixtures_enabled(bool(state.get("use_fixtures"))):
            code = fixture_screening_code()
            print(f"  [fixtures] Screening code ({len(code)} chars)")
            return {
                **state,
                "screening_code": code,
                "screening_params": {},
                "error": None,
            }

        if not is_ready():
            return {**state, "error": not_ready_message()}

        prompt = SCREENING_CODE_PROMPT.format(
            prompt=user_rule, start_date=start, end_date=end
        )
        text = call_llm(prompt)
        try:
            code = extract_screening_code(text)
        except Exception as err:
            print(f"  Screening code parse failed ({err}); repairing …")
            repair = SCREENING_REPAIR_PROMPT.format(
                error=str(err)[:400],
                prev=(text or "")[:1500],
                prompt=user_rule,
                start_date=start,
                end_date=end,
            )
            code = extract_screening_code(call_llm(repair))

        print(f"  Screening code OK ({len(code)} chars)")
        return {
            **state,
            "screening_code": code,
            "screening_params": {},
            "error": None,
        }
    except Exception as e:
        print(f"ERROR generate_screening_code: {e}")
        return {**state, "error": f"Failed to generate screening code: {e}"}


def _run_screening(state: ScreenGraphState) -> ScreenGraphState:
    """Execute generated screening Python → CSV → screening_dict (same API as before)."""
    print("--- Node: run_generated_screener ---")
    try:
        code = (state.get("screening_code") or "").strip()
        if not code:
            return {**state, "error": "No screening code to run."}

        start = state.get("start_date") or "2010-01-01"
        end = state.get("end_date") or "2017-11-10"
        screening_dict, screening_code, csv_path = run_generated_screener(
            code,
            start_date=start,
            end_date=end,
            max_unique_tickers=MAX_UNIQUE_TICKERS,
        )
        if not screening_dict:
            return {
                **state,
                "screening_code": screening_code,
                "screening_csv_path": csv_path,
                "error": "Screening returned no results — no tickers passed the criteria.",
            }
        n_pairs = sum(len(v) for v in screening_dict.values())
        print(
            f"  Screening complete: {len(screening_dict)} dates, "
            f"{n_pairs} ticker-day pairs. csv={csv_path}"
        )
        return {
            **state,
            "screening_dict": screening_dict,
            "screening_code": screening_code,
            "screening_csv_path": csv_path,
            "error": None,
        }

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"ERROR run_generated_screener: {e}")
        return {
            **state,
            "screening_code": state.get("screening_code") or "",
            "error": f"Screening failed: {e}",
        }


def _gen_strategy(state: ScreenGraphState) -> ScreenGraphState:
    print("--- Node: generate_multi_strategy_code ---")
    try:
        pct = position_size_pct(state, 10.0)
        if fixtures_enabled(bool(state.get("use_fixtures"))):
            print("  [fixtures] Loading pregenerated multi-stock SMA slope strategy")
            code, config = load_multi_fixture(pct)
            print(f"  Fixture multi strategy OK ({len(code)} chars), config={config}")
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
        prompt = multi_strategy_code_prompt(
            state["strategy_prompt"], sizing, stake_frac
        )

        text = call_llm(prompt)
        try:
            code, config = _parse_multi_strategy(text, pct)
        except Exception as err:
            print(f"  Multi-strategy parse failed ({err}); repairing …")
            repair = repair_strategy_prompt(
                error=str(err)[:400],
                prev=text[:1500],
                user_prompt=state["strategy_prompt"],
                position_sizing=sizing,
                stake_frac=stake_frac,
                multi=True,
            )
            code, config = _parse_multi_strategy(call_llm(repair), pct)

        print(f"  Multi strategy OK ({len(code)} chars), config={config}")
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
        print(f"ERROR generate_multi_strategy_code: {e}")
        return {**state, "error": f"Failed to generate multi-ticker strategy code: {e}"}


def _adapt_strategy(state: ScreenGraphState) -> ScreenGraphState:
    """Rewrite strategy code from user adaptation instruction (keeps screening)."""
    print("--- Node: adapt_multi_strategy_code ---")
    try:
        pct = position_size_pct(state, 10.0)
        instruction = (state.get("adapt_instruction") or "").strip()
        if not instruction:
            return {**state, "error": "Adapt instruction is empty."}
        if not (state.get("generated_code") or "").strip():
            return {**state, "error": "No generated code to adapt."}

        if fixtures_enabled(bool(state.get("use_fixtures"))):
            # Still allow fixture code reload (adapt ignored offline)
            code, config = load_multi_fixture(pct)
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
            multi=True,
        )
        text = call_llm(prompt)
        try:
            code, config = _parse_multi_strategy(text, pct)
        except Exception as err:
            print(f"  Multi-adapt parse failed ({err}); repairing …")
            repair = repair_strategy_prompt(
                error=str(err)[:400],
                prev=text[:1500],
                user_prompt=f"{state.get('strategy_prompt')}\nAdapt: {instruction}",
                position_sizing=sizing,
                stake_frac=stake_frac,
                multi=True,
            )
            code, config = _parse_multi_strategy(call_llm(repair), pct)

        # Merge strategy prompt with adapt history for later explain/optimize
        prior = state.get("strategy_prompt") or ""
        merged_prompt = f"{prior}\n[Adapt] {instruction}".strip()

        print(f"  Multi adapt OK ({len(code)} chars), config={config}")
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
        print(f"ERROR adapt_multi_strategy_code: {e}")
        return {**state, "error": f"Failed to adapt multi-ticker strategy code: {e}"}


def _run_bt(state: ScreenGraphState) -> ScreenGraphState:
    iteration = state["current_iteration_number"]
    print(f"--- Node: run_multi_backtest  (iteration {iteration}) ---")
    try:
        pct = position_size_pct(state, 10.0)
        metrics = run_multi_backtest_core(
            strategy_code=state["generated_code"],
            config=state["current_config"],
            screening_dict=state.get("screening_dict") or {},
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
        import traceback

        traceback.print_exc()
        print(f"ERROR in run_multi_backtest: {e}")
        return {**state, "error": str(e)}


def _optimize(state: ScreenGraphState) -> ScreenGraphState:
    iteration = state["current_iteration_number"]
    print(f"--- Node: optimize_multi_strategy  (was iteration {iteration}) ---")
    if fixtures_enabled(bool(state.get("use_fixtures"))):
        print("  [fixtures] Skipping optimize")
        return {**state, "current_iteration_number": 3}
    if not is_ready():
        return {**state, "error": not_ready_message()}
    prev = state["all_iteration_results"][-1]
    valid_keys = list(prev["config"].keys())
    slim = {
        k: v
        for k, v in prev["metrics"].items()
        if k not in ("portfolio_values", "trades")
    }
    prompt = optimize_prompt(
        user_prompt=state["strategy_prompt"],
        code=state["generated_code"],
        prev_iter=prev["iteration"],
        prev_config=prev["config"],
        prev_metrics=slim,
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
                new_config, position_size_pct(state, 10.0)
            )
        print(f"  Optimised config: {new_config}")
        return {
            **state,
            "current_config": new_config,
            "current_iteration_number": iteration + 1,
        }
    except Exception as e:
        print(f"ERROR optimize_multi: {e}")
        return {**state, "error": f"Failed to generate/parse new config: {e}"}


# ── Conditional edges ────────────────────────────────────────────────────────

def _after_params(state: ScreenGraphState) -> str:
    return END if state.get("error") else "run_screening"


def _after_screen(state: ScreenGraphState) -> str:
    return END if state.get("error") else "generate_multi_strategy_code"


def _after_gen(state: ScreenGraphState) -> str:
    if state.get("error"):
        return END
    # Staged generate: stop after code is ready
    if state.get("generate_only"):
        return END
    return "run_multi_backtest"


def _after_adapt(state: ScreenGraphState) -> str:
    # Adapt always returns to the user (no auto-optimize)
    return END


def _after_run(state: ScreenGraphState) -> str:
    if state.get("error"):
        return END
    if fixtures_enabled(bool(state.get("use_fixtures"))):
        print("  [fixtures] Done after 1 multi backtest iteration")
        return END
    if state["current_iteration_number"] >= 3:
        return END
    return "optimize_multi_strategy"


# ── Full pipeline graph (generate + optional optimize) ───────────────────────

multi_workflow = StateGraph(ScreenGraphState)
multi_workflow.add_node("generate_screening_params", _gen_screening)
multi_workflow.add_node("run_screening", _run_screening)
multi_workflow.add_node("generate_multi_strategy_code", _gen_strategy)
multi_workflow.add_node("run_multi_backtest", _run_bt)
multi_workflow.add_node("optimize_multi_strategy", _optimize)
multi_workflow.set_entry_point("generate_screening_params")
multi_workflow.add_conditional_edges(
    "generate_screening_params",
    _after_params,
    {"run_screening": "run_screening", END: END},
)
multi_workflow.add_conditional_edges(
    "run_screening",
    _after_screen,
    {"generate_multi_strategy_code": "generate_multi_strategy_code", END: END},
)
multi_workflow.add_conditional_edges(
    "generate_multi_strategy_code",
    _after_gen,
    {"run_multi_backtest": "run_multi_backtest", END: END},
)
multi_workflow.add_conditional_edges(
    "run_multi_backtest",
    _after_run,
    {"optimize_multi_strategy": "optimize_multi_strategy", END: END},
)
multi_workflow.add_edge("optimize_multi_strategy", "run_multi_backtest")
multi_app = multi_workflow.compile()


# ── Staged entrypoints (called by services) ──────────────────────────────────

def run_multi_generate(inputs: dict) -> dict:
    """Screen + generate strategy code only (no backtest / config opt)."""
    state = {**inputs, "generate_only": True, "optimize_only": False}
    return multi_app.invoke(state)


def run_multi_adapt(state: dict, adapt_instruction: str) -> dict:
    """Adapt existing code; keep screening_dict. No backtest."""
    return _adapt_strategy({**state, "adapt_instruction": adapt_instruction})


def run_multi_optimize(state: dict) -> dict:
    """Run backtest + config optimization loop on an existing draft."""
    if state.get("error"):
        return state
    if not state.get("generated_code"):
        return {**state, "error": "No generated code to optimize."}
    if not state.get("screening_dict"):
        return {**state, "error": "No screening results — regenerate first."}

    # Reset iteration bookkeeping for a clean optimize run
    st: ScreenGraphState = {
        **state,
        "current_iteration_number": 1,
        "all_iteration_results": [],
        "best_config_so_far": {},
        "error": None,
        "generate_only": False,
    }

    # First backtest
    st = _run_bt(st)
    if st.get("error"):
        return st

    # Optimize loop (same as graph: up to 3 iterations)
    while True:
        nxt = _after_run(st)
        if nxt == END or nxt is None:
            break
        if nxt == "optimize_multi_strategy":
            st = _optimize(st)
            if st.get("error"):
                return st
            st = _run_bt(st)
            if st.get("error"):
                return st
        else:
            break
    return st
