"""Tests for tools.builtin.calculator math.* whitelist.

Covers [IMPROVE-25]. Before this commit the calculator accepted only
arithmetic (+ - * / % ** and unary -). Agents needing sqrt / sin / pi
fell back to run_python — a far more dangerous tool. The upgrade adds
an ast.Call branch with a 13-function whitelist plus three named
constants (pi, e, tau), preserving the AST-walker safety invariant:
no attribute access, no arbitrary names, no imports, no kwargs.

Each test block covers one invariant:
  1. backward-compat — the old arithmetic surface still returns the
     same numeric answers,
  2. constants + functions — new happy paths work,
  3. whitelist violations — ten distinct jailbreak shapes all reject
     AND emit status=error with error_code='UnsafeExpression' (the
     signal driving the weekly jailbreak review),
  4. runtime errors (math domain, zero division) emit error_code
     'EvalError' — crucial because math.sqrt(-1) also raises
     ValueError, so without the _UnsafeExpressionError subclass the
     weekly UnsafeExpression query would be polluted by user errors,
  5. syntax errors emit error_code='SyntaxError',
  6. success emits status='ok' with a perf.result_type discriminator.

Strategy: monkeypatch builtin.emit with a capturing list so every
calculator() call's observability side-effect is inspectable without
touching app_events.
"""
from __future__ import annotations

import math

import pytest

from local_ai_platform.tools import builtin


@pytest.fixture
def captured_emits(monkeypatch):
    """Replace builtin.emit with a recorder; return the captured list."""
    events: list[dict] = []

    def _fake_emit(subsystem, action, status="ok", duration_ms=None,
                   error_code=None, error_message=None,
                   context=None, perf=None):
        events.append({
            "subsystem": subsystem,
            "action": action,
            "status": status,
            "error_code": error_code,
            "error_message": error_message,
            "context": context,
            "perf": perf,
        })

    monkeypatch.setattr(builtin, "emit_typed", _fake_emit)
    return events


def _result_value(output: str) -> str:
    """Pull the 'X' from calculator's 'expr = X' output string."""
    assert " = " in output, f"expected success output, got: {output!r}"
    return output.split(" = ", 1)[1]


# ── Backward-compat: the old arithmetic surface ──────────────────────


def test_addition_and_precedence(captured_emits):
    assert _result_value(builtin.calculator("2 + 3 * 4")) == "14"


def test_power_operator(captured_emits):
    assert _result_value(builtin.calculator("2 ** 10")) == "1024"


def test_modulo(captured_emits):
    assert _result_value(builtin.calculator("17 % 5")) == "2"


def test_true_division_returns_float(captured_emits):
    assert _result_value(builtin.calculator("10 / 4")) == "2.5"


def test_unary_minus(captured_emits):
    assert _result_value(builtin.calculator("-5 + 2")) == "-3"


def test_float_literal(captured_emits):
    # Makes sure numeric floats still flow through unchanged.
    out = _result_value(builtin.calculator("1.5 + 2.5"))
    assert float(out) == pytest.approx(4.0)


# ── Named constants ──────────────────────────────────────────────────


def test_pi_constant(captured_emits):
    out = _result_value(builtin.calculator("pi"))
    assert float(out) == pytest.approx(math.pi)


def test_e_constant(captured_emits):
    out = _result_value(builtin.calculator("e"))
    assert float(out) == pytest.approx(math.e)


def test_tau_constant(captured_emits):
    out = _result_value(builtin.calculator("tau"))
    assert float(out) == pytest.approx(math.tau)


def test_constant_in_expression(captured_emits):
    out = _result_value(builtin.calculator("2 * pi"))
    assert float(out) == pytest.approx(2 * math.pi)


# ── Whitelisted function calls ───────────────────────────────────────


def test_sqrt(captured_emits):
    out = _result_value(builtin.calculator("sqrt(16)"))
    assert float(out) == pytest.approx(4.0)


def test_sin_at_zero(captured_emits):
    out = _result_value(builtin.calculator("sin(0)"))
    assert float(out) == pytest.approx(0.0)


def test_cos_tan(captured_emits):
    assert float(_result_value(builtin.calculator("cos(0)"))) == pytest.approx(1.0)
    assert float(_result_value(builtin.calculator("tan(0)"))) == pytest.approx(0.0)


def test_log_and_log10(captured_emits):
    assert float(_result_value(builtin.calculator("log(e)"))) == pytest.approx(1.0)
    assert float(_result_value(builtin.calculator("log10(1000)"))) == pytest.approx(3.0)


def test_exp(captured_emits):
    assert float(_result_value(builtin.calculator("exp(1)"))) == pytest.approx(math.e)


def test_abs_pow_floor_ceil_round(captured_emits):
    assert _result_value(builtin.calculator("abs(-7)")) == "7"
    # builtin pow preserves int type when all args are int.
    assert _result_value(builtin.calculator("pow(2, 8)")) == "256"
    assert _result_value(builtin.calculator("floor(2.9)")) == "2"
    assert _result_value(builtin.calculator("ceil(2.1)")) == "3"
    assert _result_value(builtin.calculator("round(2.567, 2)")) == "2.57"


def test_min_max_multi_arg(captured_emits):
    assert _result_value(builtin.calculator("min(3, 1, 2)")) == "1"
    assert _result_value(builtin.calculator("max(3, 1, 2)")) == "3"


def test_nested_function_calls(captured_emits):
    # 3-4-5 triangle: sqrt(3**2 + 4**2) = 5
    out = _result_value(builtin.calculator("sqrt(pow(3, 2) + pow(4, 2))"))
    assert float(out) == pytest.approx(5.0)


def test_constant_inside_function(captured_emits):
    # sin(pi/2) = 1
    out = _result_value(builtin.calculator("sin(pi / 2)"))
    assert float(out) == pytest.approx(1.0)


# ── Whitelist violations → UnsafeExpression ──────────────────────────
# Each of these must (a) return an Error string and (b) emit one
# observability event with error_code='UnsafeExpression'. The weekly
# jailbreak review filters on that code, so mis-labeling any of these
# would blind the dashboard.


@pytest.mark.parametrize("expr", [
    "__import__('os')",                # dunder name + call
    "open('/etc/passwd')",             # non-whitelisted builtin
    "os.system('ls')",                 # attribute access on call
    "math.sqrt(4)",                    # even 'math.' prefix is rejected
    "eval('1+1')",                     # eval/exec are obvious red flags
    "exec('x=1')",
    "(lambda x: x)(1)",                # lambda
    "[1, 2, 3]",                       # list literal
    "'hello'",                         # string literal
    "undefined_variable",              # arbitrary name
    "True",                            # bool literal (would slip via int subclass)
    "sqrt(16, base=2)",                # keyword argument
])
def test_rejects_unsafe_expression(expr, captured_emits):
    out = builtin.calculator(expr)
    assert out.startswith("Error evaluating"), f"expected rejection for {expr!r}, got {out!r}"
    assert len(captured_emits) == 1
    ev = captured_emits[0]
    assert ev["subsystem"] == "tool"
    assert ev["action"] == "calculator_eval"
    assert ev["status"] == "error"
    assert ev["error_code"] == "UnsafeExpression", (
        f"{expr!r} was labeled {ev['error_code']!r} — must be UnsafeExpression "
        "so the weekly jailbreak review captures it"
    )
    assert ev["context"]["expression_length"] == len(expr)


# ── Runtime errors → EvalError (NOT UnsafeExpression) ────────────────
# This is the key separation: math.sqrt(-1) raises ValueError, and
# without the _UnsafeExpressionError subclass every math-domain error
# would be misfiled as a jailbreak attempt.


def test_sqrt_negative_emits_eval_error(captured_emits):
    out = builtin.calculator("sqrt(-1)")
    assert out.startswith("Error evaluating")
    assert len(captured_emits) == 1
    assert captured_emits[0]["error_code"] == "EvalError"


def test_log_zero_emits_eval_error(captured_emits):
    out = builtin.calculator("log(0)")
    assert out.startswith("Error evaluating")
    assert captured_emits[0]["error_code"] == "EvalError"


def test_zero_division_emits_eval_error(captured_emits):
    out = builtin.calculator("1 / 0")
    assert out.startswith("Error evaluating")
    assert captured_emits[0]["error_code"] == "EvalError"


# ── Syntax errors → SyntaxError code ─────────────────────────────────


def test_incomplete_expression_emits_syntax_error(captured_emits):
    out = builtin.calculator("2 +")
    assert out.startswith("Error evaluating")
    assert captured_emits[0]["error_code"] == "SyntaxError"


def test_unbalanced_parens_emit_syntax_error(captured_emits):
    out = builtin.calculator("(((")
    assert out.startswith("Error evaluating")
    assert captured_emits[0]["error_code"] == "SyntaxError"


# ── Success-path observability ───────────────────────────────────────


def test_success_emits_ok_with_result_type(captured_emits):
    builtin.calculator("2 + 3")
    assert len(captured_emits) == 1
    ev = captured_emits[0]
    assert ev["status"] == "ok"
    assert ev["error_code"] is None
    assert ev["context"]["expression_length"] == len("2 + 3")
    # 2 + 3 stays int-typed end-to-end (we dropped the eager float() cast).
    assert ev["perf"]["result_type"] == "int"


def test_success_with_float_result_tags_perf(captured_emits):
    builtin.calculator("sqrt(2)")
    ev = captured_emits[-1]
    assert ev["status"] == "ok"
    assert ev["perf"]["result_type"] == "float"


# ── Tool-registry wiring stays intact ────────────────────────────────


def test_builtin_tools_still_registers_calculator():
    tools = builtin.get_builtin_tools()
    names = {t.name for t in tools}
    assert "calculator" in names
    calc = next(t for t in tools if t.name == "calculator")
    # Description must mention the new capabilities so the LLM can pick
    # calculator instead of run_python for math tasks.
    assert "sqrt" in calc.description
    assert "pi" in calc.description
