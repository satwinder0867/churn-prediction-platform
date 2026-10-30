"""Retention policy: turn churn probabilities into actions (none / email / call).

Shared by evaluate.py (to measure the policy) and score.py (to apply it),
so both always use exactly the same rules.
"""
import numpy as np
import pandas as pd


def customer_value(monthly_charges, biz: dict) -> np.ndarray:
    """Revenue kept if a churner is saved: monthly charge x months they stay."""
    return np.asarray(monthly_charges, dtype=float) * biz["value_months"]


def expected_profits(p, value, biz: dict) -> tuple[np.ndarray, np.ndarray]:
    """Expected profit of an email and of a call: p x save_rate x value - cost."""
    email, call = biz["email"], biz["call"]
    ev_email = p * email["save_rate"] * value - email["cost"]
    ev_call = p * call["save_rate"] * value - call["cost"]
    return ev_email, ev_call


def choose_actions(p, value, biz: dict) -> np.ndarray:
    """Pick the action with the highest expected profit per customer.

    Calls are capped by team capacity: the customers where a call adds the most
    over an email get called; the rest fall back to the best other option.
    """
    p = np.asarray(p, dtype=float)
    ev_email, ev_call = expected_profits(p, value, biz)
    ev_without_call = np.maximum(ev_email, 0)

    actions = np.where(ev_email > 0, "email", "none").astype(object)
    wants_call = ev_call > ev_without_call
    capacity = int(biz["max_call_share"] * len(p))

    gain = np.where(wants_call, ev_call - ev_without_call, -np.inf)
    top = np.argsort(-gain)[:capacity]
    top = top[wants_call[top]]
    actions[top] = "call"
    return actions


def expected_profit_of(actions, p, value, biz: dict) -> np.ndarray:
    """Expected profit of the action each customer was given (0 for 'none')."""
    ev_email, ev_call = expected_profits(np.asarray(p, dtype=float), value, biz)
    actions = pd.Series(actions)
    return np.select([actions == "email", actions == "call"], [ev_email, ev_call], 0.0)
