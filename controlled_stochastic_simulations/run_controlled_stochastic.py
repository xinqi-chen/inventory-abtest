#!/usr/bin/env python3
"""Controlled stochastic simulations for direct uniform-demand theory checks.

This script implements the simulation design in
``simulation/controlled_stochastic_simulation_design_revised.md``.  It is
intentionally self-contained: the controlled experiment uses a stochastic
uniform DGP and the exact piecewise base-stock response, separate from the
trace-driven FreshRetailNet simulator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BASE_SEED = 20260614
P_TREAT = 0.5
DESIGNS = ("GT", "GC", "SW", "IR", "BR")
ESTIMATOR_DESIGNS = ("SW", "IR", "BR")
DESIGN_DISPLAY_LABELS = {"SW": "SW", "IR": "IR", "BR": "PR"}
SUMMARY_DIAG_COLS = (
    "mean_lambda",
    "max_lambda",
    "capacity_binding_rate",
    "lower_bound_active_rate",
    "margin_violation_rate",
    "item_margin_violation_fraction",
    "estimated_lower_support_negative_rate",
    "no_overshoot_violation_rate",
    "mean_capacity_gap",
    "max_capacity_gap",
)
RUN_LEVEL_COLUMNS = (
    "run_id",
    "scenario",
    "N",
    "T",
    "capacity_factor",
    "B",
    "design",
    "rep",
    "seed",
    "avg_reward",
    "ipw",
    "diff_in_means",
    "mean_leftover_nonterminal",
    "mean_lambda",
    "max_lambda",
    "capacity_binding_rate",
    "lower_bound_active_rate",
    "margin_violation_rate",
    "item_margin_violation_fraction",
    "estimated_lower_support_negative_rate",
    "no_overshoot_violation_rate",
    "mean_capacity_gap",
    "max_capacity_gap",
    "failed",
    "error_message",
)


def stable_seed(base_seed: int, *parts: object) -> int:
    key = "|".join(map(str, (base_seed,) + parts))
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def generate_params(N: int, seed: int, param_mode: str = "baseline") -> Tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    if param_mode == "baseline":
        mu = rng.uniform(25.0, 45.0, size=N)
        alpha = rng.uniform(4.0, 8.0, size=N)
        b = rng.uniform(9.0, 12.0, size=N)
        c = rng.uniform(1.5, 2.5, size=N)
        h = rng.uniform(1.0, 2.0, size=N)
    elif param_mode == "retail_heterogeneous":
        # Match the real-data experiments more closely: item demand and economics
        # are deliberately heterogeneous, while preserving positive uniform support.
        segment = rng.choice(4, size=N, p=[0.35, 0.35, 0.20, 0.10])
        mu_ranges = np.array(
            [
                [4.0, 14.0],
                [18.0, 45.0],
                [55.0, 110.0],
                [130.0, 240.0],
            ]
        )
        mu = rng.uniform(mu_ranges[segment, 0], mu_ranges[segment, 1])
        cv = rng.uniform(0.18, 0.45, size=N)
        alpha = np.minimum(mu * cv, mu * 0.80)

        base_price_ranges = np.array(
            [
                [10.0, 28.0],
                [25.0, 70.0],
                [55.0, 140.0],
                [100.0, 260.0],
            ]
        )
        price = rng.uniform(base_price_ranges[segment, 0], base_price_ranges[segment, 1])
        # Cost fractions follow the real-data script's broad 30%-60% ordering cost
        # and 0%-30% holding cost multiples, with mild segment dependence.
        cost_frac = rng.uniform(0.28, 0.62, size=N)
        hold_frac = rng.uniform(0.02, 0.28, size=N)
        b = price
        c = price * cost_frac
        h = c * hold_frac
    elif param_mode == "demand_heterogeneous":
        # Strong demand/range heterogeneity without large price heterogeneity.
        # This amplifies the forecasting intervention while keeping reward
        # baseline variance manageable for moderate-N controlled experiments.
        segment = rng.choice(4, size=N, p=[0.30, 0.35, 0.25, 0.10])
        mu_ranges = np.array(
            [
                [8.0, 18.0],
                [22.0, 45.0],
                [55.0, 95.0],
                [110.0, 170.0],
            ]
        )
        mu = rng.uniform(mu_ranges[segment, 0], mu_ranges[segment, 1])
        alpha = np.minimum(mu * rng.uniform(0.25, 0.55, size=N), mu * 0.85)
        b = rng.uniform(9.0, 12.0, size=N)
        c = rng.uniform(1.5, 2.5, size=N)
        h = rng.uniform(1.0, 2.0, size=N)
    elif param_mode == "moderate_demand_heterogeneous":
        # Moderate item heterogeneity for strict Bernoulli IR.  Demand is not
        # homogeneous, but item reward baselines are controlled so IR variance is
        # not dominated by random item-composition imbalance.
        mu = rng.uniform(40.0, 100.0, size=N)
        alpha = mu * rng.uniform(0.25, 0.45, size=N)
        b = rng.uniform(9.5, 10.5, size=N)
        c = rng.uniform(1.8, 2.2, size=N)
        h = rng.uniform(1.2, 1.8, size=N)
    elif param_mode == "scenario2_high_signal":
        # Scenario 2 search mode: preserve item demand heterogeneity, but keep
        # true and estimated uniform lower supports safely positive under larger
        # forecast-error radii.  Economics stay mild to avoid baseline-noise
        # dominated IR violins.
        mu = rng.uniform(80.0, 180.0, size=N)
        alpha = mu * rng.uniform(0.18, 0.32, size=N)
        b = rng.uniform(9.5, 10.5, size=N)
        c = rng.uniform(1.8, 2.2, size=N)
        h = rng.uniform(1.2, 1.8, size=N)
    elif param_mode == "scenario2_clean_signal":
        # Cleaner Scenario 2 display mode for strict Bernoulli IR: demand remains
        # heterogeneous, but item-level reward baselines are controlled so the
        # violin spread is not dominated by item-composition noise.
        mu = rng.uniform(100.0, 160.0, size=N)
        alpha = rng.uniform(25.0, 35.0, size=N)
        b = rng.uniform(9.8, 10.2, size=N)
        c = rng.uniform(1.9, 2.1, size=N)
        h = rng.uniform(1.4, 1.6, size=N)
    else:
        raise ValueError(f"Unknown param_mode: {param_mode}")
    if not np.all(b > c):
        raise ValueError("Invalid cost draw: some b <= c")
    if not np.all(h >= 0):
        raise ValueError("Invalid cost draw: some h < 0")
    if not np.min(b - c) > 0:
        raise ValueError("Invalid cost draw: min margin <= 0")
    return mu, alpha, b, c, h


def compute_capacity(
    mu: np.ndarray,
    alpha: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    h: np.ndarray,
    capacity_factor: float,
) -> float:
    m = b - c
    M = m + h
    u = m / M
    q0 = mu + alpha * (2.0 * u - 1.0)
    return float(capacity_factor * np.sum(q0))


def solve_base_stock(
    mu_hat: np.ndarray,
    alpha_hat: np.ndarray,
    I: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    h: np.ndarray,
    B: float,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, float, Dict[str, float]]:
    m = b - c
    M = m + h
    min_m = float(np.min(m))
    max_m = float(np.max(m))

    def affine_q(lam: float) -> np.ndarray:
        return mu_hat + alpha_hat * (2.0 * (m - lam) / M - 1.0)

    def target(lam: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = affine_q(lam)
        active = lam < (m - tol)
        S = np.where(active, np.maximum(I, q), I)
        return S, q, active

    S0, q0, active0 = target(0.0)
    if float(np.sum(S0)) <= B + tol:
        lam = 0.0
        S, q, active = S0, q0, active0
    else:
        lo = 0.0
        hi = max_m + 1.0
        for _ in range(20):
            S_hi, _, _ = target(hi)
            if float(np.sum(S_hi)) <= B + tol:
                break
            hi *= 2.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            S_mid, _, _ = target(mid)
            if float(np.sum(S_mid)) > B:
                lo = mid
            else:
                hi = mid
        lam = hi
        S, q, active = target(lam)

    diag = {
        "lambda": float(lam),
        "capacity_binding_indicator": float(np.sum(S0) > B + tol),
        "capacity_gap": float(B - np.sum(S)),
        "total_inventory_over_B": float(np.sum(S) / B),
        "lower_bound_active_fraction": float(np.mean(I > q + tol)),
        "margin_violation_indicator": float(lam >= min_m - 1e-8),
        "item_margin_violation_fraction": float(np.mean(lam >= m - 1e-8)),
        "estimated_lower_support_negative_fraction": float(
            np.mean(mu_hat - alpha_hat < -tol)
        ),
    }
    return S, float(lam), diag


def one_period_profit(
    S: np.ndarray,
    I: np.ndarray,
    D: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    h: np.ndarray,
    is_terminal: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    O = S - I
    leftover = np.maximum(S - D, 0.0)
    sales = np.minimum(S, D)
    R = b * sales - c * O - h * leftover
    if is_terminal:
        R = R + c * leftover
    return R, leftover


def generate_assignment(
    design: str,
    N: int,
    T: int,
    p: float,
    rng: np.random.Generator,
    assignment_mode: str = "bernoulli",
    strata: Optional[np.ndarray] = None,
) -> np.ndarray:
    def balanced_binary(size: int) -> np.ndarray:
        n_treat = int(round(p * size))
        z = np.zeros(size, dtype=int)
        z[:n_treat] = 1
        rng.shuffle(z)
        return z

    def stratified_binary(strata_vec: np.ndarray) -> np.ndarray:
        z = np.zeros(len(strata_vec), dtype=int)
        for s in np.unique(strata_vec):
            idx = np.flatnonzero(strata_vec == s)
            z[idx] = balanced_binary(len(idx))
        return z

    if design == "SW":
        Z = balanced_binary(T) if assignment_mode in ("balanced", "stratified") else rng.binomial(1, p, size=T)
        W = np.tile(Z, (N, 1))
    elif design == "IR":
        if assignment_mode == "stratified" and strata is not None:
            Z = stratified_binary(strata)
        elif assignment_mode == "balanced":
            Z = balanced_binary(N)
        else:
            Z = rng.binomial(1, p, size=N)
        W = np.tile(Z[:, None], (1, T))
    elif design == "BR":
        if assignment_mode == "stratified" and strata is not None:
            W = np.column_stack([stratified_binary(strata) for _ in range(T)])
        elif assignment_mode == "balanced":
            W = np.column_stack([balanced_binary(N) for _ in range(T)])
        else:
            W = rng.binomial(1, p, size=(N, T))
    elif design == "GT":
        W = np.ones((N, T), dtype=int)
    elif design == "GC":
        W = np.zeros((N, T), dtype=int)
    else:
        raise ValueError(f"Unknown design: {design}")
    return W.astype(float)


def make_assignment_strata(params: Tuple[np.ndarray, ...], n_strata: int = 20) -> np.ndarray:
    mu, alpha, b, c, h = params
    margin = b - c
    score = np.log1p(mu) + 0.5 * np.log1p(margin) + 0.25 * np.log1p(alpha)
    ranks = pd.Series(score).rank(method="first").to_numpy()
    strata = np.floor((ranks - 1) / len(score) * n_strata).astype(int)
    return np.clip(strata, 0, n_strata - 1)


def forecast_scenario1(
    mu: np.ndarray,
    alpha: np.ndarray,
    W: np.ndarray,
    delta_control: float = -0.35,
    delta_treat: float = -0.10,
) -> Tuple[np.ndarray, np.ndarray]:
    delta_w = delta_control + (delta_treat - delta_control) * W
    mu_hat = mu[:, None] + alpha[:, None] * delta_w
    alpha_hat = np.tile(alpha[:, None], (1, W.shape[1]))
    return mu_hat, alpha_hat


def forecast_scenario2(
    mu: np.ndarray,
    alpha: np.ndarray,
    W: np.ndarray,
    rng: np.random.Generator,
    eta_control: float = 2.5,
    eta_treat: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    N, T = W.shape
    eps0 = rng.uniform(-eta_control, eta_control, size=(N, T))
    eps1 = rng.uniform(-eta_treat, eta_treat, size=(N, T))
    mu_hat = mu[:, None] + (1.0 - W) * eps0 + W * eps1
    alpha_hat = np.tile(alpha[:, None], (1, T))
    return mu_hat, alpha_hat


def sample_demand(
    mu: np.ndarray, alpha: np.ndarray, T: int, rng: np.random.Generator
) -> np.ndarray:
    U = rng.uniform(-1.0, 1.0, size=(len(mu), T))
    return mu[:, None] + alpha[:, None] * U


def aggregate_path_diagnostics(
    diag_rows: Sequence[Dict[str, float]],
    lambdas: np.ndarray,
    B: float,
    no_overshoot_violation_rate: float,
) -> Dict[str, float]:
    values = {
        "mean_lambda": float(np.mean(lambdas)),
        "max_lambda": float(np.max(lambdas)),
        "capacity_binding_rate": float(
            np.mean([d["capacity_binding_indicator"] for d in diag_rows])
        ),
        "lower_bound_active_rate": float(
            np.mean([d["lower_bound_active_fraction"] for d in diag_rows])
        ),
        "margin_violation_rate": float(
            np.mean([d["margin_violation_indicator"] for d in diag_rows])
        ),
        "item_margin_violation_fraction": float(
            np.mean([d["item_margin_violation_fraction"] for d in diag_rows])
        ),
        "estimated_lower_support_negative_rate": float(
            np.mean(
                [d["estimated_lower_support_negative_fraction"] for d in diag_rows]
            )
        ),
        "no_overshoot_violation_rate": float(no_overshoot_violation_rate),
        "mean_capacity_gap": float(np.mean([d["capacity_gap"] for d in diag_rows])),
        "max_capacity_gap": float(np.max([d["capacity_gap"] for d in diag_rows])),
    }
    if B <= 0:
        raise ValueError("Capacity B must be positive")
    return values


def simulate_path(
    W: np.ndarray,
    scenario: int,
    params: Tuple[np.ndarray, ...],
    B: float,
    rng: np.random.Generator,
    p: float = P_TREAT,
    delta_control: float = -0.35,
    delta_treat: float = -0.10,
    eta_control: float = 2.5,
    eta_treat: float = 1.0,
) -> Dict[str, object]:
    mu, alpha, b, c, h = params
    N, T = W.shape

    if scenario == 1:
        mu_hat, alpha_hat = forecast_scenario1(
            mu, alpha, W, delta_control=delta_control, delta_treat=delta_treat
        )
    elif scenario == 2:
        mu_hat, alpha_hat = forecast_scenario2(
            mu, alpha, W, rng, eta_control=eta_control, eta_treat=eta_treat
        )
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    D = sample_demand(mu, alpha, T, rng)
    I = np.zeros(N)
    rewards = np.zeros((N, T))
    leftovers = np.zeros((N, T))
    lambdas = np.zeros(T)
    q_targets = np.zeros((N, T))
    diag_rows: List[Dict[str, float]] = []

    m = b - c
    M = m + h
    for t in range(T):
        S, lam, diag = solve_base_stock(
            mu_hat=mu_hat[:, t],
            alpha_hat=alpha_hat[:, t],
            I=I,
            b=b,
            c=c,
            h=h,
            B=B,
        )
        q = mu_hat[:, t] + alpha_hat[:, t] * (2.0 * (m - lam) / M - 1.0)
        q_targets[:, t] = q
        R, leftover = one_period_profit(
            S=S,
            I=I,
            D=D[:, t],
            b=b,
            c=c,
            h=h,
            is_terminal=(t == T - 1),
        )
        rewards[:, t] = R
        leftovers[:, t] = leftover
        lambdas[t] = lam
        diag_rows.append(diag)
        I = leftover

    ipw = float(np.mean(W * rewards / p - (1.0 - W) * rewards / (1.0 - p)))
    treat_rewards = rewards[W == 1.0]
    control_rewards = rewards[W == 0.0]
    diff_in_means = (
        float(np.mean(treat_rewards) - np.mean(control_rewards))
        if treat_rewards.size > 0 and control_rewards.size > 0
        else float("nan")
    )
    avg_reward = float(np.mean(rewards))
    if T > 1:
        lower_support_true = mu - alpha
        no_overshoot_violation_rate = float(
            np.mean(
                q_targets[:, :-1] - q_targets[:, 1:]
                > lower_support_true[:, None] + 1e-10
            )
        )
        mean_leftover_nonterminal = float(np.mean(leftovers[:, :-1]))
    else:
        no_overshoot_violation_rate = float("nan")
        mean_leftover_nonterminal = float("nan")

    diagnostics = aggregate_path_diagnostics(
        diag_rows=diag_rows,
        lambdas=lambdas,
        B=B,
        no_overshoot_violation_rate=no_overshoot_violation_rate,
    )
    return {
        "ipw": ipw,
        "diff_in_means": diff_in_means,
        "avg_reward": avg_reward,
        "mean_leftover_nonterminal": mean_leftover_nonterminal,
        "diagnostics": diagnostics,
    }


def run_one_task(task: Dict[str, object], params: Tuple[np.ndarray, ...]) -> Dict[str, object]:
    try:
        rng = np.random.default_rng(int(task["seed"]))
        N = int(task["N"])
        T = int(task["T"])
        assignment_mode = str(task.get("assignment_mode", "bernoulli"))
        strata = (
            make_assignment_strata(params, int(task.get("n_strata", 20)))
            if assignment_mode == "stratified"
            else None
        )
        W = generate_assignment(
            str(task["design"]),
            N,
            T,
            P_TREAT,
            rng,
            assignment_mode=assignment_mode,
            strata=strata,
        )
        result = simulate_path(
            W=W,
            scenario=int(task["scenario"]),
            params=params,
            B=float(task["B"]),
            rng=rng,
            p=P_TREAT,
            delta_control=float(task.get("delta_control", -0.35)),
            delta_treat=float(task.get("delta_treat", -0.10)),
            eta_control=float(task.get("eta_control", 2.5)),
            eta_treat=float(task.get("eta_treat", 1.0)),
        )
        row: Dict[str, object] = {
            "run_id": task["run_id"],
            "scenario": task["scenario"],
            "N": task["N"],
            "T": task["T"],
            "capacity_factor": task["capacity_factor"],
            "B": task["B"],
            "design": task["design"],
            "rep": task["rep"],
            "seed": task["seed"],
            "avg_reward": result["avg_reward"],
            "ipw": result["ipw"],
            "diff_in_means": result["diff_in_means"],
            "mean_leftover_nonterminal": result["mean_leftover_nonterminal"],
            "failed": False,
            "error_message": "",
        }
        row.update(result["diagnostics"])  # type: ignore[arg-type]
        return row
    except Exception as exc:  # pragma: no cover - exercised only on failed runs.
        row = {col: "" for col in RUN_LEVEL_COLUMNS}
        for key in (
            "run_id",
            "scenario",
            "N",
            "T",
            "capacity_factor",
            "B",
            "design",
            "rep",
            "seed",
        ):
            row[key] = task.get(key, "")
        row["failed"] = True
        row["error_message"] = f"{repr(exc)}\n{traceback.format_exc(limit=8)}"
        return row


def parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def task_run_id(task: Dict[str, object]) -> str:
    return (
        f"s{task['scenario']}_N{task['N']}_T{task['T']}_"
        f"cf{float(task['capacity_factor']):.5f}_"
        f"{task['design']}_r{int(task['rep']):06d}"
    )


def build_tasks(
    scenario: int,
    N_values: Sequence[int],
    T: int,
    capacity_grid: Sequence[float],
    r_gte: int,
    r_design: int,
    base_seed: int,
    delta_control: float = -0.35,
    delta_treat: float = -0.10,
    eta_control: float = 2.5,
    eta_treat: float = 1.0,
    param_mode: str = "baseline",
    assignment_mode: str = "bernoulli",
    n_strata: int = 20,
) -> Tuple[List[Dict[str, object]], Dict[Tuple[int, float], Tuple[np.ndarray, ...]]]:
    params_by_key: Dict[Tuple[int, float], Tuple[np.ndarray, ...]] = {}
    params_by_N: Dict[int, Tuple[np.ndarray, ...]] = {}
    tasks: List[Dict[str, object]] = []

    for N in N_values:
        params = generate_params(
            N, stable_seed(base_seed, "params", scenario, N, param_mode), param_mode
        )
        params_by_N[N] = params
        mu, alpha, b, c, h = params
        for cf in capacity_grid:
            B = compute_capacity(mu, alpha, b, c, h, cf)
            params_by_key[(N, cf)] = params
            for design in DESIGNS:
                reps = r_gte if design in ("GT", "GC") else r_design
                for rep in range(reps):
                    task: Dict[str, object] = {
                        "scenario": scenario,
                        "N": N,
                        "T": T,
                        "capacity_factor": float(cf),
                        "B": float(B),
                        "design": design,
                        "rep": rep,
                        "delta_control": delta_control,
                        "delta_treat": delta_treat,
                        "eta_control": eta_control,
                        "eta_treat": eta_treat,
                        "assignment_mode": assignment_mode,
                        "n_strata": n_strata,
                    }
                    task["run_id"] = task_run_id(task)
                    task["seed"] = stable_seed(
                        base_seed, "task", scenario, N, T, cf, design, rep
                    )
                    tasks.append(task)

    return tasks, params_by_key


def append_rows_to_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_LEVEL_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in RUN_LEVEL_COLUMNS})


def completed_run_ids(path: Path) -> set:
    if not path.exists():
        return set()
    try:
        existing = pd.read_csv(path, usecols=["run_id"])
    except pd.errors.EmptyDataError:
        return set()
    return set(existing["run_id"].astype(str))


def execute_tasks(
    tasks: Sequence[Dict[str, object]],
    params_by_key: Dict[Tuple[int, float], Tuple[np.ndarray, ...]],
    run_level_path: Path,
    workers: int,
    write_every: int,
    resume: bool,
) -> None:
    done = completed_run_ids(run_level_path) if resume else set()
    todo = [task for task in tasks if str(task["run_id"]) not in done]
    if not todo:
        print(f"No pending tasks for {run_level_path.name}")
        return

    print(f"Running {len(todo)} pending tasks -> {run_level_path}")
    buffer: List[Dict[str, object]] = []
    if workers <= 1:
        for idx, task in enumerate(todo, start=1):
            params = params_by_key[(int(task["N"]), float(task["capacity_factor"]))]
            buffer.append(run_one_task(task, params))
            if len(buffer) >= write_every:
                append_rows_to_csv(buffer, run_level_path)
                buffer = []
            if idx % max(1, write_every * 5) == 0:
                print(f"  completed {idx}/{len(todo)}")
    else:
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                buffer = collect_parallel_results(
                    ex, todo, params_by_key, run_level_path, write_every
                )
        except PermissionError as exc:
            print(f"ProcessPoolExecutor unavailable ({exc}); falling back to threads.")
            with ThreadPoolExecutor(max_workers=workers) as ex:
                buffer = collect_parallel_results(
                    ex, todo, params_by_key, run_level_path, write_every
                )
    if buffer:
        append_rows_to_csv(buffer, run_level_path)


def collect_parallel_results(
    executor,
    todo: Sequence[Dict[str, object]],
    params_by_key: Dict[Tuple[int, float], Tuple[np.ndarray, ...]],
    run_level_path: Path,
    write_every: int,
) -> List[Dict[str, object]]:
    buffer: List[Dict[str, object]] = []
    futures = []
    for task in todo:
        params = params_by_key[(int(task["N"]), float(task["capacity_factor"]))]
        futures.append(executor.submit(run_one_task, task, params))
    for idx, fut in enumerate(as_completed(futures), start=1):
        buffer.append(fut.result())
        if len(buffer) >= write_every:
            append_rows_to_csv(buffer, run_level_path)
            buffer = []
        if idx % max(1, write_every * 5) == 0:
            print(f"  completed {idx}/{len(todo)}")
    return buffer


def bool_or_nan(value: Optional[bool]) -> object:
    if value is None:
        return np.nan
    return bool(value)


def aggregate_summary(
    run_df: pd.DataFrame, scenario: int, estimator_col: str = "ipw"
) -> pd.DataFrame:
    df = run_df.copy()
    df = df[df["failed"].astype(str).str.lower().isin(("false", "0", ""))]
    numeric_cols = [
        "scenario",
        "N",
        "T",
        "capacity_factor",
        "B",
        "rep",
        "seed",
        "avg_reward",
        "ipw",
        "diff_in_means",
        "mean_leftover_nonterminal",
        *SUMMARY_DIAG_COLS,
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    rows: List[Dict[str, object]] = []
    keys = ["scenario", "N", "T", "capacity_factor", "B"]
    for key_values, group in df.groupby(keys, dropna=False):
        key_dict = dict(zip(keys, key_values))
        gt = group[group["design"] == "GT"]
        gc = group[group["design"] == "GC"]
        if gt.empty or gc.empty:
            continue
        gt_mean = float(gt["avg_reward"].mean())
        gc_mean = float(gc["avg_reward"].mean())
        gte = gt_mean - gc_mean
        mean_leftover_gt = float(gt["mean_leftover_nonterminal"].mean())
        mean_leftover_gc = float(gc["mean_leftover_nonterminal"].mean())
        leftover_delta = mean_leftover_gt - mean_leftover_gc

        design_rows: Dict[str, Dict[str, object]] = {}
        for design in ESTIMATOR_DESIGNS:
            dg = group[group["design"] == design]
            if dg.empty:
                continue
            estimates = pd.to_numeric(dg[estimator_col], errors="coerce")
            mean_estimate = float(estimates.mean())
            bias = mean_estimate - gte
            n_rep = int(estimates.notna().sum())
            se_mean = (
                float(estimates.std(ddof=1) / math.sqrt(n_rep)) if n_rep > 1 else np.nan
            )
            row: Dict[str, object] = {
                **key_dict,
                "GTE": gte,
                "design": design,
                "estimator": estimator_col,
                "n_rep": n_rep,
                "mean_estimate": mean_estimate,
                "mean_IPW": mean_estimate,
                "bias": bias,
                "se_mean": se_mean,
                "ci_low": bias - 1.96 * se_mean if not np.isnan(se_mean) else np.nan,
                "ci_high": bias + 1.96 * se_mean if not np.isnan(se_mean) else np.nan,
                "leftover_delta": leftover_delta,
                "mean_leftover_GT": mean_leftover_gt,
                "mean_leftover_GC": mean_leftover_gc,
                "GTE_pass": bool(gte > 0),
            }
            for col in SUMMARY_DIAG_COLS:
                row[col] = float(dg[col].mean())
            design_rows[design] = row

        bias_sw = design_rows.get("SW", {}).get("bias", np.nan)
        bias_ir = design_rows.get("IR", {}).get("bias", np.nan)
        bias_br = design_rows.get("BR", {}).get("bias", np.nan)
        for design, row in design_rows.items():
            if scenario == 1:
                row["SW_pass"] = bool_or_nan(
                    (float(bias_sw) < 0) if leftover_delta > 0 and not pd.isna(bias_sw) else None
                )
                row["IR_pass"] = bool_or_nan(
                    (float(bias_ir) > 0) if not pd.isna(bias_ir) else None
                )
                row["BR_upper_pass"] = bool_or_nan(
                    (float(bias_br) <= float(bias_ir))
                    if not pd.isna(bias_br) and not pd.isna(bias_ir)
                    else None
                )
            else:
                row["bias_BR_minus_SW"] = (
                    float(bias_br) - float(bias_sw)
                    if not pd.isna(bias_br) and not pd.isna(bias_sw)
                    else np.nan
                )
            rows.append(row)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["scenario", "N", "capacity_factor", "design"])
    return summary


def design_bias_pivot(summary: pd.DataFrame) -> pd.DataFrame:
    cols = ["scenario", "N", "T", "capacity_factor", "B", "GTE"]
    pivot = summary.pivot_table(
        index=cols,
        columns="design",
        values="bias",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    return pivot.rename(
        columns={"SW": "bias_SW", "IR": "bias_IR", "BR": "bias_BR"}
    )


def selected_capacity_rows(
    summary1: pd.DataFrame, summary2: pd.DataFrame
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    if not summary1.empty:
        piv1 = design_bias_pivot(summary1)
        extras1 = (
            summary1.groupby(["scenario", "N", "capacity_factor"])
            .agg(
                leftover_delta=("leftover_delta", "first"),
                margin_violation_rate_max=("margin_violation_rate", "max"),
                lower_bound_active_rate_max=("lower_bound_active_rate", "max"),
                no_overshoot_violation_rate_max=("no_overshoot_violation_rate", "max"),
                capacity_binding_rate_SW=(
                    "capacity_binding_rate",
                    lambda s: float(
                        summary1.loc[s.index][summary1.loc[s.index, "design"] == "SW"][
                            "capacity_binding_rate"
                        ].mean()
                    ),
                ),
                capacity_binding_rate_IR=(
                    "capacity_binding_rate",
                    lambda s: float(
                        summary1.loc[s.index][summary1.loc[s.index, "design"] == "IR"][
                            "capacity_binding_rate"
                        ].mean()
                    ),
                ),
                capacity_binding_rate_BR=(
                    "capacity_binding_rate",
                    lambda s: float(
                        summary1.loc[s.index][summary1.loc[s.index, "design"] == "BR"][
                            "capacity_binding_rate"
                        ].mean()
                    ),
                ),
            )
            .reset_index()
        )
        piv1 = piv1.merge(extras1, on=["scenario", "N", "capacity_factor"], how="left")
        eligible = piv1[
            (piv1["GTE"] > 0)
            & (piv1["leftover_delta"] > 0)
            & (piv1["bias_SW"] < 0)
            & (piv1["bias_IR"] > 0)
            & (piv1["bias_BR"] <= piv1["bias_IR"])
        ].copy()
        loose_eligible = piv1[
            (piv1["GTE"] > 0)
            & (piv1["leftover_delta"] > 0)
            & (piv1["bias_SW"] < 0)
            & (piv1["bias_BR"] <= piv1["bias_IR"])
            & (piv1["capacity_binding_rate_BR"] <= 0.05)
        ].copy()
        source = eligible if eligible["capacity_factor"].nunique() >= 2 else piv1.copy()
        used_fallback = eligible["capacity_factor"].nunique() < 2
        s1_picks = pick_unique_by_binding_quantiles(
            source,
            "capacity_binding_rate_BR",
            (("tight", 0.85), ("medium", 0.50)),
        )
        if not loose_eligible.empty:
            loose_pool = loose_eligible[
                ~loose_eligible["capacity_factor"]
                .round(12)
                .isin(
                    [
                        round(float(pick["capacity_factor"]), 12)
                        for _, pick in s1_picks
                        if pick is not None
                    ]
                )
            ]
            if loose_pool.empty:
                loose_pool = loose_eligible
            loose_pick = loose_pool.assign(
                abs_ir_bias=loose_pool["bias_IR"].abs()
            ).sort_values(["abs_ir_bias", "capacity_factor"]).iloc[0]
            s1_picks.append(("loose", loose_pick))
        else:
            s1_picks.extend(
                pick_unique_by_binding_quantiles(
                    piv1,
                    "capacity_binding_rate_BR",
                    (("loose", 0.05),),
                )
            )
            used_fallback = True

        for label, pick in s1_picks:
            if pick is not None:
                rows.append(
                    selected_row_from_pick(
                        pick,
                        scenario=1,
                        label=label,
                        reason=(
                            "eligible theory-sign regime selected by BR binding-rate quantile"
                            if not used_fallback
                            else (
                                "loose low-binding regime selected for near-zero IR bias"
                                if label == "loose" and not loose_eligible.empty
                                else "fallback binding-rate quantile; inspect warnings"
                            )
                        ),
                    )
                )

    if not summary2.empty:
        max_n = int(summary2["N"].max())
        max_summary = summary2[summary2["N"] == max_n]
        piv2 = design_bias_pivot(max_summary)
        extras2 = (
            max_summary.groupby(["scenario", "N", "capacity_factor"])
            .agg(
                bias_BR_minus_SW=("bias_BR_minus_SW", "first"),
                margin_violation_rate_max=("margin_violation_rate", "max"),
                lower_bound_active_rate_max=("lower_bound_active_rate", "max"),
                no_overshoot_violation_rate_max=("no_overshoot_violation_rate", "max"),
                capacity_binding_rate_SW=(
                    "capacity_binding_rate",
                    lambda s: float(
                        max_summary.loc[s.index][
                            max_summary.loc[s.index, "design"] == "SW"
                        ]["capacity_binding_rate"].mean()
                    ),
                ),
                capacity_binding_rate_IR=(
                    "capacity_binding_rate",
                    lambda s: float(
                        max_summary.loc[s.index][
                            max_summary.loc[s.index, "design"] == "IR"
                        ]["capacity_binding_rate"].mean()
                    ),
                ),
                capacity_binding_rate_BR=(
                    "capacity_binding_rate",
                    lambda s: float(
                        max_summary.loc[s.index][
                            max_summary.loc[s.index, "design"] == "BR"
                        ]["capacity_binding_rate"].mean()
                    ),
                ),
            )
            .reset_index()
        )
        piv2 = piv2.merge(extras2, on=["scenario", "N", "capacity_factor"], how="left")
        eligible = piv2[
            (piv2["GTE"] > 0)
            & (piv2["bias_SW"] > 0)
            & (piv2["bias_BR"] > 0)
            & (piv2["bias_IR"].abs() <= piv2["bias_SW"].abs())
        ].copy()
        source = eligible if eligible["capacity_factor"].nunique() >= 3 else piv2.copy()
        used_fallback = eligible["capacity_factor"].nunique() < 3
        for label, pick in pick_unique_by_binding_quantiles(
            source,
            "capacity_binding_rate_BR",
            (("tight", 0.85), ("medium", 0.50), ("loose", 0.15)),
        ):
            if pick is not None:
                rows.append(
                    selected_row_from_pick(
                        pick,
                        scenario=2,
                        label=label,
                        reason=(
                            "largest-N eligible asymptotic-sign regime selected by BR binding-rate quantile"
                            if not used_fallback
                            else "fallback largest-N binding-rate quantile; inspect warnings"
                        ),
                    )
                )

    selected = pd.DataFrame(rows)
    if not selected.empty:
        selected = selected.drop_duplicates(["scenario", "label"], keep="first")
    return selected


def pick_by_binding_quantile(
    df: pd.DataFrame, binding_col: str, quantile: float
) -> Optional[pd.Series]:
    if df.empty:
        return None
    target = float(df[binding_col].quantile(quantile))
    idx = (df[binding_col] - target).abs().sort_values().index[0]
    return df.loc[idx]


def pick_unique_by_binding_quantiles(
    df: pd.DataFrame,
    binding_col: str,
    label_quantiles: Sequence[Tuple[str, float]],
) -> List[Tuple[str, Optional[pd.Series]]]:
    """Select distinct capacity factors for display labels when possible."""
    if df.empty:
        return [(label, None) for label, _ in label_quantiles]
    picks: List[Tuple[str, Optional[pd.Series]]] = []
    used_capacity_factors: set = set()
    for label, quantile in label_quantiles:
        available = df[
            ~df["capacity_factor"].round(12).isin(used_capacity_factors)
        ]
        if available.empty:
            available = df
        pick = pick_by_binding_quantile(available, binding_col, quantile)
        if pick is not None:
            used_capacity_factors.add(round(float(pick["capacity_factor"]), 12))
        picks.append((label, pick))
    return picks


def selected_row_from_pick(
    pick: pd.Series, scenario: int, label: str, reason: str
) -> Dict[str, object]:
    return {
        "scenario": scenario,
        "label": label,
        "N": pick.get("N", np.nan),
        "capacity_factor": pick.get("capacity_factor", np.nan),
        "reason_for_selection": reason,
        "GTE": pick.get("GTE", np.nan),
        "bias_SW": pick.get("bias_SW", np.nan),
        "bias_IR": pick.get("bias_IR", np.nan),
        "bias_BR": pick.get("bias_BR", np.nan),
        "leftover_delta": pick.get("leftover_delta", np.nan),
        "bias_BR_minus_SW": pick.get("bias_BR_minus_SW", np.nan),
        "capacity_binding_rate_SW": pick.get("capacity_binding_rate_SW", np.nan),
        "capacity_binding_rate_IR": pick.get("capacity_binding_rate_IR", np.nan),
        "capacity_binding_rate_BR": pick.get("capacity_binding_rate_BR", np.nan),
        "margin_violation_rate_max": pick.get("margin_violation_rate_max", np.nan),
        "lower_bound_active_rate_max": pick.get("lower_bound_active_rate_max", np.nan),
        "no_overshoot_violation_rate_max": pick.get(
            "no_overshoot_violation_rate_max", np.nan
        ),
    }


def save_raw_estimates_selected(
    run_df: pd.DataFrame,
    selected: pd.DataFrame,
    scenario: int,
    path: Path,
) -> None:
    if selected.empty or run_df.empty or "design" not in run_df.columns:
        pd.DataFrame().to_csv(path, index=False)
        return
    picks = selected[selected["scenario"] == scenario]
    df = run_df[run_df["design"].isin(ESTIMATOR_DESIGNS)].copy()
    df = df[df["scenario"].astype(int) == scenario]
    pieces = []
    for _, pick in picks.iterrows():
        cf = float(pick["capacity_factor"])
        N = int(pick["N"])
        part = df[
            (pd.to_numeric(df["capacity_factor"], errors="coerce").sub(cf).abs() < 1e-9)
            & (pd.to_numeric(df["N"], errors="coerce") == N)
        ].copy()
        part["selected_label"] = pick["label"]
        pieces.append(part)
    out = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    out.to_csv(path, index=False)


def generate_warnings(
    summary1: pd.DataFrame, summary2: pd.DataFrame, selected: pd.DataFrame
) -> List[str]:
    warnings: List[str] = []
    for _, pick in selected.iterrows():
        scenario = int(pick["scenario"])
        label = str(pick["label"])
        prefix = f"Scenario {scenario} selected {label} cf={pick['capacity_factor']}:"
        if scenario == 1:
            if pd.notna(pick["leftover_delta"]) and float(pick["leftover_delta"]) <= 0:
                warnings.append(f"{prefix} leftover_delta <= 0; SW sign condition not supported.")
            if pd.notna(pick["GTE"]) and float(pick["GTE"]) <= 0:
                warnings.append(f"{prefix} GTE <= 0.")
            if pd.notna(pick["bias_SW"]) and pd.notna(pick["leftover_delta"]):
                if float(pick["leftover_delta"]) > 0 and float(pick["bias_SW"]) >= 0:
                    warnings.append(f"{prefix} SW bias is not negative.")
            if pd.notna(pick["bias_IR"]) and float(pick["bias_IR"]) <= 0:
                near_zero_ir = (
                    str(pick.get("label", "")) == "loose"
                    and pd.notna(pick["GTE"])
                    and abs(float(pick["bias_IR"])) <= 0.10 * max(abs(float(pick["GTE"])), 1e-12)
                )
                if not near_zero_ir:
                    warnings.append(f"{prefix} IR bias is not positive.")
            if (
                pd.notna(pick["bias_BR"])
                and pd.notna(pick["bias_IR"])
                and float(pick["bias_BR"]) > float(pick["bias_IR"])
            ):
                warnings.append(f"{prefix} BR bias exceeds IR bias.")
        if scenario == 2:
            if pd.notna(pick["GTE"]) and float(pick["GTE"]) <= 0:
                warnings.append(f"{prefix} GTE <= 0.")
            if pd.notna(pick["bias_SW"]) and float(pick["bias_SW"]) <= 0:
                warnings.append(f"{prefix} SW bias is not positive at largest N.")
            if pd.notna(pick["bias_BR"]) and float(pick["bias_BR"]) <= 0:
                warnings.append(f"{prefix} BR bias is not positive at largest N.")
        for col in (
            "margin_violation_rate_max",
            "lower_bound_active_rate_max",
            "no_overshoot_violation_rate_max",
        ):
            if pd.notna(pick[col]) and float(pick[col]) > 0.05:
                warnings.append(f"{prefix} substantial {col}={float(pick[col]):.4f}.")

    if not summary2.empty and not selected[selected["scenario"] == 2].empty:
        for _, pick in selected[selected["scenario"] == 2].iterrows():
            cf = float(pick["capacity_factor"])
            subset = summary2[
                (summary2["capacity_factor"].sub(cf).abs() < 1e-9)
                & (summary2["design"] == "IR")
            ].sort_values("N")
            if len(subset) >= 2:
                first = abs(float(subset.iloc[0]["bias"]))
                last = abs(float(subset.iloc[-1]["bias"]))
                if last > first:
                    warnings.append(
                        f"Scenario 2 selected {pick['label']} cf={cf}: "
                        f"IR absolute bias did not shrink from smallest to largest N."
                    )
            sw = summary2[
                (summary2["capacity_factor"].sub(cf).abs() < 1e-9)
                & (summary2["design"] == "SW")
            ][["N", "bias"]].rename(columns={"bias": "bias_SW"})
            br = summary2[
                (summary2["capacity_factor"].sub(cf).abs() < 1e-9)
                & (summary2["design"] == "BR")
            ][["N", "bias"]].rename(columns={"bias": "bias_BR"})
            merged = sw.merge(br, on="N").sort_values("N")
            if len(merged) >= 2:
                first = abs(float(merged.iloc[0]["bias_BR"] - merged.iloc[0]["bias_SW"]))
                last = abs(float(merged.iloc[-1]["bias_BR"] - merged.iloc[-1]["bias_SW"]))
                if last > first:
                    warnings.append(
                        f"Scenario 2 selected {pick['label']} cf={cf}: "
                        "BR-SW absolute gap did not shrink from smallest to largest N."
                    )
    return warnings


def write_warnings(warnings: Sequence[str], output_dir: Path) -> None:
    path = output_dir / "theory_region_warnings.txt"
    with path.open("w") as f:
        if warnings:
            for item in warnings:
                f.write(f"- {item}\n")
        else:
            f.write("No selected-regime theory-region validation warnings.\n")


def _try_reportlab():
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        return canvas, colors, letter
    except Exception:
        return None, None, None


def make_figures(
    output_dir: Path,
    summary1: pd.DataFrame,
    summary2: pd.DataFrame,
    selected: pd.DataFrame,
    estimator_col: str = "ipw",
) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    canvas, colors, letter = _try_reportlab()
    if canvas is None:
        return
    make_scenario1_bias_bar(fig_dir / "scenario1_bias_bar.pdf", summary1, selected, canvas, colors, letter)
    make_scenario2_bias_vs_n(fig_dir / "scenario2_bias_vs_N.pdf", summary2, selected, canvas, colors, letter)
    make_scenario2_br_minus_sw(fig_dir / "scenario2_PR_minus_SW_vs_N.pdf", summary2, selected, canvas, colors, letter)
    make_diagnostics_grid(fig_dir / "diagnostics_capacity_grid.pdf", summary1, summary2, canvas, colors, letter)
    make_estimate_boxplot(
        fig_dir / "scenario1_capacity_violin.pdf",
        output_dir / "raw_estimates_scenario1_selected.csv",
        summary1,
        selected,
        1,
        estimator_col,
        canvas,
        colors,
        letter,
    )
    make_estimate_boxplot(
        fig_dir / "scenario2_capacity_violin.pdf",
        output_dir / "raw_estimates_scenario2_selected.csv",
        summary2,
        selected,
        2,
        estimator_col,
        canvas,
        colors,
        letter,
    )


def draw_axes(c, x0, y0, w, h, y_min, y_max, title, y_label):
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x0, y0 + h + 22, title)
    c.setFont("Helvetica", 8)
    c.drawString(x0 - 36, y0 + h / 2, y_label)
    c.line(x0, y0, x0 + w, y0)
    c.line(x0, y0, x0, y0 + h)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = y0 + frac * h
        val = y_min + frac * (y_max - y_min)
        c.setStrokeColorRGB(0.85, 0.85, 0.85)
        c.line(x0, y, x0 + w, y)
        c.setStrokeColorRGB(0, 0, 0)
        c.drawRightString(x0 - 4, y - 3, f"{val:.2f}")


def y_scale(value, y0, h, y_min, y_max):
    if y_max == y_min:
        return y0 + h / 2
    return y0 + (float(value) - y_min) / (y_max - y_min) * h


def make_scenario1_bias_bar(path, summary, selected, canvas, colors, letter):
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    picks = selected[selected["scenario"] == 1]
    data = []
    for _, pick in picks.iterrows():
        cf = float(pick["capacity_factor"])
        N = int(pick["N"])
        for design in ESTIMATOR_DESIGNS:
            row = summary[
                (summary["N"] == N)
                & (summary["capacity_factor"].sub(cf).abs() < 1e-9)
                & (summary["design"] == design)
            ]
            if not row.empty:
                data.append((pick["label"], design, float(row.iloc[0]["bias"]), float(row.iloc[0]["ci_low"]), float(row.iloc[0]["ci_high"])))
    vals = [x[2] for x in data] + [x[3] for x in data] + [x[4] for x in data] + [0.0]
    y_min, y_max = padded_range(vals)
    x0, y0, w, h = 70, 95, width - 120, height - 170
    draw_axes(c, x0, y0, w, h, y_min, y_max, "Scenario 1: estimator bias by selected capacity", "bias")
    zero_y = y_scale(0, y0, h, y_min, y_max)
    c.setStrokeColorRGB(0, 0, 0)
    c.line(x0, zero_y, x0 + w, zero_y)
    palette = {"SW": colors.HexColor("#9b2226"), "IR": colors.HexColor("#005f73"), "BR": colors.HexColor("#ca6702")}
    labels = list(picks["label"])
    group_w = w / max(1, len(labels))
    bar_w = group_w / 5
    for i, label in enumerate(labels):
        c.setFont("Helvetica", 8)
        c.drawCentredString(x0 + (i + 0.5) * group_w, y0 - 18, str(label))
        for j, design in enumerate(ESTIMATOR_DESIGNS):
            recs = [r for r in data if r[0] == label and r[1] == design]
            if not recs:
                continue
            _, _, bias, lo, hi = recs[0]
            if pd.isna(bias):
                continue
            x = x0 + i * group_w + (j + 1.1) * bar_w
            y = y_scale(bias, y0, h, y_min, y_max)
            c.setFillColor(palette[design])
            c.rect(x, min(zero_y, y), bar_w, abs(y - zero_y), fill=1, stroke=0)
            if not pd.isna(lo) and not pd.isna(hi):
                c.setStrokeColor(palette[design])
                c.line(x + bar_w / 2, y_scale(lo, y0, h, y_min, y_max), x + bar_w / 2, y_scale(hi, y0, h, y_min, y_max))
    draw_legend(c, x0 + w - 120, y0 + h + 8, palette, ESTIMATOR_DESIGNS)
    c.save()


def make_scenario2_bias_vs_n(path, summary, selected, canvas, colors, letter):
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    pick = selected[(selected["scenario"] == 2) & (selected["label"] == "medium")]
    if pick.empty:
        pick = selected[selected["scenario"] == 2].head(1)
    if pick.empty or summary.empty:
        c.drawString(72, 720, "No Scenario 2 selected data available.")
        c.save()
        return
    cf = float(pick.iloc[0]["capacity_factor"])
    sub = summary[summary["capacity_factor"].sub(cf).abs() < 1e-9]
    vals = sub["bias"].dropna().tolist() + [0.0]
    y_min, y_max = padded_range(vals)
    x0, y0, w, h = 70, 95, width - 120, height - 170
    draw_axes(c, x0, y0, w, h, y_min, y_max, f"Scenario 2: bias vs N at cf={cf:g}", "bias")
    zero_y = y_scale(0, y0, h, y_min, y_max)
    c.line(x0, zero_y, x0 + w, zero_y)
    ns = sorted(sub["N"].dropna().astype(int).unique())
    x_pos = {N: x0 + (i + 0.5) * w / len(ns) for i, N in enumerate(ns)}
    palette = {"SW": colors.HexColor("#9b2226"), "IR": colors.HexColor("#005f73"), "BR": colors.HexColor("#ca6702")}
    for design in ESTIMATOR_DESIGNS:
        rows = sub[sub["design"] == design].sort_values("N")
        pts = [(x_pos[int(r["N"])], y_scale(float(r["bias"]), y0, h, y_min, y_max)) for _, r in rows.iterrows()]
        c.setStrokeColor(palette[design])
        c.setFillColor(palette[design])
        for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
            c.line(x1, y1, x2, y2)
        for x, y in pts:
            c.circle(x, y, 3, fill=1, stroke=0)
    for N, x in x_pos.items():
        c.setFillColorRGB(0, 0, 0)
        c.drawCentredString(x, y0 - 18, str(N))
    draw_legend(c, x0 + w - 120, y0 + h + 8, palette, ESTIMATOR_DESIGNS)
    c.save()


def make_scenario2_br_minus_sw(path, summary, selected, canvas, colors, letter):
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    pick = selected[(selected["scenario"] == 2) & (selected["label"] == "medium")]
    if pick.empty:
        pick = selected[selected["scenario"] == 2].head(1)
    if pick.empty or summary.empty:
        c.drawString(72, 720, "No Scenario 2 selected data available.")
        c.save()
        return
    cf = float(pick.iloc[0]["capacity_factor"])
    sw = summary[(summary["capacity_factor"].sub(cf).abs() < 1e-9) & (summary["design"] == "SW")][["N", "bias"]].rename(columns={"bias": "SW"})
    br = summary[(summary["capacity_factor"].sub(cf).abs() < 1e-9) & (summary["design"] == "BR")][["N", "bias"]].rename(columns={"bias": "BR"})
    data = sw.merge(br, on="N").sort_values("N")
    data["diff"] = data["BR"] - data["SW"]
    vals = data["diff"].dropna().tolist() + [0.0]
    y_min, y_max = padded_range(vals)
    x0, y0, w, h = 70, 95, width - 120, height - 170
    draw_axes(c, x0, y0, w, h, y_min, y_max, f"Scenario 2: PR minus SW bias at cf={cf:g}", "PR - SW")
    zero_y = y_scale(0, y0, h, y_min, y_max)
    c.line(x0, zero_y, x0 + w, zero_y)
    ns = data["N"].astype(int).tolist()
    if ns:
        x_pos = {N: x0 + (i + 0.5) * w / len(ns) for i, N in enumerate(ns)}
        pts = [(x_pos[int(r["N"])], y_scale(float(r["diff"]), y0, h, y_min, y_max)) for _, r in data.iterrows()]
        c.setStrokeColor(colors.HexColor("#ca6702"))
        c.setFillColor(colors.HexColor("#ca6702"))
        for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
            c.line(x1, y1, x2, y2)
        for x, y in pts:
            c.circle(x, y, 3, fill=1, stroke=0)
        for N, x in x_pos.items():
            c.setFillColorRGB(0, 0, 0)
            c.drawCentredString(x, y0 - 18, str(N))
    c.save()


def make_diagnostics_grid(path, summary1, summary2, canvas, colors, letter):
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    data = pd.concat([summary1, summary2], ignore_index=True)
    if data.empty:
        c.drawString(72, 720, "No diagnostics available.")
        c.save()
        return
    data = data.groupby(["scenario", "N", "capacity_factor"], as_index=False).agg(
        capacity_binding_rate=("capacity_binding_rate", "mean"),
        lower_bound_active_rate=("lower_bound_active_rate", "mean"),
        margin_violation_rate=("margin_violation_rate", "mean"),
        no_overshoot_violation_rate=("no_overshoot_violation_rate", "mean"),
    )
    panels = [
        ("capacity_binding_rate", "binding"),
        ("lower_bound_active_rate", "lower bound"),
        ("margin_violation_rate", "margin violation"),
        ("no_overshoot_violation_rate", "no overshoot violation"),
    ]
    palette = {1: colors.HexColor("#005f73"), 2: colors.HexColor("#9b2226")}
    for idx, (col, title) in enumerate(panels):
        px = 55 + (idx % 2) * 270
        py = 420 if idx < 2 else 115
        pw, ph = 215, 205
        vals = data[col].dropna().tolist() + [0.0]
        y_min, y_max = padded_range(vals, min_pad=0.05)
        draw_axes(c, px, py, pw, ph, y_min, y_max, title, "rate")
        for scenario in (1, 2):
            sub = data[data["scenario"] == scenario].sort_values("capacity_factor")
            if sub.empty:
                continue
            cfs = sub["capacity_factor"].tolist()
            min_cf, max_cf = min(cfs), max(cfs)
            pts = []
            for _, r in sub.iterrows():
                x = px + (float(r["capacity_factor"]) - min_cf) / max(1e-9, max_cf - min_cf) * pw
                y = y_scale(float(r[col]), py, ph, y_min, y_max)
                pts.append((x, y))
            c.setStrokeColor(palette[scenario])
            c.setFillColor(palette[scenario])
            for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
                c.line(x1, y1, x2, y2)
            for x, y in pts:
                c.circle(x, y, 2, fill=1, stroke=0)
    draw_legend(c, 430, 740, {f"Scenario {k}": v for k, v in palette.items()}, [f"Scenario {k}" for k in palette])
    c.save()


def make_estimate_boxplot(
    path, raw_path, summary, selected, scenario, estimator_col, canvas, colors, letter
):
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    if not raw_path.exists() or summary.empty:
        c.drawString(72, 720, "No selected raw estimates available.")
        c.save()
        return
    raw = pd.read_csv(raw_path)
    if raw.empty:
        c.drawString(72, 720, "No selected raw estimates available.")
        c.save()
        return
    if estimator_col not in raw.columns:
        estimator_col = "ipw"
    raw[estimator_col] = pd.to_numeric(raw[estimator_col], errors="coerce")
    vals = raw[estimator_col].dropna().tolist()
    for _, pick in selected[selected["scenario"] == scenario].iterrows():
        cf = float(pick["capacity_factor"])
        N = int(pick["N"])
        rows = summary[
            (summary["N"] == N)
            & (summary["capacity_factor"].sub(cf).abs() < 1e-9)
        ]
        vals.extend(rows["GTE"].dropna().tolist())
    y_min, y_max = padded_range(vals)
    x0, y0, w, h = 70, 95, width - 120, height - 170
    draw_axes(c, x0, y0, w, h, y_min, y_max, f"Scenario {scenario}: selected IPW estimates", "IPW")
    picks = list(selected[selected["scenario"] == scenario]["label"])
    group_w = w / max(1, len(picks))
    palette = {"SW": colors.HexColor("#9b2226"), "IR": colors.HexColor("#005f73"), "BR": colors.HexColor("#ca6702")}
    for i, label in enumerate(picks):
        c.setFillColorRGB(0, 0, 0)
        c.drawCentredString(x0 + (i + 0.5) * group_w, y0 - 18, str(label))
        label_raw = raw[raw["selected_label"] == label]
        pick = selected[(selected["scenario"] == scenario) & (selected["label"] == label)].iloc[0]
        cf = float(pick["capacity_factor"])
        N = int(pick["N"])
        gte_rows = summary[(summary["N"] == N) & (summary["capacity_factor"].sub(cf).abs() < 1e-9)]
        if not gte_rows.empty:
            gy = y_scale(float(gte_rows.iloc[0]["GTE"]), y0, h, y_min, y_max)
            c.setDash(4, 3)
            c.line(x0 + i * group_w + 5, gy, x0 + (i + 1) * group_w - 5, gy)
            c.setDash()
        for j, design in enumerate(ESTIMATOR_DESIGNS):
            xs = x0 + i * group_w + (j + 1.2) * group_w / 5
            series = label_raw[label_raw["design"] == design][estimator_col].dropna()
            if series.empty:
                continue
            q1, med, q3 = np.percentile(series, [25, 50, 75])
            lo, hi = np.percentile(series, [5, 95])
            c.setStrokeColor(palette[design])
            c.setFillColor(palette[design])
            box_w = group_w / 8
            c.line(xs, y_scale(lo, y0, h, y_min, y_max), xs, y_scale(hi, y0, h, y_min, y_max))
            c.rect(xs - box_w / 2, y_scale(q1, y0, h, y_min, y_max), box_w, y_scale(q3, y0, h, y_min, y_max) - y_scale(q1, y0, h, y_min, y_max), fill=0)
            c.line(xs - box_w / 2, y_scale(med, y0, h, y_min, y_max), xs + box_w / 2, y_scale(med, y0, h, y_min, y_max))
    draw_legend(c, x0 + w - 120, y0 + h + 8, palette, ESTIMATOR_DESIGNS)
    c.save()


def padded_range(vals: Sequence[float], min_pad: float = 0.1) -> Tuple[float, float]:
    arr = np.asarray([v for v in vals if pd.notna(v)], dtype=float)
    if arr.size == 0:
        return -1.0, 1.0
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if lo == hi:
        pad = max(min_pad, abs(lo) * 0.1)
        return lo - pad, hi + pad
    pad = max(min_pad, 0.08 * (hi - lo))
    return lo - pad, hi + pad


def draw_legend(c, x, y, palette, labels):
    c.setFont("Helvetica", 8)
    for i, label in enumerate(labels):
        color = palette[label] if label in palette else palette.get(str(label))
        c.setFillColor(color)
        c.rect(x, y - i * 12, 8, 8, fill=1, stroke=0)
        c.setFillColorRGB(0, 0, 0)
        c.drawString(x + 12, y - i * 12, DESIGN_DISPLAY_LABELS.get(str(label), str(label)))


def aggregate_and_write(
    output_dir: Path,
    estimator_col: str = "ipw",
    write_diagnostic_figures: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    path1 = output_dir / "run_level_results_scenario1.csv"
    path2 = output_dir / "run_level_results_scenario2.csv"
    run1 = pd.read_csv(path1) if path1.exists() else pd.DataFrame()
    run2 = pd.read_csv(path2) if path2.exists() else pd.DataFrame()
    summary1 = (
        aggregate_summary(run1, 1, estimator_col=estimator_col)
        if not run1.empty
        else pd.DataFrame()
    )
    summary2 = (
        aggregate_summary(run2, 2, estimator_col=estimator_col)
        if not run2.empty
        else pd.DataFrame()
    )
    summary1.to_csv(output_dir / "summary_scenario1_full_grid.csv", index=False)
    summary2.to_csv(output_dir / "summary_scenario2_full_grid.csv", index=False)
    selected = selected_capacity_rows(summary1, summary2)
    selected.to_csv(output_dir / "selected_capacity_regimes.csv", index=False)
    save_raw_estimates_selected(run1, selected, 1, output_dir / "raw_estimates_scenario1_selected.csv")
    save_raw_estimates_selected(run2, selected, 2, output_dir / "raw_estimates_scenario2_selected.csv")
    warnings = generate_warnings(summary1, summary2, selected)
    write_warnings(warnings, output_dir)
    if write_diagnostic_figures:
        make_figures(output_dir, summary1, summary2, selected, estimator_col=estimator_col)
    return summary1, summary2, selected, warnings


def write_metadata(args: argparse.Namespace, output_dir: Path) -> None:
    metadata = {
        "base_seed": args.base_seed,
        "p_treat": P_TREAT,
        "param_mode": args.param_mode,
        "assignment_mode": args.assignment_mode,
        "n_strata": args.n_strata,
        "summary_estimator": args.estimator,
        "scenario1": {
            "N": args.n,
            "T": args.t,
            "r_gte": args.r_gte,
            "r_design": args.r_design,
            "capacity_grid": args.capacity_grid,
            "delta_control": args.delta_control,
            "delta_treat": args.delta_treat,
        },
        "scenario2": {
            "N_list": args.n_list,
            "T": args.t_scenario2,
            "r_gte": args.r_gte_scenario2,
            "r_design": args.r_design_scenario2,
            "capacity_grid": args.capacity_grid_scenario2,
            "eta_control": args.eta_control,
            "eta_treat": args.eta_treat,
        },
        "notes": [
            "Controlled stochastic uniform-demand simulation, separate from trace-driven FreshRetailNet experiments.",
            "Diagnostics are reported and not used to filter simulation runs.",
        ],
    }
    with (output_dir / "simulation_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--write-every", type=int, default=50)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true", default=True)
    resume_group.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--skip-run", action="store_true", help="Only aggregate existing run-level CSVs.")
    parser.add_argument("--scenario", choices=("1", "2", "both"), default="both")
    parser.add_argument(
        "--param-mode",
        choices=(
            "baseline",
            "retail_heterogeneous",
            "demand_heterogeneous",
            "moderate_demand_heterogeneous",
            "scenario2_high_signal",
            "scenario2_clean_signal",
        ),
        default="baseline",
    )
    parser.add_argument(
        "--estimator",
        choices=("ipw", "diff_in_means"),
        default="ipw",
        help="Estimator used for summaries, bias checks, selected regimes, and figures.",
    )
    parser.add_argument(
        "--skip-diagnostic-figures",
        action="store_true",
        help=(
            "Do not write the optional ReportLab PDF diagnostics under figures/. "
            "Use plot_abtest_violins.py for the main paper-style PNG figures."
        ),
    )
    parser.add_argument(
        "--assignment-mode",
        choices=("bernoulli", "balanced", "stratified"),
        default="bernoulli",
    )
    parser.add_argument("--n-strata", type=int, default=20)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--t", type=int, default=30)
    parser.add_argument("--r-gte", type=int, default=500)
    parser.add_argument("--r-design", type=int, default=300)
    parser.add_argument("--delta-control", type=float, default=-0.35)
    parser.add_argument("--delta-treat", type=float, default=-0.10)
    parser.add_argument("--n-list", type=parse_int_list, default=[50, 100, 300])
    parser.add_argument("--t-scenario2", type=int, default=30)
    parser.add_argument("--r-gte-scenario2", type=int, default=500)
    parser.add_argument("--r-design-scenario2", type=int, default=300)
    parser.add_argument("--eta-control", type=float, default=2.5)
    parser.add_argument("--eta-treat", type=float, default=1.0)
    parser.add_argument(
        "--capacity-grid",
        type=parse_float_list,
        default=[0.88, 0.92, 0.96, 1.00, 1.04, 1.08, 1.12],
    )
    parser.add_argument(
        "--capacity-grid-scenario2",
        type=parse_float_list,
        default=None,
        help="Scenario 2 capacity grid. Defaults to --capacity-grid when omitted.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.capacity_grid_scenario2 is None:
        args.capacity_grid_scenario2 = args.capacity_grid
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_run:
        if args.scenario in ("1", "both"):
            tasks1, params1 = build_tasks(
                scenario=1,
                N_values=[args.n],
                T=args.t,
                capacity_grid=args.capacity_grid,
                r_gte=args.r_gte,
                r_design=args.r_design,
                base_seed=args.base_seed,
                delta_control=args.delta_control,
                delta_treat=args.delta_treat,
                eta_control=args.eta_control,
                eta_treat=args.eta_treat,
                param_mode=args.param_mode,
                assignment_mode=args.assignment_mode,
                n_strata=args.n_strata,
            )
            execute_tasks(
                tasks1,
                params1,
                args.output_dir / "run_level_results_scenario1.csv",
                workers=args.workers,
                write_every=args.write_every,
                resume=args.resume,
            )
        if args.scenario in ("2", "both"):
            tasks2, params2 = build_tasks(
                scenario=2,
                N_values=args.n_list,
                T=args.t_scenario2,
                capacity_grid=args.capacity_grid_scenario2,
                r_gte=args.r_gte_scenario2,
                r_design=args.r_design_scenario2,
                base_seed=args.base_seed,
                delta_control=args.delta_control,
                delta_treat=args.delta_treat,
                eta_control=args.eta_control,
                eta_treat=args.eta_treat,
                param_mode=args.param_mode,
                assignment_mode=args.assignment_mode,
                n_strata=args.n_strata,
            )
            execute_tasks(
                tasks2,
                params2,
                args.output_dir / "run_level_results_scenario2.csv",
                workers=args.workers,
                write_every=args.write_every,
                resume=args.resume,
            )

    write_metadata(args, args.output_dir)
    summary1, summary2, selected, warnings = aggregate_and_write(
        args.output_dir,
        estimator_col=args.estimator,
        write_diagnostic_figures=not args.skip_diagnostic_figures,
    )
    print(f"Wrote outputs to {args.output_dir}")
    print(f"Scenario 1 summary rows: {len(summary1)}")
    print(f"Scenario 2 summary rows: {len(summary2)}")
    print(f"Selected regimes: {len(selected)}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("No selected-regime theory-region validation warnings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
