#!/usr/bin/env python3
"""
model_selection_v03.py

Pre-registered model selection for the connection-disruption risk-ranking
prototype (synthetic data, v0.3).

PURPOSE
    Decide which of five model families ranks severe passengers best when
    assistance capacity is limited, using TRAINING EVENTS ONLY. The saved
    held-out test events are never read for scoring. (Their IDs are read once,
    only to confirm they do not overlap the training events.)

THE RULE (fixed before any result was seen)
    Winner = the model with the highest average, over all outer folds, of the
    "mean capture": the share of severe passengers found when the top 5%, 10%,
    20% and 30% of passengers are selected, averaged over those four levels.
    If two models are close, the higher score still wins. The output reports
    whether the winner's lead is smaller than the fold-to-fold spread, so the
    README can say so honestly.

DESIGN
    * Same 10 operational features for every model; outcome columns are
      forbidden as inputs and the script stops if one appears.
    * Outer evaluation: repeated, event-grouped, stratified 5-fold
      cross-validation. All passengers of one event stay in one fold.
    * Inner tuning (nested): inside each outer training set, every model gets
      the same random-search budget, scored on the same capture metric with
      event-grouped inner folds. Tuning never sees the outer test fold.
    * No class weights anywhere, so Brier scores ("raw") are comparable.

USAGE
    python model_selection_v03.py [package_dir] [--quick] [--models NAME ...]
                                  [--out-dir DIR]

    package_dir  Folder containing data/ and event_split/ (default: current).
    --quick      Smoke test on a small sample of events with a tiny search.
                 Results go to a separate folder and are NOT meaningful.
    --models     Run only some models (useful for splitting a long run).
    --out-dir    Output folder (default: model_selection_outputs, or
                 model_selection_outputs_quick with --quick).

    The run is resumable: finished (model, repeat, fold) results are saved
    after every fit, and re-running the same command continues from there.
    Changing any setting requires a new --out-dir (the script checks this).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import loguniform
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# Fixed definitions (the "pre-registered" part)
# --------------------------------------------------------------------------

GROUP_COL = "event_id"
TARGET_COL = "severe_connection_disruption"

# The 10 operational features used by the project's models (from README.md).
FEATURES = [
    "weather_severity",
    "atc_restriction",
    "airport_congestion",
    "inbound_operational_pressure",
    "forecast_delay_minutes",
    "connection_buffer_minutes",
    "expected_transfer_minutes",
    "expected_alternative_flights",
    "expected_recovery_options",
    "number_of_connections",
]

# Columns that describe what happened AFTER the disruption (or define the
# label). They must never be model inputs.
OUTCOME_COLUMNS = [
    "event_actual_delay_minutes",
    "actual_transfer_minutes",
    "original_connection_missed",
    "original_final_arrival_minutes",
    "same_day_recovery_available",
    "best_recovery_arrival_minutes",
    "actual_recovery_success",
    TARGET_COL,
]

CAPACITIES = (0.05, 0.10, 0.20, 0.30)
CAPACITY_LABELS = tuple(int(round(c * 100)) for c in CAPACITIES)

SEED = 42
FULL_SETTINGS = {"outer_splits": 5, "repeats": 3, "inner_splits": 3, "n_iter": 12}
QUICK_SETTINGS = {"outer_splits": 3, "repeats": 1, "inner_splits": 2, "n_iter": 2}
QUICK_EVENTS = 600

MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
    "extra_trees",
    "mlp_neural_network",
]

WINNER_RULE = (
    "Highest average of mean capture (share of severe passengers found in the "
    "top 5%, 10%, 20% and 30% of the ranked queue, averaged over those four "
    "levels) across all outer folds. Higher score wins even if the gap is small."
)


# --------------------------------------------------------------------------
# Metric: capture at limited capacity
# --------------------------------------------------------------------------

def capture_at(y_true: np.ndarray, scores: np.ndarray, capacity: float,
               tiebreak: np.ndarray) -> float:
    """Share of all severe cases found in the top `capacity` of the queue.

    The queue size is floor(n * capacity), the same floor rule the project
    README uses. A tiny epsilon guards against floating-point error (for
    example 20 * 0.05 evaluating to 0.9999999999). Ties in score are broken
    with a fixed random vector so ties never favour row order.
    """
    n_selected = int(math.floor(len(y_true) * capacity + 1e-9))
    order = np.lexsort((tiebreak, -scores))  # best score first, ties by tiebreak
    return float(y_true[order[:n_selected]].sum() / y_true.sum())


def capture_profile(y_true: np.ndarray, scores: np.ndarray) -> dict[int, float]:
    """Capture at each capacity level, with a fixed tie-break vector."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    if y_true.sum() == 0:
        raise ValueError("Evaluation set contains no severe cases.")
    tiebreak = np.random.default_rng(SEED).random(len(y_true))
    return {
        label: capture_at(y_true, scores, cap, tiebreak)
        for label, cap in zip(CAPACITY_LABELS, CAPACITIES)
    }


def mean_capture(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average capture over all capacity levels (the headline measure)."""
    return float(np.mean(list(capture_profile(y_true, scores).values())))


# Scorer used for inner tuning, so tuning optimises the same measure that
# decides the winner. `response_method="predict_proba"` passes the
# positive-class probability for binary problems.
MEAN_CAPTURE_SCORER = make_scorer(
    mean_capture, response_method="predict_proba", greater_is_better=True
)


# --------------------------------------------------------------------------
# Data loading and safety checks
# --------------------------------------------------------------------------

def load_training_data(package_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Load passenger rows for the saved TRAINING events only.

    Stops with a clear message if anything about the split or columns is not
    as expected, rather than guessing.
    """
    passenger_path = package_dir / "data" / "passenger_itineraries_v03_10000.csv"
    train_path = package_dir / "event_split" / "train_event_ids.csv"
    test_path = package_dir / "event_split" / "test_event_ids.csv"
    for path in (passenger_path, train_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    # Inputs must never overlap with outcome columns.
    overlap = sorted(set(FEATURES) & set(OUTCOME_COLUMNS))
    if overlap:
        raise ValueError(f"Outcome columns listed as features: {overlap}")

    needed = [GROUP_COL, TARGET_COL, *FEATURES]
    header = list(pd.read_csv(passenger_path, nrows=0).columns)
    missing = [c for c in needed if c not in header]
    if missing:
        raise ValueError(f"Passenger file is missing columns: {missing}")

    # Read only the columns we need; nothing else can leak into the model.
    df = pd.read_csv(passenger_path, usecols=needed)

    train_ids = _read_event_ids(train_path)
    test_ids = _read_event_ids(test_path)
    if not train_ids.is_unique or not test_ids.is_unique:
        raise ValueError("Duplicate event IDs in a split file.")
    shared = set(train_ids) & set(test_ids)
    if shared:
        raise ValueError(f"{len(shared)} event IDs appear in both train and test.")
    all_events = set(df[GROUP_COL].unique())
    if set(train_ids) | set(test_ids) != all_events:
        raise ValueError(
            "Train + test event IDs do not exactly cover the events in the "
            "passenger file. Check the split files before continuing."
        )

    train_df = df[df[GROUP_COL].isin(set(train_ids))].reset_index(drop=True)

    values = train_df[[*FEATURES, TARGET_COL]]
    if values.isna().any().any():
        raise ValueError("Missing values found in features or target.")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite values found in features or target.")
    if not set(train_df[TARGET_COL].unique()) <= {0, 1}:
        raise ValueError(f"{TARGET_COL} must contain only 0 and 1.")

    info = {
        "train_events": int(train_df[GROUP_COL].nunique()),
        "train_rows": int(len(train_df)),
        "train_severe_rate": float(train_df[TARGET_COL].mean()),
        "test_events_excluded": int(len(test_ids)),
    }
    return train_df, info


def _read_event_ids(path: Path) -> pd.Series:
    table = pd.read_csv(path)
    if GROUP_COL not in table.columns:
        raise ValueError(f"{path} has no '{GROUP_COL}' column.")
    return table[GROUP_COL].astype(str)


# --------------------------------------------------------------------------
# Models and search spaces (identical search budget for every model)
# --------------------------------------------------------------------------

def build_models() -> dict[str, tuple]:
    """Return {name: (estimator, search_space)}.

    No model uses class weights. Models that depend on feature scale
    (logistic regression, MLP) get a StandardScaler inside a Pipeline, so it
    is fitted on training data only within every fit.
    """
    return {
        "logistic_regression": (
            Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000)),
            ]),
            {"clf__C": loguniform(1e-3, 1e2)},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED),
            {
                "max_depth": [6, 10, 14, None],
                "min_samples_leaf": [1, 5, 20, 50],
                "max_features": ["sqrt", 0.5, 1.0],
            },
        ),
        "hist_gradient_boosting": (
            # early_stopping=False: the built-in early stopping would hold out
            # a random validation slice that ignores event groups.
            HistGradientBoostingClassifier(early_stopping=False, random_state=SEED),
            {
                "learning_rate": loguniform(0.02, 0.3),
                "max_iter": [100, 200, 400],
                "max_leaf_nodes": [15, 31, 63],
                "min_samples_leaf": [20, 50, 100],
                "l2_regularization": [0.0, 0.1, 1.0, 10.0],
            },
        ),
        "extra_trees": (
            ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=SEED),
            {
                "max_depth": [6, 10, 14, None],
                "min_samples_leaf": [1, 5, 20, 50],
                "max_features": ["sqrt", 0.5, 1.0],
            },
        ),
        "mlp_neural_network": (
            # early_stopping left off for the same reason as HGB above.
            Pipeline([
                ("scale", StandardScaler()),
                ("clf", MLPClassifier(max_iter=300, random_state=SEED)),
            ]),
            {
                "clf__hidden_layer_sizes": [(32,), (64,), (32, 16), (64, 32)],
                "clf__alpha": loguniform(1e-5, 1e-1),
                "clf__learning_rate_init": loguniform(1e-4, 1e-2),
            },
        ),
    }


def _converged(estimator) -> float:
    """1.0/0.0 if the fitted model is iterative and reports convergence, else NaN."""
    step = estimator[-1] if isinstance(estimator, Pipeline) else estimator
    if isinstance(step, (LogisticRegression, MLPClassifier)):
        return float(int(np.max(step.n_iter_)) < step.max_iter)
    return float("nan")


# --------------------------------------------------------------------------
# Running the comparison
# --------------------------------------------------------------------------

def run_config(settings: dict, quick: bool, data_info: dict) -> dict:
    """Everything that must stay identical between a run and its resumption."""
    return {
        "features": FEATURES,
        "capacities": list(CAPACITIES),
        "seed": SEED,
        "quick": quick,
        "settings": settings,
        "data": data_info,
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "winner_rule": WINNER_RULE,
    }


def check_or_write_config(out_dir: Path, config: dict) -> None:
    path = out_dir / "run_config.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved != json.loads(json.dumps(config)):
            raise SystemExit(
                f"{path} differs from the current settings, data or library "
                "versions. Use a new --out-dir for a different run."
            )
    else:
        path.write_text(json.dumps(config, indent=2))


def load_done_keys(scores_path: Path) -> set[tuple[str, int, int]]:
    if not scores_path.exists():
        return set()
    done = pd.read_csv(scores_path)
    return set(zip(done["model"], done["repeat"], done["fold"]))


def evaluate_fold(name: str, estimator, space: dict, settings: dict,
                  X_train, y_train, groups_train, X_test, y_test) -> dict:
    """Tune on the outer-training part only, then score on the outer-test fold."""
    inner_cv = StratifiedGroupKFold(
        n_splits=settings["inner_splits"], shuffle=True, random_state=SEED
    )
    search = RandomizedSearchCV(
        estimator,
        space,
        n_iter=settings["n_iter"],
        scoring=MEAN_CAPTURE_SCORER,
        cv=inner_cv,
        refit=True,
        random_state=SEED,   # same candidate settings in every outer fold
        n_jobs=1,
        error_score="raise",
    )
    started = time.time()
    search.fit(X_train, y_train, groups=groups_train)
    scores = search.best_estimator_.predict_proba(X_test)[:, 1]
    elapsed = time.time() - started

    profile = capture_profile(y_test, scores)
    row = {f"capture_{label}pct": value for label, value in profile.items()}
    row.update({
        "mean_capture": float(np.mean(list(profile.values()))),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "pr_auc": float(average_precision_score(y_test, scores)),
        "brier": float(brier_score_loss(y_test, scores)),  # raw, unweighted
        "n_test_rows": int(len(y_test)),
        "n_test_severe": int(y_test.sum()),
        "fit_seconds": round(elapsed, 1),
        "converged": _converged(search.best_estimator_),
        "best_params": json.dumps(search.best_params_, default=str),
    })
    return row


def run_comparison(train_df: pd.DataFrame, settings: dict, selected: list[str],
                   out_dir: Path) -> None:
    X = train_df[FEATURES]
    y = train_df[TARGET_COL].to_numpy()
    groups = train_df[GROUP_COL].to_numpy()

    scores_path = out_dir / "fold_scores.csv"
    done = load_done_keys(scores_path)
    models = build_models()
    total = settings["repeats"] * settings["outer_splits"] * len(selected)
    print(f"Planned fits this invocation: up to {total} "
          f"({len(done)} already finished)")

    for repeat in range(settings["repeats"]):
        outer_cv = StratifiedGroupKFold(
            n_splits=settings["outer_splits"], shuffle=True, random_state=repeat
        )
        for fold, (tr, te) in enumerate(outer_cv.split(X, y, groups)):
            for name in selected:
                if (name, repeat, fold) in done:
                    continue
                estimator, space = models[name]
                row = evaluate_fold(
                    name, estimator, space, settings,
                    X.iloc[tr], y[tr], groups[tr], X.iloc[te], y[te],
                )
                row = {"model": name, "repeat": repeat, "fold": fold, **row}
                pd.DataFrame([row]).to_csv(
                    scores_path, mode="a", header=not scores_path.exists(),
                    index=False,
                )
                print(f"[{name}] repeat {repeat} fold {fold}: "
                      f"mean capture {row['mean_capture']:.4f}, "
                      f"AUC {row['roc_auc']:.4f}, {row['fit_seconds']}s")


# --------------------------------------------------------------------------
# Summary and winner
# --------------------------------------------------------------------------

def summarise(out_dir: Path, settings: dict) -> None:
    scores_path = out_dir / "fold_scores.csv"
    if not scores_path.exists():
        print("No results yet; nothing to summarise.")
        return
    scores = pd.read_csv(scores_path)
    expected = settings["repeats"] * settings["outer_splits"]
    counts = scores.groupby("model").size().reindex(MODEL_NAMES).fillna(0).astype(int)
    incomplete = counts[counts != expected]
    if len(incomplete):
        print("\nNot all models are finished, so no winner is declared yet.")
        print(f"Completed folds per model (need {expected} each):")
        print(counts.to_string())
        print("Run the same command again to continue.")
        return

    metrics = [f"capture_{label}pct" for label in CAPACITY_LABELS]
    metrics += ["mean_capture", "roc_auc", "pr_auc", "brier"]
    table = scores.groupby("model")[metrics].agg(["mean", "std"])
    table.columns = [f"{m}_{s}" for m, s in table.columns]
    per_repeat = scores.groupby(["model", "repeat"])["mean_capture"].mean().unstack()
    per_repeat.columns = [f"mean_capture_repeat{c}" for c in per_repeat.columns]
    table = table.join(per_repeat)
    table["total_fit_minutes"] = scores.groupby("model")["fit_seconds"].sum() / 60
    table = table.sort_values("mean_capture_mean", ascending=False)
    table.to_csv(out_dir / "summary_by_model.csv")

    # Paired comparison: same outer folds for every model.
    wide = scores.pivot_table(index=["repeat", "fold"], columns="model",
                              values="mean_capture")
    winner = table.index[0]
    paired_rows = []
    for other in table.index[1:]:
        diff = wide[winner] - wide[other]
        paired_rows.append({
            "winner": winner,
            "other": other,
            "mean_difference": diff.mean(),
            "sd_of_differences": diff.std(ddof=1),
            "folds_winner_better": int((diff > 0).sum()),
            "n_folds": int(len(diff)),
        })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(out_dir / "paired_vs_winner.csv", index=False)

    runner_up = paired.iloc[0]
    # Rough guide only: outer folds share training data, so fold-to-fold
    # spread understates the true uncertainty.
    within_noise = bool(runner_up["mean_difference"] < runner_up["sd_of_differences"])
    verdict = {
        "winner": winner,
        "rule": WINNER_RULE,
        "winner_mean_capture": float(table.loc[winner, "mean_capture_mean"]),
        "runner_up": runner_up["other"],
        "lead_over_runner_up": float(runner_up["mean_difference"]),
        "lead_smaller_than_fold_to_fold_spread": within_noise,
        "note": (
            "If the lead is smaller than the fold-to-fold spread, report that "
            "the model was selected by a pre-set rule, not shown to be clearly "
            "better. Selecting among five models also makes the winner's "
            "cross-validated score slightly optimistic."
        ),
    }
    (out_dir / "winner.json").write_text(json.dumps(verdict, indent=2))

    show = ["mean_capture_mean", "mean_capture_std", "capture_10pct_mean",
            "roc_auc_mean", "pr_auc_mean", "brier_mean", "total_fit_minutes"]
    print("\n=== Model comparison (training events only) ===")
    print(table[show].round(4).to_string())
    print("\n=== Paired differences vs winner (mean capture) ===")
    print(paired.round(4).to_string(index=False))
    print(f"\nWinner by the pre-set rule: {winner}")
    print(f"Lead over runner-up ({verdict['runner_up']}): "
          f"{verdict['lead_over_runner_up']:.4f}")
    print("Lead smaller than fold-to-fold spread: "
          f"{'YES' if within_noise else 'no'}")
    print(f"Files written to: {out_dir}")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("package_dir", nargs="?", default=".",
                        help="Folder containing data/ and event_split/")
    parser.add_argument("--quick", action="store_true",
                        help="Small smoke test; results are not meaningful")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES,
                        help="Run only these models")
    parser.add_argument("--out-dir", default=None, help="Output folder")
    args = parser.parse_args()

    package_dir = Path(args.package_dir)
    settings = QUICK_SETTINGS if args.quick else FULL_SETTINGS
    out_dir = Path(args.out_dir or (
        "model_selection_outputs_quick" if args.quick else "model_selection_outputs"
    ))
    selected = args.models or MODEL_NAMES

    # MLP and logistic regression may hit their iteration limit on some fits;
    # this is recorded per fit in the `converged` column, so the repeated
    # warnings are silenced to keep the progress log readable.
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    train_df, data_info = load_training_data(package_dir)
    if args.quick:
        rng = np.random.default_rng(SEED)
        events = np.sort(train_df[GROUP_COL].unique())
        chosen = rng.choice(events, size=min(QUICK_EVENTS, len(events)), replace=False)
        train_df = train_df[train_df[GROUP_COL].isin(chosen)].reset_index(drop=True)
        data_info = {**data_info, "quick_events_used": int(len(chosen)),
                     "train_rows": int(len(train_df))}

    print(f"Training events: {train_df[GROUP_COL].nunique()}, "
          f"rows: {len(train_df)}, severe rate: {train_df[TARGET_COL].mean():.3f}")
    print(f"scikit-learn {sklearn.__version__}, numpy {np.__version__}, "
          f"pandas {pd.__version__}")

    out_dir.mkdir(parents=True, exist_ok=True)
    check_or_write_config(out_dir, run_config(settings, args.quick, data_info))
    run_comparison(train_df, settings, selected, out_dir)
    summarise(out_dir, settings)


if __name__ == "__main__":
    main()