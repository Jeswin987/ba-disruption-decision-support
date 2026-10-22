#!/usr/bin/env python3
"""
final_check_v03.py

Final check of the model chosen by model_selection_v03.py, plus a reference.

WHAT THIS IS (AND IS NOT)
    The saved test events were already inspected during development, so this
    is SUPPORTING EVIDENCE, not an independent final test. Do not describe it
    as a new, untouched test set.

WHAT IT DOES
    1. Confirms the pre-set winner (winner.json) is the MLP, as this script
       was designed for. If it is not, the script stops.
    2. Tunes the MLP and HGB on ALL training events with the same, larger
       search budget (event-grouped inner folds, same capture-based score).
       Tuned settings are saved per model, so tuning can resume.
    3. Refits each model on all training events with its tuned settings and
       scores the saved test events ONCE.
    4. Reports capture at 5/10/20/30%, ROC-AUC, PR-AUC and raw Brier score,
       with 95% intervals from resampling whole events, and a paired
       MLP-minus-HGB difference.

LOOK-ONCE GUARD
    If test_metrics.csv already exists, the script refuses to score the test
    events again. Repeated looking would turn this into tuning on the test
    set. Delete that file deliberately only if you accept that consequence.

USAGE
    python final_check_v03.py [package_dir] [--quick] [--selection-dir DIR]
                              [--out-dir DIR]

    --quick   Smoke test. Uses a small sample of TRAINING events and holds
              out part of them in place of the real test events, so the real
              test events are never touched. Results are not meaningful.

    Requires model_selection_v03.py in the same folder (it provides the
    features, the capture metric, the model definitions and search spaces).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold

from model_selection_v03 import (
    CAPACITIES,
    CAPACITY_LABELS,
    FEATURES,
    GROUP_COL,
    MEAN_CAPTURE_SCORER,
    OUTCOME_COLUMNS,
    QUICK_EVENTS,
    SEED,
    TARGET_COL,
    _read_event_ids,
    build_models,
    capture_profile,
    load_training_data,
)

# --------------------------------------------------------------------------
# Fixed definitions
# --------------------------------------------------------------------------

WINNER_MODEL = "mlp_neural_network"          # pre-set winner this script expects
REFERENCE_MODEL = "hist_gradient_boosting"   # reference, same budget
FINAL_MODELS = [WINNER_MODEL, REFERENCE_MODEL]

# Larger budget than the selection run (which used 12 settings, 3 inner folds).
FINAL_SETTINGS = {"inner_splits": 5, "n_iter": 30, "bootstrap_replicates": 1000}
QUICK_SETTINGS = {"inner_splits": 2, "n_iter": 2, "bootstrap_replicates": 50}

PASSENGER_ID_COL = "passenger_id"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_test_rows(package_dir: Path) -> pd.DataFrame:
    """Passenger rows for the saved TEST events, with the same checks as training."""
    passenger_path = package_dir / "data" / "passenger_itineraries_v03_10000.csv"
    test_path = package_dir / "event_split" / "test_event_ids.csv"
    needed = [PASSENGER_ID_COL, GROUP_COL, TARGET_COL, *FEATURES]
    df = pd.read_csv(passenger_path, usecols=needed)
    test_ids = set(_read_event_ids(test_path))
    test_df = df[df[GROUP_COL].isin(test_ids)].reset_index(drop=True)
    if test_df[GROUP_COL].nunique() != len(test_ids):
        raise ValueError("Some test event IDs have no rows in the passenger file.")
    values = test_df[[*FEATURES, TARGET_COL]]
    if values.isna().any().any() or not np.isfinite(values.to_numpy(float)).all():
        raise ValueError("Missing or non-finite values in the test rows.")
    if not set(test_df[TARGET_COL].unique()) <= {0, 1}:
        raise ValueError(f"{TARGET_COL} must contain only 0 and 1.")
    return test_df


def quick_split(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Smoke-test split: sampled training events, with 20% held out as a
    stand-in for the test events. The real test events are not read."""
    rng = np.random.default_rng(SEED)
    events = np.sort(train_df[GROUP_COL].unique())
    chosen = rng.choice(events, size=min(QUICK_EVENTS, len(events)), replace=False)
    held_out = set(rng.choice(chosen, size=len(chosen) // 5, replace=False))
    sampled = train_df[train_df[GROUP_COL].isin(chosen)]
    fake_test = sampled[sampled[GROUP_COL].isin(held_out)].reset_index(drop=True)
    fake_train = sampled[~sampled[GROUP_COL].isin(held_out)].reset_index(drop=True)
    # The training loader does not read passenger_id, so quick mode uses a row
    # number in its place (real mode reads the true IDs from the data file).
    fake_test[PASSENGER_ID_COL] = "quick_row_" + fake_test.index.astype(str)
    return fake_train, fake_test


# --------------------------------------------------------------------------
# Saved tuned settings (JSON cannot hold numpy types or tuples directly)
# --------------------------------------------------------------------------

def _to_jsonable(params: dict) -> dict:
    out = {}
    for key, value in params.items():
        if isinstance(value, tuple):
            value = [int(v) for v in value]
        elif isinstance(value, np.integer):
            value = int(value)
        elif isinstance(value, np.floating):
            value = float(value)
        out[key] = value
    return out


def _restore_params(params: dict) -> dict:
    restored = dict(params)
    for key in restored:
        if key.endswith("hidden_layer_sizes"):
            restored[key] = tuple(int(v) for v in restored[key])
    return restored


# --------------------------------------------------------------------------
# Stage 1: tuning on all training events
# --------------------------------------------------------------------------

def tune_model(name: str, train_df: pd.DataFrame, settings: dict,
               params_path: Path) -> dict:
    """Random search on ALL training events, event-grouped inner folds."""
    if params_path.exists():
        print(f"[{name}] tuned settings already saved; skipping tuning.")
        return json.loads(params_path.read_text())

    estimator, space = build_models()[name]
    inner_cv = StratifiedGroupKFold(
        n_splits=settings["inner_splits"], shuffle=True, random_state=SEED
    )
    search = RandomizedSearchCV(
        estimator,
        space,
        n_iter=settings["n_iter"],
        scoring=MEAN_CAPTURE_SCORER,
        cv=inner_cv,
        refit=False,          # the final refit happens in stage 2
        random_state=SEED,
        n_jobs=1,
        error_score="raise",
        verbose=2,            # one progress line per fit
    )
    print(f"[{name}] tuning: {settings['n_iter']} settings x "
          f"{settings['inner_splits']} inner folds")
    started = time.time()
    search.fit(train_df[FEATURES], train_df[TARGET_COL].to_numpy(),
               groups=train_df[GROUP_COL].to_numpy())
    result = {
        "model": name,
        "best_params": _to_jsonable(search.best_params_),
        "best_inner_cv_mean_capture": float(search.best_score_),
        "tuning_minutes": round((time.time() - started) / 60, 1),
    }
    params_path.write_text(json.dumps(result, indent=2))
    print(f"[{name}] best inner-CV mean capture "
          f"{result['best_inner_cv_mean_capture']:.4f} "
          f"({result['tuning_minutes']} min)")
    return result


# --------------------------------------------------------------------------
# Stage 2: score the test events once
# --------------------------------------------------------------------------

def fit_and_predict(name: str, tuned: dict, train_df: pd.DataFrame,
                    test_df: pd.DataFrame) -> np.ndarray:
    """Refit on all training events with the tuned settings; score the test rows."""
    estimator, _ = build_models()[name]
    estimator.set_params(**_restore_params(tuned["best_params"]))
    estimator.fit(train_df[FEATURES], train_df[TARGET_COL].to_numpy())
    return estimator.predict_proba(test_df[FEATURES])[:, 1]


def point_metrics(y: np.ndarray, scores: np.ndarray) -> dict:
    profile = capture_profile(y, scores)
    row = {f"capture_{label}pct": value for label, value in profile.items()}
    row["mean_capture"] = float(np.mean(list(profile.values())))
    row["roc_auc"] = float(roc_auc_score(y, scores))
    row["pr_auc"] = float(average_precision_score(y, scores))
    row["brier"] = float(brier_score_loss(y, scores))  # raw, unweighted
    return row


def event_bootstrap(test_df: pd.DataFrame, score_map: dict[str, np.ndarray],
                    replicates: int) -> pd.DataFrame:
    """Resample whole EVENTS with replacement. The same resamples are used for
    every model, so differences between models are properly paired."""
    rng = np.random.default_rng(SEED)
    y = test_df[TARGET_COL].to_numpy()
    events = test_df[GROUP_COL].to_numpy()
    order = np.argsort(events, kind="stable")
    unique, starts = np.unique(events[order], return_index=True)
    bounds = np.append(starts, len(events))
    rows_by_event = [order[bounds[i]:bounds[i + 1]] for i in range(len(unique))]

    records = []
    for b in range(replicates):
        picked = rng.integers(0, len(unique), size=len(unique))
        idx = np.concatenate([rows_by_event[i] for i in picked])
        for name, scores in score_map.items():
            yb, sb = y[idx], scores[idx]
            profile = capture_profile(yb, sb)
            records.append({
                "replicate": b,
                "model": name,
                **{f"capture_{k}pct": v for k, v in profile.items()},
                "mean_capture": float(np.mean(list(profile.values()))),
                "roc_auc": float(roc_auc_score(yb, sb)),
            })
    return pd.DataFrame(records)


def score_test_once(train_df: pd.DataFrame, test_df: pd.DataFrame,
                    tuned: dict[str, dict], settings: dict, out_dir: Path) -> None:
    y = test_df[TARGET_COL].to_numpy()
    score_map, metric_rows = {}, []
    for name in FINAL_MODELS:
        started = time.time()
        scores = fit_and_predict(name, tuned[name], train_df, test_df)
        score_map[name] = scores
        metric_rows.append({"model": name, **point_metrics(y, scores),
                            "refit_and_score_seconds": round(time.time() - started, 1)})
        print(f"[{name}] refit on all training events and scored the test events")

    # Predictions are saved so nothing ever needs to be rescored.
    predictions = test_df[[PASSENGER_ID_COL, GROUP_COL, TARGET_COL]].copy()
    for name, scores in score_map.items():
        predictions[f"score_{name}"] = scores
    predictions.to_csv(out_dir / "test_predictions.csv", index=False)

    boot = event_bootstrap(test_df, score_map, settings["bootstrap_replicates"])
    metrics_to_ci = [f"capture_{label}pct" for label in CAPACITY_LABELS]
    metrics_to_ci += ["mean_capture", "roc_auc"]
    intervals = []
    for name in FINAL_MODELS:
        sub = boot[boot["model"] == name]
        for metric in metrics_to_ci:
            intervals.append({
                "model": name, "metric": metric,
                "ci_low_2_5": float(sub[metric].quantile(0.025)),
                "ci_high_97_5": float(sub[metric].quantile(0.975)),
            })
    pd.DataFrame(intervals).to_csv(out_dir / "test_bootstrap_intervals.csv", index=False)

    wide = boot.pivot(index="replicate", columns="model", values="mean_capture")
    diff = wide[WINNER_MODEL] - wide[REFERENCE_MODEL]
    paired = {
        "comparison": f"{WINNER_MODEL} minus {REFERENCE_MODEL} (mean capture)",
        "ci_low_2_5": float(diff.quantile(0.025)),
        "ci_high_97_5": float(diff.quantile(0.975)),
        "share_of_resamples_winner_ahead": float((diff > 0).mean()),
        "replicates": int(len(diff)),
    }
    (out_dir / "test_paired_difference.json").write_text(json.dumps(paired, indent=2))

    # Written LAST: its existence is the look-once guard.
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_dir / "test_metrics.csv", index=False)

    n_events = test_df[GROUP_COL].nunique()
    print(f"\n=== Test events: {n_events} events, {len(test_df)} passengers, "
          f"{int(y.sum())} severe ===")
    print("SUPPORTING EVIDENCE ONLY: these test events were inspected during "
          "development.\n")
    show = [f"capture_{label}pct" for label in CAPACITY_LABELS]
    show += ["mean_capture", "roc_auc", "pr_auc", "brier"]
    print(metrics.set_index("model")[show].round(4).to_string())
    print("\n95% intervals from resampling whole events:")
    print(pd.DataFrame(intervals).round(4).to_string(index=False))
    print(f"\nPaired {paired['comparison']}: "
          f"95% interval [{paired['ci_low_2_5']:.4f}, {paired['ci_high_97_5']:.4f}], "
          f"winner ahead in {paired['share_of_resamples_winner_ahead']:.1%} of resamples")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def check_winner(selection_dir: Path) -> None:
    path = selection_dir / "winner.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run model_selection_v03.py first: {path} not found.")
    winner = json.loads(path.read_text())["winner"]
    if winner != WINNER_MODEL:
        raise SystemExit(
            f"The pre-set winner in {path} is '{winner}', but this script was "
            f"designed for '{WINNER_MODEL}'. The design needs to be revisited."
        )


def check_or_write_config(out_dir: Path, config: dict) -> None:
    path = out_dir / "final_check_config.json"
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(config)):
            raise SystemExit(
                f"{path} differs from the current settings, data or library "
                "versions. Use a new --out-dir for a different run."
            )
    else:
        path.write_text(json.dumps(config, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("package_dir", nargs="?", default=".")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test; real test events are not touched")
    parser.add_argument("--selection-dir", default="model_selection_outputs")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    package_dir = Path(args.package_dir)
    settings = QUICK_SETTINGS if args.quick else FINAL_SETTINGS
    out_dir = Path(args.out_dir or (
        "final_check_outputs_quick" if args.quick else "final_check_outputs"
    ))

    # Look-once guard, checked before any expensive work.
    if (out_dir / "test_metrics.csv").exists():
        raise SystemExit(
            f"{out_dir / 'test_metrics.csv'} already exists: the test events "
            "have already been scored. Scoring them again would turn this into "
            "tuning on the test set. Delete the file only if you accept that."
        )

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    check_winner(Path(args.selection_dir))

    train_df, data_info = load_training_data(package_dir)
    if args.quick:
        train_df, test_df = quick_split(train_df)
    else:
        test_df = load_test_rows(package_dir)

    if set(FEATURES) & set(OUTCOME_COLUMNS):
        raise ValueError("Outcome columns are listed as features.")
    if set(train_df[GROUP_COL]) & set(test_df[GROUP_COL]):
        raise ValueError("Training and test events overlap.")

    print(f"Training events: {train_df[GROUP_COL].nunique()}, "
          f"rows: {len(train_df)}")
    print(f"Scoring set events: {test_df[GROUP_COL].nunique()}, "
          f"rows: {len(test_df)}"
          f"{' (quick mode: held-out TRAINING events)' if args.quick else ''}")
    print(f"scikit-learn {sklearn.__version__}, numpy {np.__version__}, "
          f"pandas {pd.__version__}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tuned_params").mkdir(exist_ok=True)
    config = {
        "models": FINAL_MODELS, "settings": settings, "seed": SEED,
        "quick": args.quick, "features": FEATURES,
        "capacities": list(CAPACITIES), "data": data_info,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
    }
    check_or_write_config(out_dir, config)

    tuned = {name: tune_model(name, train_df, settings,
                              out_dir / "tuned_params" / f"{name}.json")
             for name in FINAL_MODELS}
    score_test_once(train_df, test_df, tuned, settings, out_dir)
    print(f"\nFiles written to: {out_dir}")


if __name__ == "__main__":
    main()