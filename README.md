# Connection-disruption risk ranking: synthetic decision-support prototype (v0.3)

> **Scope:** This is an educational prototype built from synthetic data. It is not a British Airways system, does not use British Airways operational data, and does not demonstrate performance on real airline disruptions.

## Project overview

The prototype explores whether affected passenger itineraries can be ranked by synthetic severe connection-disruption risk so that limited recovery attention can be prioritized after a disruption is identified. It compares a simple operational rule with logistic regression, Random Forest and gradient boosting (HistGradientBoosting is the primary model).

The synthetic target is `severe_connection_disruption = 1` when both conditions hold:

1. The original connection is missed.
2. No simulated recovery itinerary reaches the final destination within the same local calendar day as the originally scheduled final arrival.

This is the frozen synthetic specification's definition, not a claim about British Airways' internal definitions or procedures. See [`data/synthetic_spec_v03_frozen.md`](data/synthetic_spec_v03_frozen.md). The meaning of the same-day flag in the saved data is not fully resolved; see "Limitations and unresolved questions".

## Data and evaluation design

- **Disruption events:** 10,000.
- **Passenger itineraries:** 100,072, with 5–15 itineraries per event.
- **Split:** event-level 75/25 split using `random_state=42`; no event IDs are shared between training and test sets.
- **Test set:** 2,500 events and 25,119 passenger rows; 2,654 severe cases (about 10.6%).
- **Primary model family:** `HistGradientBoostingClassifier` using 10 operational/itinerary features.

The model features are weather severity, ATC restriction, airport congestion, inbound operational pressure, forecast delay, connection buffer, expected transfer time, expected alternative flights, expected recovery options, and number of connections. These are features in the synthetic prototype, not validated live operational signals.

The held-out test events were inspected during development. They are documented as held out from model fitting under the stated split, but they should not be described as an untouched final test set.

## Results snapshot

All figures are on the saved held-out test set (2,500 events, 25,119 itineraries, 2,654 severe cases). They are synthetic-prototype results, not estimates of performance in real airline operations.

**Test-set limitation:** the project author reports that these test events were inspected during development. The split is documented as held out from model fitting, but if test results informed development decisions the test set was not an untouched final evaluation set. These figures should therefore not be read as unbiased final estimates, and they say nothing about real-world performance.

### Capturing severe cases under limited capacity

Share of the 2,654 severe cases found in the top share of the ranked queue, as a point estimate with a 95% interval from an event-level bootstrap (2,000 resamples of the test events; conditional on the saved rankings, no refitting). Capacity is global: the top floor(25,119 × capacity) passengers are selected (1,255 / 2,511 / 5,023 / 7,535).

| Capacity | Random (expected) | Reconstructed simple-rule baseline | Logistic, balanced | Logistic, unweighted | HGB, saved refit queue |
|---|---:|---:|---:|---:|---:|
| 5% | 5.0% | 11.8% (10.6–13.1) | 17.4% (16.2–18.6) | 18.5% (17.2–19.8) | 19.4% (18.1–20.7) |
| 10% | 10.0% | 22.0% (20.6–23.7) | 30.1% (28.3–31.6) | 30.9% (29.3–32.5) | 32.7% (31.1–34.3) |
| 20% | 20.0% | 42.7% (40.8–44.6) | 51.1% (49.2–53.0) | 51.8% (50.0–53.8) | 53.9% (52.0–55.7) |
| 30% | 30.0% | 59.6% (57.8–61.7) | 69.1% (67.1–70.8) | 69.3% (67.4–71.2) | 69.5% (67.9–71.5) |

Number of severe cases captured at 10% capacity: HGB refit 868, logistic (unweighted) 819, logistic (balanced) 799, reconstructed simple-rule baseline 585; random selection expects about 265 (10% of 2,654; the original table's single random draw captured 264).

Paired differences, percentage points (point estimate; 95% interval):

| Capacity | HGB − reconstructed simple-rule baseline | HGB − logistic, unweighted |
|---|---:|---:|
| 5% | +7.6 (5.9 to 9.0) | +0.9 (−0.3 to 2.0) |
| 10% | +10.7 (8.6 to 12.5) | +1.8 (0.5 to 3.1) |
| 20% | +11.2 (9.0 to 13.3) | +2.1 (0.5 to 3.4) |
| 30% | +9.9 (8.0 to 12.0) | +0.2 (−0.8 to 1.6) |

HGB's advantage over the reconstructed simple-rule baseline is clear at every tested capacity (all four intervals exclude zero). Its advantage over unweighted logistic regression is comparatively small, and its interval includes zero at 5% and 30% capacity, so those two capacity levels do not establish a positive advantage. These intervals describe this synthetic test data and the saved rankings only.

### Discrimination on the test set

| Model | Test ROC-AUC | Test PR-AUC | Event-grouped 5-fold CV ROC-AUC |
|---|---:|---:|---:|
| HGB, saved refit queue | 0.8150 | 0.3218 | – |
| HGB, original v0.3 report | 0.8162 | 0.3257 | 0.8193 ± 0.0026 |
| Random Forest, original v0.3 report | 0.8149 | 0.3213 | 0.8167 ± 0.0027 |
| Logistic, balanced | 0.8074 | 0.2922 | 0.8103 ± 0.0032 |
| Logistic, unweighted | 0.8078 | 0.3011 | – |
| Reconstructed simple-rule baseline | 0.7612 | 0.2204 | – |

The `±` in the cross-validation column is the sample standard deviation (n − 1 denominator) of the metric across the 5 cross-validation folds. It is not a standard error. This was established by recomputing the standard deviation from the per-fold values in `results/ba_disruption_model_validation_v03__event_cv_fold_metrics.csv`, which matches `ROC_AUC_std` in `event_cv_summary.csv` to six decimal places; the original cross-validation code was not recovered, so the definition is inferred from that match.

Random Forest has no capture counts because its row-level predictions were not recovered.

### Original HGB report, saved refit, and seed refits

| | 5% | 10% | 20% | 30% |
|---|---:|---:|---:|---:|
| Original HGB report | 518 | 863 | 1,432 | 1,862 |
| Saved HGB refit queue | 516 | 868 | 1,431 | 1,845 |
| 12 seed refits, mean ± sd (range) | 514.0 ± 6.6 (503–524) | 863.9 ± 7.9 (847–877) | 1,422.5 ± 8.5 (1,407–1,433) | 1,851.2 ± 5.5 (1,841–1,861) |

The original HGB hyperparameters, seed and training source were not recovered, so the saved refit is not an exact reproduction of the originally reported model. These are different rankings: the original report's test ROC-AUC is 0.8162 and the saved refit's is 0.8150. The original capacity counts are consistent with the original model's own top-10% table (recall 0.32517 = 863 / 2,654 in `results/ba_disruption_modeling_v03__model_family_top10_v03.csv`).

Against the committed seed refits (scikit-learn 1.9.0), the original counts sit inside the seed range at 5% and 10%, at its upper edge at 20%, and above every one of the 12 seeds at 30% (1,862 vs a maximum of 1,861). One seed (seed 8) reaches the original test ROC-AUC of 0.8162 but captures 1,854 at 30%. When the same script and seeds are rerun with the export environment's library versions (scikit-learn 1.8.0, NumPy 2.3.5, pandas 2.2.3), the 12-seed range at 30% is 1,847–1,869, which includes 1,862, while the saved-queue results are unchanged. So the library version changes HGB refit counts. The original library version was not recorded (the export environment is documented as not a historical freeze), so the cause of the 1,862 difference is not established.

### Sources

- Original HGB, simple-rule and random counts: [`results/ba_disruption_decision_support_v03__prioritization_capacity_results.csv`](results/ba_disruption_decision_support_v03__prioritization_capacity_results.csv).
- Refit counts, and test ROC-AUC and PR-AUC for the refit, logistic and rule rows: [`followup_outputs/point_estimates_test.csv`](followup_outputs/point_estimates_test.csv).
- Percentages and intervals: [`followup_outputs/bootstrap_capture_ci.csv`](followup_outputs/bootstrap_capture_ci.csv); paired differences: [`followup_outputs/bootstrap_paired_differences.csv`](followup_outputs/bootstrap_paired_differences.csv).
- Seed refits: [`followup_outputs/hgb_seed_spread.csv`](followup_outputs/hgb_seed_spread.csv).
- Original HGB and Random Forest test AUCs: [`results/ba_disruption_modeling_v03__model_family_comparison_v03.csv`](results/ba_disruption_modeling_v03__model_family_comparison_v03.csv); cross-validation: [`results/ba_disruption_model_validation_v03__event_cv_summary.csv`](results/ba_disruption_model_validation_v03__event_cv_summary.csv).
- Refit provenance: [`results/ba_disruption_decision_support_v03__final_decision_support_readme_v03.md`](results/ba_disruption_decision_support_v03__final_decision_support_readme_v03.md).
- All capture counts use the floor rule; only the rank-based priority-band counts use rounded boundaries (for example 1,256 for the top 5% band).

## What can be inspected and rerun

### Saved artifacts that can be inspected

- Frozen synthetic datasets and validation summary in `data/`.
- Train/test event membership in `event_split/`.
- Saved prediction and queue artifacts in `predictions/` and `results/`.
- Saved follow-up tables in `followup_outputs/` and `results/followup_v03_results__*`.
- Source recovery notes in `source_status/`.

Inspecting a saved artifact is not the same as regenerating it from the original training pipeline.

### Rerun the follow-up analyses

Run commands from the root of the unzipped repository, after installing the dependencies. Use a separate output directory to avoid overwriting the committed evidence:

```bash
python -m pip install -r requirements.txt
python reproduce_followup_v03.py . rerun_outputs
python sensitivity_alt_target_v03.py . rerun_outputs
python Shuffle_check_v03.py . rerun_outputs
python check_connection_miss_rule.py .
```

The first three scripts accept `[package_dir] [out_dir]` arguments; `check_connection_miss_rule.py` accepts only `[package_dir]` and prints its results. All four are follow-up analysis code, not recovered original v0.3 generator or training scripts. The shuffle-check filename is case-sensitive on some systems; use `Shuffle_check_v03.py` exactly as it appears in the repository.

Verified behaviour: on 2026-10-15, on Linux with Python 3.13.16 and the pinned `requirements.txt` versions (installed into a fresh virtual environment), the four commands above, run exactly as written, all exited successfully. The three analysis scripts reproduced all eight tables in `followup_outputs/` exactly, `python -m pip check` reported no broken requirements, and the connection-miss check printed 316 / 0.3158. Reproducing the follow-up analyses is not the same as reproducing the original experiment.

Version sensitivity: with scikit-learn 1.8.0, NumPy 2.3.5 and pandas 2.2.3, the point estimates, bootstrap intervals, paired differences and per-event comparison were identical, but the 12-seed HGB refit table differed. Anything that refits HGB (seed refits, ablations, shuffled labels, retrained alternative-target models) can therefore vary with library version, and scikit-learn's HistGradientBoosting also uses a random validation split for early stopping above 10,000 rows. Not run: the documented commands on Windows or macOS, or on Python 3.13.7.

### Cannot be reproduced from this repository

The original data generator, the original model-training code, the exact original HGB and Random Forest settings, row-level predictions for the original HGB, Random Forest and simple rule, the original bootstrap code, calibration-training code, and oracle/latent probabilities. See [`source_status/SOURCE_CODE_RECOVERY_STATUS.md`](source_status/SOURCE_CODE_RECOVERY_STATUS.md).

## Limitations and unresolved questions

- **Synthetic data:** the results do not establish performance on real disruption data or production suitability.
- **Test-set inspection:** the project author reports that the held-out test events were inspected during development. They are documented as held out from training; this can be verified from code only for the follow-up refits, because the original training code was not recovered. This is not an untouched final evaluation.
- **Connection-miss rule not fully reconstructed:** a rule inferred from the data, `event_actual_delay_minutes + actual_transfer_minutes > connection_buffer_minutes`, disagrees with the stored `original_connection_missed` label on 316 of 100,072 rows (0.3158%). The rule is not in the specification or source notes, and it uses outcome columns the models never see. All 316 have a stored label of 0 where the rule predicts 1; none has a recovery value, and all have a severe label of 0; 75 are in the test rows. These are disagreements between an inferred rule and the stored label. They are not confirmed labelling errors, and this check does not establish which interpretation is correct, because the original generator was not recovered. The stored labels and the frozen target are unchanged, and this check does not show that the 75 test-row exceptions affect any model's severe-disruption predictions. The check is reproduced by `check_connection_miss_rule.py`.
- **Target semantics unresolved:** the saved follow-up notes report 97.044% agreement between the same-day flag and `best_recovery_arrival_minutes < 1440`. This was recomputed on the 39,378 missed itineraries that have a recovery value (97.6% over all 48,554 missed itineraries). A later script (`sensitivity_alt_target_v03.py`) finds that `original_final_arrival_minutes + best_recovery_arrival_minutes < 1440` matches the flag on all 39,378. The saved notes also say the two fields are not on a directly verifiable common clock. No document in this repository defines the units of either column, so the two readings are unreconciled, and neither should be silently substituted for the frozen target definition. The headline results use the stored `severe_connection_disruption` flag and do not depend on this. The 180-minute alternative-target analysis assumes the second reading and would be invalid if it were wrong. Resolving it needs the original generator source or a written definition of both columns.
- **Reconstructed simple-rule baseline:** the original simple-rule implementation was not recovered. The follow-up scripts use `forecast_delay_minutes + expected_transfer_minutes − connection_buffer_minutes` as the score. This formula is not documented in the specification or source notes. It reproduces the saved aggregate counts (314 / 585 / 1,134 / 1,582 at 5/10/20/30% capacity) exactly, but matching aggregate counts does not prove that the original formula has been recovered. Documented source facts, reconstructed logic and assumptions are kept separate throughout this README.
- **Missing original source:** the original synthetic data generator and original training code for Logistic Regression, Random Forest, and HGB were not recovered. Exact HGB and Random Forest hyperparameters, original calibration-training code, some row-level predictions, bootstrap implementation source, and oracle/latent probabilities are also unavailable. See [`source_status/SOURCE_CODE_RECOVERY_STATUS.md`](source_status/SOURCE_CODE_RECOVERY_STATUS.md).
- **Preprocessing not recovered:** the only documented logistic configuration is `class_weight="balanced"`; the follow-up scripts add a `StandardScaler`, and the original preprocessing is not documented.
- **Structural proxy and buffer dominance:** dropping `connection_buffer_minutes` lowers HGB test ROC-AUC from about 0.815 to 0.683 (`followup_outputs/ablations_hgb.csv`: 0.8146 with all features, 0.6834 without the buffer, 0.6731 without both the buffer and expected alternative flights). `expected_alternative_flights` is the mean of the simulator's recovery-alternatives draw. These are findings about the synthetic mechanism, not about real operations.
- **Leakage checks:** 40 shuffled-label fits give a mean test ROC-AUC of 0.505 (SE 0.007; `followup_outputs/shuffle_check_40.csv`). A smaller 5-run shuffled-label row in `ablations_hgb.csv` (0.455 ± 0.042) is not used; single runs vary because test passengers cluster in 2,500 events.
- **Simulation assumptions:** recovery availability is an influential assumption of the synthetic mechanism. The 5/10/20/30% capacity levels are illustrative, not staffing levels, and the results support no causal conclusions.
- **Alternative target and uncertainty:** only one alternative severity target (recovery delay over 180 minutes) was tested, without intervals. Bootstrap intervals are conditional on the saved rankings and do not include model refitting.
- **Capacity policy:** the reported prioritization uses a global passenger-capacity percentage; it does not establish how capacity should be allocated in live operations.
- **Operational use:** this prototype does not account for all real-world constraints, including seat contention, accessibility needs, passenger-specific circumstances, legal rights, fairness, or human decision-making requirements. It should not be used to make real passenger decisions.

## Repository map

```text
.
├── data/                 # Frozen synthetic datasets, specification, validation
├── event_split/          # Saved train/test event IDs
├── predictions/          # Saved test predictions and offline outcomes
├── results/              # Main v0.3 results, historical outputs, follow-up summaries
├── followup_outputs/     # Follow-up point estimates, sensitivity, bootstrap and checks
├── source_status/        # Source recovery status and documented fragments
├── environment/          # Environment snapshots and version records
├── reproduce_followup_v03.py
├── sensitivity_alt_target_v03.py
├── Shuffle_check_v03.py
├── check_connection_miss_rule.py
├── requirements.txt
├── MANIFEST.csv          # Checksums for the original v0.3 export only; not updated for later files
├── MANIFEST.json
└── README.md
```

Older v0.1/v0.2 outputs are retained in `results/` as historical artifacts; they should not be confused with the v0.3 results. The original v0.3 audit and explainability documents in `results/` do not include the follow-up analyses above (they contain no bootstrap intervals or seed refits), so their statements, such as "HGB is the best model tested", should be read together with the intervals in this README.

## Environment and dependencies

The current `requirements.txt` pins NumPy 2.5.2, pandas 3.0.5, and scikit-learn 1.9.0. The author confirms that the follow-up scripts were run on Python 3.13.7. On 2026-10-15 the author also confirmed, in the project's virtual environment, that Python is 3.13.7, that the installed NumPy, pandas and scikit-learn versions match these pins, and that `python -m pip check` reports `No broken requirements found.` No log of these runs or checks is stored in the repository; the later audit rerun used Python 3.13.16.

The saved export environment recorded in [`environment/relevant_versions.json`](environment/relevant_versions.json) is different: Python 3.13.5, NumPy 2.3.5, pandas 2.2.3, and scikit-learn 1.8.0 (with SciPy 1.17.0 and joblib 1.5.3). The `requirements.txt` file documents the current follow-up environment; it does not recreate the historical export or original v0.3 training environment.

Install the listed dependencies with:

```bash
python -m pip install -r requirements.txt
```

Check dependency consistency with:

```bash
python -m pip check
```

A successful dependency check verifies package consistency in that environment; it does not establish historical pipeline reproducibility. `requirements.txt` does not pin a Python version. The pinned packages were also installed into a fresh virtual environment on Python 3.13.16 on 2026-10-15, where `pip check` reported no broken requirements.

## Licence

No licence has been selected yet.