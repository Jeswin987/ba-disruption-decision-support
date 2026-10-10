# Final Decision-Support Queue — v0.3

## Purpose
Rank affected passenger itineraries by estimated severe connection-disruption risk so limited recovery attention can be prioritized.

## Model
HistGradientBoostingClassifier using only pre-outcome operational/itinerary variables:
weather_severity, atc_restriction, airport_congestion, inbound_operational_pressure, forecast_delay_minutes, connection_buffer_minutes, expected_transfer_minutes, expected_alternative_flights, expected_recovery_options, number_of_connections

## Data split
Event-level 75/25 split with `random_state=42`; no event appears in both train and test.

## Operational output
`priority_queue_test_v03.csv` contains risk scores, ranks, rank-based priority bands, and model-visible inputs. It intentionally excludes the eventual disruption outcome.

`priority_queue_top10pct_v03.csv` is the illustrative 10% capacity queue.

`event_priority_summary_v03.csv` aggregates the queue at event level without exposing the outcome.

## Evaluation separation
`priority_queue_evaluation_v03.csv` contains the eventual synthetic outcome for offline evaluation only. It is not an operational input.

## Test performance of this re-fit
- ROC-AUC: 0.8150
- PR-AUC: 0.3218
- Test rows: 25,119
- Test events: 2,500

The previously reported decision-support comparison remains the primary capacity comparison; this queue is the concrete operational artifact generated from the frozen feature policy and event-level split.

## Important limitation
The risk score is meaningful only under the synthetic-data assumptions. It is not a BA production score and should not be interpreted as a real BA probability.
