# Explainability and Operational Interpretation — v0.3

## Purpose

This document explains the behavior of the selected HistGradientBoosting (HGB) model used to rank passenger itineraries for synthetic disruption-recovery prioritization.

The explanations describe **model behavior and empirical associations in the synthetic dataset**. They are not causal claims and do not represent British Airways' production logic.

## 1. Global feature importance

Permutation importance was measured as the reduction in ROC-AUC when each feature was shuffled. Larger values indicate that the fitted model relied more heavily on that feature for discrimination.

| Rank | Feature | ROC-AUC drop |
|---|---|---:|
| 1 | connection_buffer_minutes | 0.2266 |
| 2 | expected_alternative_flights | 0.0677 |
| 3 | forecast_delay_minutes | 0.0175 |
| 4 | expected_transfer_minutes | 0.0134 |
| 5 | airport_congestion | 0.0017 |
| 6 | weather_severity | 0.0014 |
| 7 | atc_restriction | 0.0012 |
| 8 | inbound_operational_pressure | 0.0006 |
| 9 | expected_recovery_options | 0.0003 |
| 10 | number_of_connections | approximately 0 |

### Interpretation

The model is dominated by **connection buffer**. This is operationally plausible because less time between the expected arrival of an affected flight and the onward connection leaves less tolerance for delay or transfer uncertainty.

The second strongest signal is **expected alternative flights**, which is also directionally sensible: fewer expected alternatives make recovery harder when a connection is missed.

Forecast delay and expected transfer time provide additional information about the amount of disruption pressure and transfer difficulty.

The relatively small permutation importance of weather, ATC restriction and congestion does **not** mean those factors are unimportant operationally. Their effects can be represented indirectly through other correlated variables, especially forecast delay and recovery availability.

## 2. Directional sanity checks

Severe-disruption rates were examined across empirical quartiles.

| Feature | Q1 severe rate | Q4 severe rate | Expected direction |
|---|---:|---:|---|
| connection_buffer_minutes | 19.20% | 0.41% | lower buffer → higher risk |
| forecast_delay_minutes | 6.54% | 14.91% | higher delay → higher risk |
| expected_alternative_flights | 19.27% | 4.51% | fewer alternatives → higher risk |
| airport_congestion | 6.69% | 14.84% | higher congestion → higher risk |
| inbound_operational_pressure | 9.61% | 11.00% | weak positive relationship |

The strongest relationships are directionally consistent with the intended operational mechanism.

## 3. What a high-priority passenger looks like

The final test-set queue shows clear differences between the prioritized groups and the bottom 80%.

| Priority band | Count | Mean risk score | Mean connection buffer (min) | Mean forecast delay (min) | Mean expected alternatives |
|---|---:|---:|---:|---:|---:|
| Critical (top 5%) | 1,256 | 0.403 | 67.1 | 62.4 | 1.02 |
| High (5–10%) | 1,256 | 0.284 | 64.3 | 56.0 | 1.37 |
| Medium (10–20%) | 2,512 | 0.224 | 63.2 | 50.2 | 1.57 |
| Monitor (bottom 80%) | 20,095 | 0.058 | 114.7 | 37.4 | 1.93 |

This supports an operational interpretation of the ranking: higher-priority itineraries generally have tighter connection buffers, greater forecast delay, and fewer expected recovery alternatives.

## 4. Example interpretation rule

For an individual queue entry, the model score should be interpreted as:

> **Estimated probability of the synthetic severe-connection-disruption outcome, given the pre-outcome operational features available to the model.**

A high score does **not** mean that the passenger will definitely miss a connection. It means the itinerary is comparatively high-risk under the synthetic model and therefore appears earlier in a constrained recovery-priority queue.

## 5. Important limitation: connection buffer dominance

Connection buffer is substantially more important than the other features in the permutation analysis. This is not something to hide or describe as proof that connection buffer is the universal most important factor in real airline operations.

The synthetic target explicitly depends on whether an original connection is missed and whether recovery can occur within the original local calendar day. Therefore, connection buffer is structurally close to the target-generation mechanism.

This means the result demonstrates that the prototype can learn a coherent operational relationship under the frozen synthetic assumptions, but it does **not** establish the same feature importance in real BA data.

## 6. What this model should and should not be used for

### Appropriate use in this project

- Rank affected itineraries when recovery attention is limited.
- Compare risk-based prioritization with a simple delay/buffer rule or random selection.
- Demonstrate how operational predictions can become a decision-support queue.
- Provide an auditable research prototype using synthetic data.

### Not supported by this project

- Automatic passenger rebooking.
- Real BA operational decisions.
- Claims about actual BA passenger behavior.
- Monetary savings or customer-service improvements.
- Causal claims such as “increasing connection buffer reduces disruption.”
- Production deployment or safety-critical decision automation.

## 7. Recommended presentation statement

> The HGB model concentrates higher-risk synthetic itineraries toward the top of a constrained recovery-priority queue. Its ranking is driven primarily by connection buffer, expected recovery availability and forecast delay. These relationships are operationally plausible and directionally consistent within the synthetic environment, but the feature importance and performance should not be interpreted as evidence about BA's proprietary production system or real passenger outcomes.
