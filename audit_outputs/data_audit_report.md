# BA Disruption v0.3 Data Audit

- PASS: 21
- FAIL: 0
- Total checks: 21

| Status | Check | Detail |
|---|---|---|
| PASS | Event row count | 10000 |
| PASS | Passenger row count | 100072 |
| PASS | Event required columns | [] |
| PASS | Passenger required columns | [] |
| PASS | Event IDs unique | Unique=10000, rows=10000 |
| PASS | Passenger IDs unique | Unique=100072, rows=100072 |
| PASS | Event IDs non-null | 0 |
| PASS | Passenger IDs non-null | 0 |
| PASS | Passenger event references valid | Unknown event IDs=0 |
| PASS | Every event has passengers | Events without passengers=0 |
| PASS | 5–15 passengers per event | Min=5, max=15, violations=0 |
| PASS | weather_severity values in 0–3 | [np.int64(0), np.int64(1), np.int64(2), np.int64(3)] |
| PASS | atc_restriction values in 0–3 | [np.int64(0), np.int64(1), np.int64(2), np.int64(3)] |
| PASS | airport_congestion within [0, 1] | Missing/non-numeric=0, min=0.0171660693612796, max=0.9707059695167496 |
| PASS | inbound_operational_pressure within [0, 1] | Missing/non-numeric=0, min=0.008402767166955, max=0.9856724746045004 |
| PASS | Model features have no missing values | {} |
| PASS | Target matches frozen Boolean rule | Mismatched rows=0 |
| PASS | Training split has 7,500 events | 7500 |
| PASS | Test split has 2,500 events | 2500 |
| PASS | No train/test event overlap | Shared IDs=0 |
| PASS | Split covers all dataset events | Missing=0, unknown=0 |

## Missing values

```json
{
  "best_recovery_arrival_minutes": 60694
}
```

## Not verified

- Original generator implementation and random draw sequence
- Whether generation followed every intended rule
- Recovery clock semantics and the documented discrepancy
- Whether the original training code used only approved features

PASS means the saved data passed that specific check. It does not prove that the unavailable generator followed every rule.
