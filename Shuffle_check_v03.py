"""
Shuffled-label leakage check for the v0.3 prototype (40 independent shuffles).

Trains default HistGradientBoosting on the TRAINING events with the labels randomly permuted,
then scores the held-out test events. With no leakage the test ROC-AUC should centre on 0.5.
Individual runs vary a lot (about +/-0.05) because test passengers cluster in only 2,500 events,
so report the mean and standard error, not a single run.

Run from the unzipped package folder:
    python shuffle_check_v03.py [package_dir] [out_dir]
"""
import sys, os, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
PKG = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "followup_outputs"
os.makedirs(OUT, exist_ok=True)
N_SHUFFLES = 40
FEATURES = ["weather_severity", "atc_restriction", "airport_congestion", "inbound_operational_pressure",
            "forecast_delay_minutes", "connection_buffer_minutes", "expected_transfer_minutes",
            "expected_alternative_flights", "expected_recovery_options", "number_of_connections"]

P = pd.read_csv(f"{PKG}/data/passenger_itineraries_v03_10000.csv")
test_ids = set(pd.read_csv(f"{PKG}/event_split/test_event_ids.csv").event_id)
tr = P[~P.event_id.isin(test_ids)]
te = P[P.event_id.isin(test_ids)]
y = te.severe_connection_disruption.values

rows = []
for s in range(N_SHUFFLES):
    y_shuf = np.random.default_rng(1000 + s).permutation(tr.severe_connection_disruption.values)
    p = HistGradientBoostingClassifier(random_state=s).fit(tr[FEATURES], y_shuf).predict_proba(te[FEATURES])[:, 1]
    rows.append({"shuffle": s, "test_roc_auc": roc_auc_score(y, p)})
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/shuffle_check_40.csv", index=False)
a = df.test_roc_auc.values
print(f"{N_SHUFFLES} shuffles: mean {a.mean():.4f}, sd {a.std(ddof=1):.4f}, "
      f"se {a.std(ddof=1)/np.sqrt(len(a)):.4f}, min {a.min():.3f}, max {a.max():.3f}")