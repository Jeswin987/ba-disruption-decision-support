"""
Check how well a rule inferred from the data matches the stored `original_connection_missed` label.

Rule (inferred from the data; not in the frozen specification):
    event_actual_delay_minutes + actual_transfer_minutes > connection_buffer_minutes

The rule uses outcome-side columns that the models never see. This script only reads the data and
reports disagreements; it does not change any stored label.

Run from the repository root (the folder that contains data/):
    python check_connection_miss_rule.py [package_dir]
"""
import sys
import pandas as pd

PKG = sys.argv[1] if len(sys.argv) > 1 else "."

P = pd.read_csv(f"{PKG}/data/passenger_itineraries_v03_10000.csv")
test_ids = set(pd.read_csv(f"{PKG}/event_split/test_event_ids.csv").event_id)

inputs = [
    "event_actual_delay_minutes",
    "actual_transfer_minutes",
    "connection_buffer_minutes",
]

print("Missing rule inputs:")
print(P[inputs].isna().sum())


rule = (P.event_actual_delay_minutes + P.actual_transfer_minutes > P.connection_buffer_minutes).astype(int)
bad = P.original_connection_missed != rule

print(bad.sum(), round(bad.mean() * 100, 4))
print(pd.crosstab(P.original_connection_missed[bad], rule[bad]))

e = P[bad]
print("rows:", len(P))
print("exceptions with a recovery value:", int(e.best_recovery_arrival_minutes.notna().sum()))
print("exceptions with severe_connection_disruption = 1:", int(e.severe_connection_disruption.sum()))
print("exceptions in test rows:", int(e.event_id.isin(test_ids).sum()))