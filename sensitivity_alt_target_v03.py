"""
Alternative-target sensitivity for the v0.3 prototype.

Finding that makes this possible (verified on all 39,378 missed itineraries with a recovery):
    same_day_recovery_available == (original_final_arrival_minutes + best_recovery_arrival_minutes < 1440)
so best_recovery_arrival_minutes behaves as minutes added to the original scheduled final arrival,
and the recovery arrival delay is best_recovery_arrival_minutes.

Alternative target (no midnight cliff):
    severe_alt = missed AND (no recovery itinerary OR recovery delay > 180 minutes)

Run from the unzipped package folder:
    python sensitivity_alt_target_v03.py [package_dir] [out_dir]
"""
import sys, os, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")
PKG = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "followup_outputs"
os.makedirs(OUT, exist_ok=True)
CAPS = [0.05, 0.10, 0.20, 0.30]
THRESH = 180
FEATURES = ["weather_severity", "atc_restriction", "airport_congestion", "inbound_operational_pressure",
            "forecast_delay_minutes", "connection_buffer_minutes", "expected_transfer_minutes",
            "expected_alternative_flights", "expected_recovery_options", "number_of_connections"]

P = pd.read_csv(f"{PKG}/data/passenger_itineraries_v03_10000.csv")
test_ids = set(pd.read_csv(f"{PKG}/event_split/test_event_ids.csv").event_id)

# --- identity check
m = P[(P.original_connection_missed == 1) & P.best_recovery_arrival_minutes.notna()]
ident = ((m.original_final_arrival_minutes + m.best_recovery_arrival_minutes < 1440).astype(int) == m.same_day_recovery_available).mean()
print(f"same-day flag identity holds on {ident:.4%} of {len(m)} missed itineraries with a recovery")
assert ident == 1.0

P["severe_alt"] = ((P.original_connection_missed == 1) &
                   (P.best_recovery_arrival_minutes.isna() | (P.best_recovery_arrival_minutes > THRESH))).astype(int)
tr = P[~P.event_id.isin(test_ids)].reset_index(drop=True)
te = P[P.event_id.isin(test_ids)].reset_index(drop=True)
print(f"prevalence original target: all {P.severe_connection_disruption.mean():.4f}, test {te.severe_connection_disruption.mean():.4f}")
print(f"prevalence alt target (>{THRESH} min): all {P.severe_alt.mean():.4f}, test {te.severe_alt.mean():.4f}")
print("overlap: P(alt=1 | orig=1) = %.3f ; P(orig=1 | alt=1) = %.3f" % (
    te.loc[te.severe_connection_disruption == 1, "severe_alt"].mean(),
    te.loc[te.severe_alt == 1, "severe_connection_disruption"].mean()))

q = pd.read_csv(f"{PKG}/results/ba_disruption_decision_support_v03__priority_queue_evaluation_v03.csv")
hgb_saved = te[["passenger_id"]].merge(q[["passenger_id", "risk_score"]], on="passenger_id")["risk_score"].values
rule = (te.forecast_delay_minutes + te.expected_transfer_minutes - te.connection_buffer_minutes).values
lr = make_pipeline(StandardScaler(), LogisticRegression()).fit(tr[FEATURES], tr.severe_connection_disruption)
lr_orig = lr.predict_proba(te[FEATURES])[:, 1]

def cap(score, y, c, strict=True):
    o = np.argsort(-score, kind="mergesort"); k = int(np.floor(len(y) * c)); return y[o[:k]].sum() / y.sum()

rows = []
for target in ["severe_connection_disruption", "severe_alt"]:
    y = te[target].values
    for name, s in [("Random(expected)", None), ("Simple_rule", rule), ("Logistic_trained_on_original", lr_orig),
                    ("HGB_saved_queue(trained_on_original)", hgb_saved)]:
        r = {"label": target, "strategy": name,
             "roc_auc": np.nan if s is None else roc_auc_score(y, s),
             "pr_auc": np.nan if s is None else average_precision_score(y, s)}
        for c in CAPS:
            r[f"capture@{int(c*100)}%"] = c if s is None else cap(s, y, c)   # random pick captures a share c of severe cases in expectation
        rows.append(r)

# retrain on the alternative target (HGB, 5 seeds; logistic) and evaluate on the alternative label
y_alt = te.severe_alt.values
for name in ["HGB_retrained_on_alt(5 seeds)", "Logistic_retrained_on_alt"]:
    res = []
    for s in range(5 if name.startswith("HGB") else 1):
        mdl = (HistGradientBoostingClassifier(random_state=s) if name.startswith("HGB")
               else make_pipeline(StandardScaler(), LogisticRegression()))
        p = mdl.fit(tr[FEATURES], tr.severe_alt).predict_proba(te[FEATURES])[:, 1]
        res.append([roc_auc_score(y_alt, p), average_precision_score(y_alt, p)] + [cap(p, y_alt, c) for c in CAPS])
    a = np.mean(res, 0)
    rows.append({"label": "severe_alt", "strategy": name, "roc_auc": a[0], "pr_auc": a[1],
                 **{f"capture@{int(c*100)}%": a[2 + i] for i, c in enumerate(CAPS)}})

out = pd.DataFrame(rows)
out.to_csv(f"{OUT}/sensitivity_alt_target.csv", index=False)
print(out.round(4).to_string(index=False))