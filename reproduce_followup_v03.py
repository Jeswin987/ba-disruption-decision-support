"""
Reproducible follow-up analysis for the BA disruption v0.3 prototype.

Run from the root of the unzipped reproducibility package:
    pip install numpy pandas scikit-learn scipy
    python reproduce_followup_v03.py [package_dir] [out_dir]

Writes CSVs to out_dir (default ./followup_outputs). Everything here is computed
from the saved artifacts; nothing is copied from earlier chat transcripts.
Environment used for the checked run: Python 3.13, numpy 2.3, pandas 2.2,
scikit-learn 1.8.
"""
import sys, os, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")
PKG = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "followup_outputs"
os.makedirs(OUT, exist_ok=True)
CAPS = [0.05, 0.10, 0.20, 0.30]
N_BOOT = 2000
RNG = np.random.default_rng(20261005)

FEATURES = ["weather_severity", "atc_restriction", "airport_congestion",
            "inbound_operational_pressure", "forecast_delay_minutes",
            "connection_buffer_minutes", "expected_transfer_minutes",
            "expected_alternative_flights", "expected_recovery_options",
            "number_of_connections"]

# ------------------------------------------------------------------ load + checks
P = pd.read_csv(f"{PKG}/data/passenger_itineraries_v03_10000.csv")
test_ids = set(pd.read_csv(f"{PKG}/event_split/test_event_ids.csv").event_id)
train_ids = set(pd.read_csv(f"{PKG}/event_split/train_event_ids.csv").event_id)

ev = P["event_id"].drop_duplicates()
_, te_rebuilt = train_test_split(ev, test_size=0.25, random_state=42)
assert set(te_rebuilt) == test_ids, "event split does not reproduce"
assert not (test_ids & train_ids), "train/test event overlap"
assert len(P) == 100072 and P.event_id.nunique() == 10000
chk = ((P.original_connection_missed == 1) & (P.same_day_recovery_available == 0)).astype(int)
assert (chk == P.severe_connection_disruption).all(), "target definition mismatch"

tr = P[~P.event_id.isin(test_ids)].reset_index(drop=True)
te = P[P.event_id.isin(test_ids)].reset_index(drop=True)
y = te.severe_connection_disruption.values
n, tot = len(te), int(y.sum())
print(f"train rows {len(tr)}, test rows {n}, test severe {tot} ({tot/n:.4%})")

# ------------------------------------------------------------------ scores
q = pd.read_csv(f"{PKG}/results/ba_disruption_decision_support_v03__priority_queue_evaluation_v03.csv")
hgb_saved = te[["passenger_id"]].merge(q[["passenger_id", "risk_score"]], on="passenger_id", how="left")["risk_score"].values
assert not np.isnan(hgb_saved).any()

rule = (te.forecast_delay_minutes + te.expected_transfer_minutes - te.connection_buffer_minutes).values

lr_bal = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced")).fit(tr[FEATURES], tr.severe_connection_disruption)
lr_unw = make_pipeline(StandardScaler(), LogisticRegression()).fit(tr[FEATURES], tr.severe_connection_disruption)
scores = {
    "HGB_saved_queue": hgb_saved,
    "Simple_rule": rule,
    "Logistic_balanced": lr_bal.predict_proba(te[FEATURES])[:, 1],
    "Logistic_unweighted": lr_unw.predict_proba(te[FEATURES])[:, 1],
}
orders = {k: np.argsort(-v, kind="mergesort") for k, v in scores.items()}

def point_capture(order, cap):
    k = int(np.floor(n * cap)); return int(y[order[:k]].sum())

rows = []
for name, o in orders.items():
    rows.append({"strategy": name, "roc_auc": roc_auc_score(y, scores[name]),
                 "pr_auc": average_precision_score(y, scores[name]),
                 **{f"captured@{int(c*100)}%": point_capture(o, c) for c in CAPS}})
pd.DataFrame(rows).to_csv(f"{OUT}/point_estimates_test.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))

# ------------------------------------------------------------------ HGB seed spread
seed_rows = []
for s in range(12):
    m = HistGradientBoostingClassifier(random_state=s).fit(tr[FEATURES], tr.severe_connection_disruption)
    p = m.predict_proba(te[FEATURES])[:, 1]; o = np.argsort(-p, kind="mergesort")
    seed_rows.append({"seed": s, "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
                      **{f"captured@{int(c*100)}%": point_capture(o, c) for c in CAPS}})
pd.DataFrame(seed_rows).to_csv(f"{OUT}/hgb_seed_spread.csv", index=False)

# ------------------------------------------------------------------ event-level bootstrap (global capacity)
event_codes, event_index = np.unique(te.event_id.values, return_inverse=True)
n_ev = len(event_codes)
boot_w = np.empty((N_BOOT, n_ev), dtype=np.int32)
for b in range(N_BOOT):
    boot_w[b] = np.bincount(RNG.integers(0, n_ev, n_ev), minlength=n_ev)

def global_capture(order, w_row, k):
    """capture rate with global top-k on a weighted (resampled) population; fractional boundary."""
    w = w_row[order]; cw = np.cumsum(w); cy = np.cumsum(w * y[order])
    i = np.searchsorted(cw, k, side="left")            # first position where cumw >= k
    prev_w = cw[i - 1] if i > 0 else 0; prev_y = cy[i - 1] if i > 0 else 0
    return (prev_y + (k - prev_w) * y[order][i]) / (w_row * y).sum()

res = {s: {c: np.empty(N_BOOT) for c in CAPS} for s in list(scores) + ["Random_expected"]}
for b in range(N_BOOT):
    w_row = boot_w[b][event_index].astype(float)
    n_b, sev_b = w_row.sum(), (w_row * y).sum()
    for c in CAPS:
        k = np.floor(n_b * c)
        for name, o in orders.items():
            res[name][c][b] = global_capture(o, w_row, k)
        res["Random_expected"][c][b] = k / n_b          # a random pick of k passengers captures k/n of severe cases in expectation (recall, not prevalence)

summ = []
for name in res:
    for c in CAPS:
        a = res[name][c]
        summ.append({"strategy": name, "capacity": c, "mean": a.mean(),
                     "ci95_low": np.percentile(a, 2.5), "ci95_high": np.percentile(a, 97.5)})
pd.DataFrame(summ).to_csv(f"{OUT}/bootstrap_capture_ci.csv", index=False)

pairs = [("HGB_saved_queue", "Simple_rule"), ("HGB_saved_queue", "Logistic_balanced"),
         ("HGB_saved_queue", "Logistic_unweighted"), ("Logistic_unweighted", "Simple_rule")]
pd_rows = []
for a_, b_ in pairs:
    for c in CAPS:
        d = res[a_][c] - res[b_][c]
        pd_rows.append({"A": a_, "B": b_, "capacity": c, "mean_diff_pp": 100 * d.mean(),
                        "ci95_low_pp": 100 * np.percentile(d, 2.5), "ci95_high_pp": 100 * np.percentile(d, 97.5),
                        "share_of_resamples_A_better": (d > 0).mean()})
pd.DataFrame(pd_rows).to_csv(f"{OUT}/bootstrap_paired_differences.csv", index=False)

# ------------------------------------------------------------------ per-event ABSOLUTE capacity vs matched-budget global
rank_in_event = {}
for name, s in scores.items():
    df = pd.DataFrame({"e": event_index, "s": s, "i": np.arange(n)}).sort_values(["e", "s"], ascending=[True, False], kind="mergesort")
    df["r"] = df.groupby("e").cumcount() + 1
    rr = np.empty(n, dtype=int); rr[df.i.values] = df.r.values; rank_in_event[name] = rr

pe_rows = []
for m in [1, 2, 3]:
    for name in scores:
        sel = rank_in_event[name] <= m
        cap_pe = np.empty(N_BOOT); cap_gl = np.empty(N_BOOT); budget = np.empty(N_BOOT)
        for b in range(N_BOOT):
            w_row = boot_w[b][event_index].astype(float)
            bud = (w_row * sel).sum()
            cap_pe[b] = (w_row * sel * y).sum() / (w_row * y).sum()
            cap_gl[b] = global_capture(orders[name], w_row, bud)
            budget[b] = bud
        d = cap_gl - cap_pe
        pe_rows.append({"strategy": name, "per_event_max": m, "mean_budget_share": (budget / (boot_w[:, event_index].sum(1))).mean(),
                        "per_event_capture": cap_pe.mean(), "global_same_budget_capture": cap_gl.mean(),
                        "global_minus_per_event_pp": 100 * d.mean(),
                        "diff_ci95_low_pp": 100 * np.percentile(d, 2.5), "diff_ci95_high_pp": 100 * np.percentile(d, 97.5)})
pd.DataFrame(pe_rows).to_csv(f"{OUT}/per_event_absolute_vs_global.csv", index=False)

# ------------------------------------------------------------------ leakage / robustness ablations (HGB, 5 seeds each)
def run_hgb(feats, shuffle=False, seeds=range(5)):
    out = []
    for s in seeds:
        ytr = tr.severe_connection_disruption.values.copy()
        if shuffle:
            ytr = np.random.default_rng(s).permutation(ytr)
        m = HistGradientBoostingClassifier(random_state=s).fit(tr[feats], ytr)
        p = m.predict_proba(te[feats])[:, 1]; o = np.argsort(-p, kind="mergesort")
        out.append((roc_auc_score(y, p), average_precision_score(y, p), point_capture(o, .10) / tot))
    a = np.array(out); return a.mean(0), a.std(0)

abl = []
for label, feats, shuf in [
    ("all 10 features", FEATURES, False),
    ("drop expected_alternative_flights", [f for f in FEATURES if f != "expected_alternative_flights"], False),
    ("drop connection_buffer_minutes", [f for f in FEATURES if f != "connection_buffer_minutes"], False),
    ("drop both", [f for f in FEATURES if f not in ("expected_alternative_flights", "connection_buffer_minutes")], False),
    ("shuffled training labels (should be ~0.5 AUC)", FEATURES, True)]:
    mu, sd = run_hgb(feats, shuf)
    abl.append({"variant": label, "roc_auc": mu[0], "roc_auc_sd": sd[0], "pr_auc": mu[1], "pr_auc_sd": sd[1],
                "capture@10%": mu[2], "capture@10%_sd": sd[2]})
pd.DataFrame(abl).to_csv(f"{OUT}/ablations_hgb.csv", index=False)
print(pd.DataFrame(abl).round(4).to_string(index=False))
print("done ->", OUT)