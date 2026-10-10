# Documented v0.3 code fragments (NOT recovered original source)

These snippets are transcribed from saved documentation/results and are included only to make documented decisions explicit. They are not claimed to be the original training scripts.

## Event-level split
```python
from sklearn.model_selection import train_test_split

unique_events = df["event_id"].drop_duplicates()
train_event_ids, test_event_ids = train_test_split(
    unique_events,
    test_size=0.25,
    random_state=42,
)
```

## Frozen Logistic follow-up configuration
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    class_weight="balanced",
)
```

The saved README states this refit uses the 10 frozen operational features and class_weight="balanced" because it exactly reproduces the saved Logistic CV metrics. It does NOT preserve any additional constructor hyperparameters that were not saved.

## Per-event capacity rule
```python
selected_k = max(1, int(len(event_rows) * capacity))
ranked = event_rows.sort_values("score", ascending=False, kind="mergesort")
selected = ranked.head(selected_k)
```

The stable descending sort and minimum-one rule are documented.
