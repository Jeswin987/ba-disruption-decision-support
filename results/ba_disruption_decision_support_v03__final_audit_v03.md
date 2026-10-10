# Final Project Audit — BA Disruption Recovery Decision-Support Prototype v0.3

## Audit purpose

This audit checks whether the current prototype's data assumptions, target definition, model evaluation, decision-support claims, and British Airways framing are defensible before the project is presented as a research/portfolio project.

## Overall verdict

**Status: Defensible with explicit caveats.**

The project has a coherent progression from a synthetic operational problem to an event-level machine-learning model and a constrained prioritization queue. The strongest evidence is the event-level cross-validation, held-out test evaluation, comparison against a simple delay/buffer rule and random selection, and the separation between operational queue outputs and offline outcome evaluation.

The project should **not** be presented as a reconstruction of BA's production system, a simulation of the real BA network, or evidence of actual BA passenger outcomes. Its defensible contribution is a **synthetic research prototype demonstrating how disruption-risk prediction can be converted into constrained recovery prioritization**.

---

## 1. Claim audit

| Claim | Status | Audit decision |
|---|---|---|
| BA uses AI/ML/optimization in operational disruption management | Supported by public BA material | Keep, with source citations in final report |
| BA considers customer connections in operational decision-making | Supported by public BA material | Keep |
| This prototype represents BA's actual operational data | Unsupported | Remove |
| The synthetic data represents real BA passenger distributions | Unsupported | Remove |
| The target is BA's formal definition of severe disruption | Unsupported | Call it the project's synthetic operational definition |
| HGB is the best model tested in this prototype | Supported | Keep; specify tested model families |
| HGB generalizes to unseen disruption events in this synthetic environment | Supported | Keep, because splitting is by event_id |
| The model predicts real BA disruption probability | Unsupported | Remove |
| Connection buffer is the most important feature in real airline operations | Unsupported | Remove |
| The model can reduce missed connections at BA by the measured percentage | Unsupported | Remove |
| The prototype can save BA money/customer-service cost | Unsupported | Remove |
| Risk ranking concentrates severe synthetic cases under limited capacity | Supported | Keep |
| 10% is BA's realistic recovery capacity | Unsupported | Do not claim; label as an illustrative capacity scenario |

---

## 2. Data and target audit

### Strengths

- The data hierarchy is explicit: disruption events contain shared event conditions, while passenger itineraries are nested within events.
- Multiple passengers sharing an event are handled by splitting on `event_id`, which reduces the major leakage risk of putting passengers from the same disruption into both train and test sets.
- The severe-disruption target is defined explicitly rather than being a vague proxy.
- Target-generating outcomes are excluded from model-visible inputs.
- The project acknowledges that the target definition is synthetic.

### Remaining limitation

The target is structurally close to `connection_buffer_minutes`. The synthetic outcome first depends on whether the original connection is missed, so a strong relationship between connection buffer and the target is expected. This explains the very large permutation importance of connection buffer and should be discussed prominently.

### Required wording

Use:

> "Under the frozen synthetic mechanism, severe connection disruption is defined as a missed original connection followed by no feasible recovery reaching the final destination within the same local calendar day as the originally scheduled final arrival."

Do not use:

> "BA defines a severe connection disruption as..."

---

## 3. Synthetic-data realism audit

### What is reasonable

The prototype includes operationally plausible variables:

- weather severity
- ATC restriction
- airport congestion
- inbound operational pressure
- forecast delay
- connection buffer
- expected transfer time
- expected alternative flights/recovery options
- number of connections

The recovery mechanism also introduces uncertainty instead of making the target a deterministic delay-minus-buffer rule.

### What is not established

The numerical distributions are **simulation assumptions**, not estimates of BA's real distributions. In particular:

- disruption severity probabilities
- forecast error
- transfer-time uncertainty
- alternative-flight availability
- recovery waiting times
- flight-duration distribution
- recovery availability multiplier

must not be described as measured BA parameters.

---

## 4. Sensitivity-analysis audit

### Important correction

The available sensitivity summary is labelled around the earlier **v0.2 mechanism** (`baseline_v02`) rather than the frozen scaled v0.3 dataset.

Therefore the project should **not claim that the sensitivity analysis fully validates the final v0.3 synthetic mechanism**.

The sensitivity analysis remains useful as evidence that the earlier mechanism was stress-tested, particularly around forecast error, recovery availability and severity composition. But it should be described accurately as a **mechanism-development sensitivity analysis**, not as a complete sensitivity analysis of the final v0.3 model.

### Strongest sensitivity finding

Recovery availability materially changes the severe-event rate and model difficulty. This is a major modeling assumption and should be highlighted as a limitation.

### Recommendation

Do not regenerate a large sensitivity sweep unless needed for the final research question. If additional work is desired, one focused v0.3 sensitivity check around recovery availability would be more valuable than adding more model families or technologies.

---

## 5. Model-selection audit

The model-selection sequence is defensible:

1. simple delay/buffer rule
2. Logistic regression baseline
3. Random Forest
4. HistGradientBoosting
5. event-level cross-validation
6. calibration comparison
7. held-out event-level test evaluation
8. constrained prioritization comparison

Five-fold event-level CV gives approximately:

| Model | ROC-AUC mean | PR-AUC mean | Brier mean |
|---|---:|---:|---:|
| HistGradientBoosting | 0.8193 | 0.3213 | 0.0801 |
| Random Forest | 0.8167 | 0.3170 | 0.1671 |
| Logistic | 0.8103 | 0.2887 | 0.1875 |

HGB is therefore a reasonable primary model for this prototype.

### Calibration decision

Raw HGB was already well calibrated in cross-fitted evaluation. Adding isotonic or sigmoid calibration did not provide a compelling improvement. Keeping raw HGB probabilities is therefore defensible and avoids unnecessary complexity.

---

## 6. Decision-support audit

This is the strongest part of the project.

At the illustrative 10% prioritization capacity on the fixed test set:

- HGB recall: about 32.5%
- simple rule recall: about 22.0%
- random selection recall: about 9.9%

The HGB recall advantage over the simple rule is about **10.5 percentage points**, and over random selection about **22.6 percentage points**.

These results support the claim that the model can **concentrate severe synthetic cases toward the top of a constrained queue**.

They do not support claims about actual operational savings, passenger satisfaction, or real-world missed-connection reduction.

### Important operational limitation

The current capacity simulation applies a **global passenger-level capacity percentage**. A real operation would likely have capacity constraints that depend on event, airport, time, agent workload, recovery options and other operational factors.

Therefore the 5%, 10%, 20% and 30% scenarios should be called **illustrative prioritization capacities**, not realistic BA staffing levels.

---

## 7. Queue-design audit

The operational queue is appropriately separated from the evaluation outcome:

- `priority_queue_test_v03.csv` contains model-visible inputs and ranking information.
- `priority_queue_evaluation_v03.csv` contains the eventual synthetic outcome for offline evaluation.

This is good practice.

The priority bands are rank-based (`top 5%`, `5–10%`, etc.), which is appropriate for a capacity-constrained prototype. They should not be described as calibrated probability categories such as "risk > 0.7 = critical".

### Further limitation

The current queue ranks passengers globally. A stronger future operational design could rank within each active disruption event or operational work queue. That is a **future refinement**, not a reason to invalidate the current prototype.

---

## 8. Explainability audit

The current explanation is appropriately framed as model behavior rather than causality.

The strongest evidence is:

- connection buffer: ROC-AUC permutation drop 0.2266
- expected alternative flights: 0.0677
- forecast delay: 0.0175
- expected transfer time: 0.0134

The directional checks are also consistent with the intended synthetic mechanism.

However, the dominance of connection buffer should be presented as a **synthetic-mechanism finding**, not a discovery about BA's real operational system.

---

## 9. Forage dataset audit

The original Forage booking dataset should remain contextual rather than being forced into the core model.

The experiments showed that adding the selected Forage context did not improve the core operational model and sometimes reduced performance. This is a positive result: the project tested whether the inherited dataset was useful for the new problem instead of assuming that every original feature had to remain.

The final narrative should explain that the original booking dataset inspired the starting point, but the redesigned operational problem required a different data structure and target.

Do not claim that the Forage variables are operational disruption predictors merely because they were present in the original assignment.

---

## 10. Technology-scope audit

No additional LLM, RAG, vector database, agent framework, or deployment platform is required to make the research argument credible.

Adding those technologies would increase implementation complexity without addressing the central research question.

A simple reproducible Python pipeline plus CSV/Markdown outputs is sufficient for the current project.

---

## 11. Final limitations that must appear in the report

1. The dataset is synthetic.
2. The synthetic distributions are assumptions, not BA measurements.
3. The model is not trained on BA operational data.
4. The target is a project-specific synthetic operational definition.
5. The project does not reproduce BA's real timetable, network, crew system, passenger inventory or recovery engine.
6. Recovery availability is a particularly influential simulation assumption.
7. Connection buffer is structurally close to the synthetic target mechanism.
8. Performance estimates describe this synthetic environment and should not be transferred to real BA operations.
9. The capacity scenarios are illustrative rather than observed staffing constraints.
10. The prioritization queue is decision support, not automatic rebooking or autonomous operational control.
11. No causal conclusions can be drawn from the feature associations.
12. No monetary, customer-satisfaction or service-level impact has been demonstrated.

---

## 12. Final recommended project claim

> **This project develops an educational synthetic decision-support prototype that ranks affected passenger itineraries by estimated severe connection-disruption risk under constrained recovery capacity. Using event-level validation and a held-out event split, a HistGradientBoosting model outperformed a simple delay/buffer rule and random prioritization in concentrating severe synthetic cases toward the top of the queue. The results demonstrate the decision-support workflow under the stated synthetic assumptions, but do not represent British Airways production performance, passenger behavior, or operational outcomes.**

## 13. Final recommendation

**Proceed to the final report and presentation.**

Do not expand the model stack. Do not add an LLM/RAG layer merely for novelty. Do not make stronger BA-specific claims.

The remaining work is primarily communication: document the methodology, show the evidence, explain the limitations, and demonstrate one end-to-end prioritization example.
