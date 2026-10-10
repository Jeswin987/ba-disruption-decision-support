# BA Disruption v0.3 Reproducibility Package

This package is a faithful export of the v0.3 artifacts available in the saved environment on 2026-10-05. It does not invent missing source code.

## Key facts
- 10,000 disruption events.
- 100,072 passenger itineraries.
- Event-level 75/25 train/test split, random_state=42.
- Test set: 2,500 events / 25,119 passenger rows.
- Primary operational queue: HistGradientBoostingClassifier using 10 operational/itinerary features.

## What can be reproduced exactly from this package
- The frozen datasets and validation checks.
- The exact train/test event membership.
- The saved HGB held-out test risk scores/ranks.
- All saved result tables.
- Saved Logistic follow-up aggregate analyses.

## What cannot be reproduced from source code alone
The original v0.3 generator, original model-training scripts/hyperparameters, and several row-level prediction artifacts are absent. See SOURCE_CODE_RECOVERY_STATUS.md.

## Environment note
`pip_freeze_current_environment.txt` is the exact environment snapshot of the current execution environment, not a historical freeze of the environment that originally generated v0.3. The historical environment version set was not saved with the artifacts.
