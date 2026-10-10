# BA Disruption Recovery Synthetic Prototype — v0.3 Frozen Specification

## Purpose
Educational synthetic prototype for prioritizing passenger itineraries at risk of severe connection disruption after a flight disruption is identified.

## Data hierarchy
- `disruption_events`: one row per disruption event.
- `passenger_itineraries`: one row per affected passenger itinerary.
- Multiple passenger rows share an `event_id`.

## Target
`severe_connection_disruption = 1` when:
1. the original connection is missed, and
2. no simulated recovery itinerary reaches the final destination within the same local calendar day as the originally scheduled final arrival.

This is a synthetic operational definition, not a claim about BA's internal target definition.

## Frozen recovery mechanism
For a missed connection:
- number of candidate alternatives ~ Poisson(expected alternative flights)
- waiting time = Uniform(25,150) + congestion/weather/ATC penalties
- flight duration ~ Normal(90,25), clipped to 45–180 minutes
- a recovery is same-day if arrival is before the end of the original calendar day
- availability multiplier = 1.0

## Anti-leakage rule
All target-generating outcomes remain hidden from model features. Splits are performed by `event_id`.

## Scaling
10,000 disruption events, 5–15 affected passenger itineraries per event.

## Modeling
Compare a simple operational rule against ML. Forage booking variables remain contextual features and are tested rather than assumed to be useful.

## Important limitation
This is not a real airline network simulator and must not be presented as BA operational data or production performance.
