# Week 2 Feature Engineering Implementation Plan

## 1. Feature Engineering Module Boundary

Week 2 feature engineering shall be treated as a separate pipeline module, conceptually located under `src/features/`. This module is responsible only for constructing leakage-safe engineered features from already prepared NASA C-MAPSS FD001 datasets.

The feature engineering module consumes prepared dataset outputs from Week 1. These inputs are assumed to include validated records, engine identifiers, ordered cycle indices, sensor values, existing train/validation/test membership, and previously assigned RUL labels.

The module must not modify raw data loading logic, schema validation logic, labeling logic, or splitting logic. In particular, it must not recompute RUL labels, alter split membership, merge splits, or redefine the experimental partitioning strategy.

The module outputs engineered datasets per split. The expected outputs are separate train, validation, and test feature datasets with consistent feature columns and preserved row membership relative to the Week 1 prepared inputs.

## 2. Stepwise Implementation Sequence

The Week 2 implementation shall follow the strict build order below. The order is designed to ensure that causal grouping, state management, and validation constraints are established before higher-risk derived features are introduced.

### A. Engine-Wise Grouping Wrapper

The first implementation step is an engine-wise grouping wrapper. This component establishes the engine as the unit of temporal continuity and ensures that observations are sorted by cycle within each engine.

This step must precede all feature construction because deltas, rolling windows, and trend features are undefined without a reliable engine-local temporal sequence. It also prevents accidental interpretation of adjacent rows from different engines as consecutive timesteps.

### B. EngineFeatureTransformer Skeleton

The second step is the `EngineFeatureTransformer` skeleton. This conceptual component provides the stateful design boundary for per-engine processing, including state reset, sequential timestep traversal, and feature row emission.

This step depends on the grouping wrapper because the transformer operates on one engine sequence at a time. It must be established before implementing stateful features so that later feature families share the same causal processing contract.

### C. Raw and Delta Feature Implementation

The third step is raw sensor pass-through and first-order delta feature construction. Raw sensor features are stateless and provide the baseline feature representation. Delta features introduce the first stateful dependency through the immediately preceding cycle of the same engine.

This step depends on the transformer skeleton because delta features require controlled access to previous-cycle state. Implementing raw and delta features before rolling and trend features allows the pipeline to validate basic temporal ordering and per-engine reset behavior with a minimal state footprint.

### D. Rolling Window Feature Implementation

The fourth step is causal rolling window feature construction. Rolling statistics include rolling mean, standard deviation, minimum, and maximum over predefined backward-looking windows.

This step depends on successful raw and delta feature construction because rolling features require more complex state management. Rolling windows must operate only over current and prior observations within the same engine and must reset at engine boundaries. The candidate window sizes are 5, 10, and 20 cycles.

### E. Trend Feature Implementation

The fifth step is degradation trend feature construction, represented conceptually by causal slope features over engine-local historical observations.

This step depends on the prior rolling-window stage because trend features introduce broader temporal summaries and therefore require mature state handling, correct cycle ordering, and established leakage controls. Trend features must remain causal and must not use final trajectory information unavailable at prediction time.

### F. Validation and Leakage Detection Hooks

The final step is validation and leakage detection hooks. These hooks are mandatory checks that inspect the generated features and transformation behavior for violations of temporal causality, engine isolation, and split isolation.

This step follows the implementation of all feature families because validation must cover the complete feature set. However, the validation requirements should be defined before implementation and applied incrementally as each feature family is added.

## 3. Split-Wise Processing Strategy

Train, validation, and test datasets must be processed independently. Feature computation for one split must not access observations, labels, state, summary statistics, or trajectory metadata from another split.

No cross-split feature computation is permitted. Rolling windows, delta features, trend features, and any derived temporal state must be computed within the split being processed and within each engine sequence in that split.

No shared statistics between splits are permitted for time-series feature construction. If any downstream preprocessing requires fitted statistics, those statistics must be estimated from the training split only and then applied without refitting to validation and test data.

Transformer behavior for validation and test data must be strictly transform-only. Validation and test processing must not update fitted state, revise feature definitions, infer global statistics, or alter the training-time transformation contract.

## 4. Final Feature Output Schema

Each engineered dataset shall preserve one row per engine-cycle observation. The output schema must include:

- `engine_id`: engine trajectory identifier.
- `cycle`: cycle index within the engine trajectory.
- Raw sensor features: current-cycle sensor values.
- Delta features: first-order differences relative to the previous cycle of the same engine.
- Rolling statistics features: causal rolling mean, standard deviation, minimum, and maximum features.
- Trend features: causal degradation slope or trend indicators.
- `RUL` label: target-only remaining useful life label from the Week 1 labeling pipeline.

The `RUL` label must be retained only as the supervised target. It must never participate in feature computation, temporal state updates, normalization decisions, or leakage validation shortcuts.

## 5. Validation Gates

The following validation gates are mandatory before the Week 2 feature engineering module can be considered methodologically acceptable.

### Temporal Causality Validation

Temporal causality validation verifies that each feature at cycle `t` depends only on observations from cycles less than or equal to `t` for the same engine. This gate must detect future leakage through centered windows, full-trajectory statistics, target-derived features, or any operation that references cycles after the current timestep.

### Engine Isolation Validation

Engine isolation validation verifies that all temporal state resets at engine boundaries. It must confirm that previous-cycle values, rolling windows, rolling extrema, trend estimates, and any cached state from one engine do not influence feature rows for another engine.

### Split Isolation Validation

Split isolation validation verifies that train, validation, and test processing remain independent. It must confirm that no feature values, fitted statistics, temporal states, or trajectory summaries are computed using data from more than one split. It must also confirm that validation and test transformations do not alter training-derived transformation state.

## 6. Strict Rules

The Week 2 feature engineering implementation plan is governed by the following strict rules:

- No future timestep usage is allowed.
- No centered windows are allowed.
- No cross-engine feature sharing is allowed.
- No global dataset leakage is allowed.
- No RUL-derived feature computation is allowed.
- No modification of Week 1 splitting, labeling, or raw loading logic is allowed.
- No sequence shuffling is allowed before temporal feature computation.

These rules preserve the validity of downstream predictive maintenance evaluation and ensure that reported F1 and ROC-AUC scores reflect deployable causal information rather than leakage artifacts.
