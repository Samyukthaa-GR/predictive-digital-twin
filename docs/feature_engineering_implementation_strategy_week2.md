# Week 2 Feature Engineering Implementation Strategy

## 1. Data Representation Contract

The input to the feature engineering pipeline shall be represented as ordered engine-level sequences. Each sequence corresponds to one engine unit from the NASA C-MAPSS FD001 dataset and contains one row per observed cycle.

Each row must contain the following fields:

- `engine_id`: identifier of the engine unit.
- `cycle`: cycle index within the engine trajectory.
- Sensor values: contemporaneous sensor measurements observed at the given engine cycle.
- `RUL`: remaining useful life label from the prior labeling pipeline.

The `RUL` field is target-only. It is retained for supervised learning and downstream evaluation but must never be used to compute engineered features. No lagged, rolling, normalized, aggregated, or transformed feature may depend on `RUL`.

Within each engine, rows must be sorted by ascending cycle index before feature generation. Temporal order is part of the data contract; feature engineering is invalid if cycle ordering is ambiguous, missing, or altered by non-temporal operations.

The data representation must preserve engine boundaries. Cycles from different engines are not consecutive observations in a single time series, even if they appear adjacent in a tabular dataset.

## 2. Transformation Strategy

Feature transformation shall enforce per-engine sequential processing. Each engine trajectory is processed independently from its first observed cycle to its final observed cycle. The feature representation for a row at cycle `t` must be computed using only the current row and previously processed rows from the same engine.

The preferred conceptual strategy is streaming-style transformation. Under this paradigm, the transformer reads one engine sequence in temporal order, updates only that engine's local causal state, emits the feature row for the current cycle, and then proceeds to the next cycle. At no point may the transformation inspect future cycles in order to compute the current row's features.

Batch-level operations that mix timesteps are not permitted for temporal feature computation unless they are mathematically equivalent to causal sequential processing and are verified to preserve temporal ordering. In particular, operations that compute statistics over an entire trajectory, entire split, or entire dataset before deriving row-level features must be treated as leakage risks.

Causal computation constraints apply to all feature families. Raw sensor pass-through uses only the current observation. Difference features use only the current and immediately preceding observation from the same engine. Rolling statistics use only the current and prior observations within the defined causal window. Trend features use only historical observations available up to the current cycle.

## 3. Stateful vs Stateless Features

Feature types shall be classified according to whether they require temporal memory.

Stateless features include:

- Raw sensor values passed through from the current row.
- Current-cycle operating variables, if retained as observed inputs.
- Engine and cycle identifiers retained as metadata.

These features require no historical state because they are fully determined by the current row.

Stateful features include:

- First-order delta features.
- Causal rolling means.
- Causal rolling standard deviations.
- Causal rolling minima.
- Causal rolling maxima.
- Degradation slope or trend indicators.

These features require access to prior observations from the same engine and therefore require explicit state management.

State storage must be isolated per engine. The state associated with one engine may include recent sensor history, previous-cycle values, rolling-window contents, and trend summaries. This state must be initialized at the beginning of an engine sequence and discarded or reset before processing another engine.

No cross-engine state sharing is allowed. A previous value, rolling window, trend estimate, or any other state derived from one engine must never influence the feature values of another engine.

## 4. Feature Transformer System Design

The conceptual system component responsible for engine-level feature construction is named `EngineFeatureTransformer`.

`EngineFeatureTransformer` is a behavioral design abstraction. It defines how leakage-safe feature computation should occur, without prescribing implementation code or a specific software library.

The responsibilities of `EngineFeatureTransformer` are:

- Reset state per engine before processing the first cycle of that engine.
- Process timesteps sequentially in ascending cycle order.
- Compute raw, delta, rolling, and trend features using only current and previously observed same-engine data.
- Emit one feature row per input engine-cycle observation.
- Preserve the `engine_id`, `cycle`, and target-only `RUL` fields without allowing `RUL` to influence feature values.
- Prevent state persistence across engine boundaries.
- Preserve split boundaries when applied to train, validation, or test datasets.

For a given engine, the transformer shall conceptually maintain only the information that would have been available at the current cycle in a real deployment setting. The emitted feature row at cycle `t` must be reproducible from the prefix of the engine trajectory ending at `t`.

## 5. Leakage Failure Modes

Leakage failure modes are a critical risk in FD001 feature engineering because the dataset consists of complete historical degradation trajectories. The following failure modes must be explicitly avoided.

### Global Normalization Before Splitting

Computing normalization parameters over the full dataset before train, validation, and test splits allows held-out engines and future observations to influence training representations. This creates an optimistic evaluation bias and violates split integrity. Any normalization or scaling parameters must be treated as fitted preprocessing state and derived from the training split only.

### Rolling Window Leakage Due to Centered Windows

Centered rolling windows use observations before and after the current timestep. Because future observations are unavailable at prediction time, centered windows leak future sensor behavior into the current feature row. All rolling windows must be backward-looking and must end at the current cycle.

### Cross-Engine Aggregation Mistakes

Aggregating statistics across engines can transfer degradation information from one engine trajectory to another. Examples include global rolling baselines, population-level sensor means used as temporal features, and shared trend estimates across engine IDs. Such operations violate engine-wise independence and may encode information unavailable for an individual engine at prediction time.

### Shuffling Before Feature Engineering

Shuffling rows before feature engineering can destroy temporal order and cause lagged or rolling features to reference incorrect cycles. Temporal feature computation must occur only after sorting by cycle within each engine. Sequence order must be preserved throughout the transformation stage.

### Incorrect Rolling Operations Across Concatenated Data

Applying rolling computations to a table formed by concatenating multiple engines can accidentally allow rolling state to continue from the last cycle of one engine into the first cycle of another. This is a boundary leakage error. Rolling, lagged, and trend computations must be grouped by engine and reset at every engine boundary.

## 6. Strict Causality Enforcement Rules

The following rules define mandatory causality enforcement for the Week 2 feature engineering pipeline:

- No feature for cycle `t` may use observations from cycle `t+1` or later.
- No feature may transfer information across engine identifiers.
- All transformations must preserve temporal order within each engine.
- All stateful computations must reset at engine boundaries.
- Train, validation, and test splits must remain isolated during transformation.
- Target-only fields, including `RUL`, must never participate in feature computation.
- Batch operations are admissible only if they are equivalent to per-engine causal sequential processing and do not mix timesteps, engines, or splits.

These rules are required to ensure that the engineered feature set supports valid downstream F1 and ROC-AUC evaluation under a leakage-safe predictive maintenance methodology.
