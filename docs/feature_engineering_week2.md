# Week 2 Feature Engineering Design Specification

## 1. Time Semantics

The NASA C-MAPSS FD001 dataset represents turbofan engine degradation trajectories as ordered sequences of operating cycles. A single timestep corresponds to one recorded cycle for one engine unit. At each timestep, the available observation consists of the engine identifier, cycle index, operating condition variables, and contemporaneous sensor measurements recorded for that engine at that cycle.

Feature construction shall follow a strict causal constraint. For an engine observed at timestep `t`, every engineered feature must be computable using only observations from timesteps less than or equal to `t` for the same engine. No feature may use information from future timesteps, including `t+1` and onward, whether directly through future sensor values or indirectly through statistics computed over future observations.

Engine-wise independence is mandatory. Each engine trajectory shall be treated as an independent degradation sequence. Feature values for one engine must never depend on measurements, cycle positions, failure times, summary statistics, or derived features from any other engine.

## 2. Feature Categories

### Raw Sensor Features

Raw sensor features consist of the original sensor measurements recorded at the current timestep for the current engine. These features preserve the instantaneous observed state of the system and form the baseline representation for downstream predictive maintenance modeling.

Raw operating condition variables may be retained where relevant, provided they are observed at the same timestep and do not encode future trajectory information. Any use of operating settings shall follow the same causal and engine-wise constraints as sensor variables.

### First-Order Differences

First-order difference features describe short-term change by comparing the current sensor value with the immediately preceding value from the same engine sequence. Conceptually, for a sensor at timestep `t`, the difference represents the change between `t` and `t-1`.

These features are intended to capture local degradation dynamics, abrupt shifts, and transient behavior. The first available timestep for an engine has no preceding observation; its treatment shall be defined consistently during implementation without introducing information from later timesteps or from other engines.

### Causal Rolling Statistics

Causal rolling statistics summarize recent historical behavior within a fixed lookback window ending at the current timestep. The permitted rolling statistics are:

- Rolling mean
- Rolling standard deviation
- Rolling minimum
- Rolling maximum

For timestep `t`, a rolling statistic must use only observations from the current engine with cycle indices less than or equal to `t`. These features are intended to represent local operating level, variability, recent extrema, and short-horizon degradation patterns.

### Optional Degradation Trend Indicators

Optional degradation trend indicators may be introduced to characterize monotonic or persistent changes within an engine trajectory. Such indicators may describe whether a sensor has shown a recent increasing or decreasing tendency, whether variability is increasing, or whether local values are diverging from earlier same-engine behavior.

Any degradation trend indicator must be computed causally, reset independently for each engine, and must not use the engine's final failure cycle, remaining trajectory length, or future values.

### Optional Health Index Concept

A health index may be considered as a future composite representation of engine condition. At this design stage, the health index is a conceptual feature family only. Its computation is intentionally not specified in this document.

If introduced later, any health index must be causal, engine-local, reproducible, and validated for absence of leakage before use in model training or evaluation.

## 3. Windowing Rules

Only causal rolling windows are permitted. For a timestep `t`, the window may include the current timestep and a fixed number of prior timesteps from the same engine. Centered windows are prohibited because they include observations after `t` and therefore violate the causal prediction setting.

The initial candidate rolling window sizes for FD001 feature engineering are:

- 5 cycles, representing short-term local dynamics
- 10 cycles, representing intermediate local behavior
- 20 cycles, representing broader degradation context

These window sizes should be treated as design candidates subject to validation. Final selection should consider predictive performance, robustness, interpretability, and computational reproducibility.

Padding strategies must not introduce future leakage. Early-cycle windows with fewer than the nominal number of historical observations must be handled using only available past and current observations for that engine. No padding may be derived from future cycles, other engines, validation data, test data, or global trajectory summaries.

## 4. Engine-Level Constraints

All feature computations must reset at engine boundaries. The first timestep of an engine sequence must not inherit lagged values, rolling state, trend state, or summary values from the preceding engine in the dataset ordering.

No cross-engine aggregation is allowed at any stage of time-series feature generation. This prohibition includes global rolling baselines, population-level degradation profiles, cross-engine sensor means, shared trajectory statistics, and any feature that uses observations from multiple engine IDs to compute a value for an individual engine timestep.

The engine identifier shall be treated as a grouping boundary for temporal computations. It must not be treated as an ordinal variable implying continuity across engines.

## 5. Validation and Leakage Prevention Rules

Feature generation must be validated using conceptual leakage tests before model training. These tests should confirm that every feature at timestep `t` is a function only of observations from the same engine at timesteps less than or equal to `t`.

Recommended leakage prevention tests include:

- Temporal dependency audit: verify that lagged, differenced, rolling, and trend features never reference `t+1` or later timesteps.
- Engine boundary audit: verify that computed features reset when the engine ID changes and do not carry state across engine trajectories.
- Split integrity audit: verify that train, validation, and test partitions remain isolated after feature generation.
- Permutation sensitivity audit: verify that reordering engine groups does not change feature values within any individual engine trajectory.
- Truncation audit: verify that features computed on a truncated prefix of an engine sequence match the corresponding features computed for the same prefix within the full sequence.

Train, validation, and test split integrity must remain intact after feature generation. Feature engineering must not introduce statistics estimated from validation or test partitions into training features, nor may it allow training-time transformations to access validation or test trajectory information. Any fitted preprocessing components used after feature generation must be estimated on the training partition only and then applied to validation and test partitions without refitting.

Model evaluation shall remain consistent with the predictive maintenance objective and project guidelines. At minimum, downstream evaluation protocols must report F1 score and ROC-AUC, with splits defined so that leakage from future cycles or held-out engines cannot inflate performance estimates.

## 6. Strict DO NOT Section

The following actions are prohibited:

- Do not use future data from `t+1` onward to construct any feature for timestep `t`.
- Do not use centered rolling windows.
- Do not use global dataset statistics that leak information into time-series features.
- Do not shuffle sequences in a way that destroys temporal ordering.
- Do not share features, rolling state, lagged values, degradation indicators, or summary statistics across engine IDs.
- Do not compute engine-level features using final cycle information when such information would not be available at prediction time.
- Do not allow validation or test partition information to influence training feature construction.
- Do not introduce padding, imputation, smoothing, or normalization schemes that depend on future timesteps or held-out data.
