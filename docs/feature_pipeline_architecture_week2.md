# Week 2 Feature Pipeline Architecture

## 1. Pipeline Stages

The feature engineering pipeline for NASA C-MAPSS FD001 shall follow a fixed sequential architecture. The ordering is part of the research design and must be preserved to prevent leakage, maintain reproducibility, and support transparent evaluation.

The required pipeline stages are:

1. Raw validated labeled data
2. Engine-wise grouping
3. Train, validation, and test splits
4. Feature engineering stage
5. Final feature datasets

The starting point is raw validated labeled data. At this stage, the FD001 observations are assumed to have passed schema validation, ordering checks, sensor availability checks, and RUL label assignment from the earlier labeling pipeline. The label is treated as an existing supervised target and is not recomputed during feature engineering.

Engine-wise grouping is then applied to establish each engine as an independent temporal sequence. The engine identifier defines the unit of temporal continuity. Cycle order is meaningful only within an engine and must not be interpreted across engine boundaries.

Train, validation, and test splits are assumed to already exist before feature generation. These splits define the evaluation protocol and must remain unchanged throughout feature engineering. No feature engineering step may alter split membership, merge split contents, or create derived observations that mix split boundaries.

The feature engineering stage constructs causal features independently within each split and within each engine sequence. This stage produces raw sensor representations, first-order delta features, and causal rolling statistics.

The final feature datasets are split-preserving tabular datasets suitable for downstream modeling, explainability analysis, and evaluation. Each final dataset must retain the engine identifier, cycle index, engineered feature columns, and the previously assigned RUL label.

## 2. Feature Engineering Paradigm Decision

The selected paradigm is post-split feature engineering.

Under this design, train, validation, and test partitions are established before engineered features are computed. Feature generation is then performed separately for each split, with all temporal computations reset per engine and all fitted transformations estimated from the training split only.

Post-split feature engineering is recommended because it minimizes the risk that validation or test information can influence training representations. In a predictive maintenance setting, leakage may occur not only through explicit labels but also through future sensor trajectories, trajectory lengths, population statistics, and cross-engine summaries. Computing features after the split reduces the probability that held-out observations contribute to learned transformations, rolling state, or summary behavior used during training.

Pre-split feature engineering is considered riskier because feature values or preprocessing parameters may inadvertently encode information from validation or test engines before the evaluation partitions are isolated. Even when features appear local, implementation errors such as global normalization, dataset-level imputation, or rolling computations over an incorrectly ordered table can leak information across temporal or engine boundaries.

Therefore, the pipeline shall use post-split feature engineering as the default and required design for Week 2.

## 3. Feature Transformer Design

The conceptual feature engineering component is named `CausalFeatureTransformer`.

`CausalFeatureTransformer` is a design abstraction responsible for constructing deterministic, causal, engine-local feature representations from already split FD001 datasets. It is not defined as implementation code in this document.

The transformer shall follow the following conceptual behavior.

### Fit Behavior

The `fit` operation may access only the training split. Its purpose is limited to learning any transformation metadata that is strictly necessary for later feature construction or feature alignment. If any fitted preprocessing statistics are required, they must be estimated from the training split only.

The `fit` operation must not access validation or test observations, validation or test labels, validation or test engine identifiers, validation or test trajectory lengths, or any summaries derived from validation or test data.

### Transform Behavior

The `transform` operation shall be applied separately to each dataset split. During transformation, all temporal feature computations must be performed independently within each engine sequence. The transformer may apply training-derived metadata where appropriate, but it must not refit or update learned statistics using validation or test data.

For validation and test splits, transformation must use only the observations available within the split being transformed and only timesteps up to the current timestep for the same engine. Validation and test transformations must not modify the fitted state of the transformer.

### Leakage Constraints

`CausalFeatureTransformer` must prevent information leakage from validation and test data into training features or fitted transformation state. It must also prevent global statistics leakage across engines or splits. Any operation that requires a statistic over multiple engines or multiple splits is outside the permitted feature engineering design unless explicitly justified, fitted on training data only, and shown not to affect causal time-series features.

The transformer shall preserve split boundaries, engine boundaries, temporal order, and RUL labels as externally defined inputs.

## 4. Rolling Feature Computation Rules

Rolling feature computation must be engine-wise. Each engine sequence is processed independently, and rolling state must reset at the start of every engine.

Rolling features must be strictly causal. For a row corresponding to engine `i` at cycle `t`, the rolling computation may use only observations from engine `i` at cycles less than or equal to `t`. Observations from cycles greater than `t` are prohibited.

Centered windows are prohibited because they include future observations relative to the current timestep. All rolling windows must end at the current cycle and extend backward over the permitted historical window.

Cross-engine aggregation is prohibited. Rolling means, rolling standard deviations, rolling minima, rolling maxima, deltas, trend indicators, and any related temporal summaries must never use observations from other engine identifiers.

Boundary leakage across splits is prohibited. Rolling computations must not concatenate train, validation, and test partitions for convenience. A rolling window in one split must never include observations from another split, even if the same engine identifier appears in more than one partition under a particular experimental protocol.

## 5. Output Data Contract

Each final feature dataset shall preserve one row per engine-cycle observation. The row represents the information available for one engine at one cycle after causal feature construction.

The required row structure is:

- `engine_id`: unique identifier for the engine trajectory
- `cycle`: cycle index within the engine trajectory
- Raw sensor features: observed sensor measurements at the current cycle
- Delta features: first-order causal differences relative to the previous cycle within the same engine
- Rolling statistics features: causal rolling mean, standard deviation, minimum, and maximum features computed within the same engine
- `RUL` label: remaining useful life label produced by the earlier labeling pipeline

The output datasets must remain split-specific. There shall be separate final feature datasets for training, validation, and testing. Column definitions should be consistent across splits, while row membership must remain identical to the corresponding pre-feature-engineering split.

## 6. Strict Leakage Safety Section

The following leakage safety rules are mandatory:

- No feature may use future timesteps from `t+1` onward when constructing the representation for timestep `t`.
- No feature may transfer information across engine identifiers.
- No rolling, lagged, differenced, or trend computation may continue across engine boundaries.
- No feature engineering operation may concatenate train, validation, and test splits in a way that mixes temporal boundaries or split membership.
- No validation or test observations may influence fitted transformation state.
- No global dataset statistic may be used if it incorporates validation data, test data, future cycles, or cross-engine information that would not be available at prediction time.
- No sequence shuffling may be performed before temporal feature computation.
- No recomputation across splits may mix temporal context, rolling state, preprocessing state, or trajectory summaries across train, validation, and test partitions.

These constraints are required to preserve the scientific validity of downstream F1 and ROC-AUC evaluation. Any apparent performance improvement obtained by violating these rules shall be considered methodologically invalid.
