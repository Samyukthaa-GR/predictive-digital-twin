# Week 2 Feature Set Mathematical Definition

## 1. Raw Sensor Vector

Let \(e \in \mathcal{E}\) denote an engine unit in the NASA C-MAPSS FD001 dataset, where \(\mathcal{E}\) is the set of observed engines. Let \(t \in \{1, 2, \ldots, T_e\}\) denote the cycle index for engine \(e\), where \(T_e\) is the final observed cycle for that engine.

At cycle \(t\), the raw sensor vector for engine \(e\) is defined as:

\[
\mathbf{S}_e(t) =
\left[
s_{e,1}(t),
s_{e,2}(t),
\ldots,
s_{e,m}(t)
\right]^\top
\]

where \(m\) is the number of sensor channels and \(s_{e,i}(t)\) denotes the observed value of sensor \(i\) for engine \(e\) at cycle \(t\).

When the engine identifier is unambiguous, the notation may be shortened to:

\[
\mathbf{S}_t =
\left[
s_1(t),
s_2(t),
\ldots,
s_m(t)
\right]^\top
\]

The raw sensor vector represents the instantaneous measured operating state of one engine at one cycle. It contains no temporal aggregation and must be interpreted only within the engine trajectory to which it belongs.

## 2. First-Order Difference Features

For each sensor \(i \in \{1, 2, \ldots, m\}\), the first-order difference feature at cycle \(t\) is defined as:

\[
\Delta s_{e,i}(t) = s_{e,i}(t) - s_{e,i}(t-1)
\]

for \(t > 1\). When the engine identifier is implicit:

\[
\Delta s_i(t) = s_i(t) - s_i(t-1)
\]

The first-order difference represents the local change in a sensor channel between two consecutive cycles of the same engine. In the predictive maintenance context, \(\Delta s_i(t)\) may be interpreted as an instantaneous degradation velocity proxy, capturing short-horizon sensor drift, abrupt operational changes, or early signs of accelerated wear.

The quantity is undefined at the first observed cycle of each engine unless a separate causal convention is specified. No value for \(\Delta s_{e,i}(1)\) may be derived from future cycles or from other engines.

## 3. Causal Rolling Statistics

Let \(k \in \{5, 10, 20\}\) denote a causal rolling window size measured in cycles. For engine \(e\), sensor \(i\), and current cycle \(t\), define the available causal window as:

\[
\mathcal{W}_{e,k}(t) =
\{\tau : \max(1, t-k+1) \leq \tau \leq t\}
\]

This window contains only the current and preceding cycles for the same engine. It is strictly causal because it excludes all cycles \(\tau > t\).

### Rolling Mean

The causal rolling mean for sensor \(i\) is defined as:

\[
\mu_{e,i}^{(k)}(t) =
\frac{1}{|\mathcal{W}_{e,k}(t)|}
\sum_{\tau \in \mathcal{W}_{e,k}(t)}
s_{e,i}(\tau)
\]

The rolling mean estimates the recent local operating level of a sensor within the causal window.

### Rolling Standard Deviation

The causal rolling standard deviation for sensor \(i\) is defined as:

\[
\sigma_{e,i}^{(k)}(t) =
\sqrt{
\frac{1}{|\mathcal{W}_{e,k}(t)|}
\sum_{\tau \in \mathcal{W}_{e,k}(t)}
\left(s_{e,i}(\tau) - \mu_{e,i}^{(k)}(t)\right)^2
}
\]

The rolling standard deviation characterizes recent variability in the sensor signal. It may reflect unstable operating behavior, increased noise, or degradation-related fluctuation.

### Rolling Minimum

The causal rolling minimum for sensor \(i\) is defined as:

\[
\operatorname{min}_{e,i}^{(k)}(t) =
\min_{\tau \in \mathcal{W}_{e,k}(t)}
s_{e,i}(\tau)
\]

The rolling minimum represents the lowest observed sensor value within the causal lookback window.

### Rolling Maximum

The causal rolling maximum for sensor \(i\) is defined as:

\[
\operatorname{max}_{e,i}^{(k)}(t) =
\max_{\tau \in \mathcal{W}_{e,k}(t)}
s_{e,i}(\tau)
\]

The rolling maximum represents the highest observed sensor value within the causal lookback window.

For every \(k \in \{5, 10, 20\}\), rolling statistics must be computed independently for each engine and each sensor. No centered window, future observation, or cross-engine value is permitted.

## 4. Degradation Trend Features

For each engine \(e\) and sensor \(i\), a causal degradation trend feature may be defined as the slope of sensor values with respect to cycle index over a causal historical interval. Let \(\mathcal{T}_e(t)\) denote a causal set of cycle indices satisfying:

\[
\mathcal{T}_e(t) \subseteq \{1, 2, \ldots, t\}
\]

A trend slope for sensor \(i\) at cycle \(t\) may be represented conceptually as:

\[
\beta_{e,i}(t)
=
\operatorname{slope}
\left(
\{(\tau, s_{e,i}(\tau)) : \tau \in \mathcal{T}_e(t)\}
\right)
\]

where \(\beta_{e,i}(t)\) denotes the estimated direction and magnitude of the sensor's temporal change using only observations from the same engine up to cycle \(t\).

A positive or negative slope may indicate persistent sensor drift associated with long-term wear, depending on the physical interpretation of the corresponding sensor channel. These features are intended to capture degradation patterns that are broader than one-cycle differences and more directional than rolling variability measures.

The trend definition is theoretical in this phase. The exact estimation procedure, trend interval, robustness criterion, and treatment of early cycles are not specified here.

## 5. Optional Health Index Concept

An optional health index may be introduced as a conceptual aggregate representation of engine condition. Let \(\tilde{s}_{e,i}(t)\) denote a normalized form of sensor \(i\) for engine \(e\) at cycle \(t\). A generic health index can be expressed as:

\[
HI_e(t) =
\sum_{i=1}^{m} w_i \tilde{s}_{e,i}(t)
\]

where \(w_i\) denotes the weight assigned to sensor \(i\).

In this Week 2 phase, the health index is conceptual only. The weights \(w_i\) are not trained, optimized, or estimated in this phase. The normalization procedure for \(\tilde{s}_{e,i}(t)\), the admissible sensor set, and the interpretation scale of \(HI_e(t)\) remain undefined.

Any future definition of \(HI_e(t)\) must preserve causal computation, engine-wise independence, and split integrity. It must not incorporate future sensor values, final failure-cycle information, or validation/test statistics during training.

## 6. Feature Stability and Normalization Notes

Raw sensor features may require scaling because different sensor channels can have different physical units, magnitudes, and variance ranges. Any such scaling must be defined as a downstream preprocessing decision and must preserve train-only fitting when applied in a supervised learning pipeline.

First-order difference features are potentially noise-sensitive because they amplify cycle-to-cycle fluctuations. Sensors with high measurement variance may produce unstable \(\Delta s_i(t)\) values, especially in early degradation periods where signal changes are subtle.

Rolling standard deviation features are also noise-sensitive because they explicitly quantify local variability. Short windows, particularly \(k=5\), may be more responsive to transient fluctuations but less stable than longer windows.

Rolling mean, rolling minimum, and rolling maximum features may require scaling when combined with raw sensor values or difference features. Rolling extrema may be sensitive to isolated outlying observations within the causal window.

Trend slope features may require scaling because their magnitude depends on both sensor units and the temporal interval over which the slope is defined. They may also be sensitive to nonstationary operating regimes and short-term disturbances.

The optional health index concept necessarily depends on normalized sensor inputs. However, no normalization procedure or weighting scheme is proposed in this phase.

## 7. Strict Causality Rules

All feature definitions in this document are subject to the following causality constraints:

- No feature at cycle \(t\) may use sensor values, labels, statistics, or derived quantities from any future cycle \(t+1\) onward.
- Every feature for engine \(e\) at cycle \(t\) must be computed using only observations from engine \(e\) at cycles less than or equal to \(t\).
- Engine-wise independence is mandatory; no feature may use observations or summaries from another engine.
- Rolling windows must be backward-looking and must end at the current cycle.
- Centered windows are prohibited.
- Dataset-wide statistics must not be used to construct time-series features when they encode future, cross-engine, validation, or test information.
- Train, validation, and test partitions must remain isolated during feature construction and any subsequent normalization design.

These rules define the admissible feature space for a leakage-resistant predictive maintenance methodology on FD001.
