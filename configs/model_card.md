# Model Card: ForecastIQ Time-Series Engines

This model card documents the specifications and details of the time-series forecasting models deployed within the **ForecastIQ** platform.

## Model Details
- **Developer**: Portfolio Owner
- **Model Types**:
  - **PatchTST**: A modern transformer architecture utilizing channel-independence and subseries patching to capture local temporal structures.
  - **N-BEATS**: A deep stack of fully-connected blocks with backward and forward residual connections designed for univariate sequence decomposition.
  - **Temporal Fusion Transformer (TFT)**: An attention-based architecture that integrates multi-horizon forecasting, static covariate encoders, and gated residual networks (GLU).
- **Task**: Multi-step Sequence Forecasting (predicting 24 steps into the future given 96 steps of history).

## Intended Use
- **Primary Intended Use**: Research, benchmarking, and comparative performance analysis of transformer-based vs. recurrent/FC-based sequence models.
- **Out of Scope Use**: Financial trading or high-stakes industrial process control without extensive domain-specific calibration.

## Datasets & Training
- **Training Data**: Evaluated using simulated multi-frequency seasonal demand trends (sinusoidal drifts, linear trends, and noise).
- **Preprocessing**:
  - Min-Max normalization mapping values relative to mean/std scaling factors.
  - Formulated as rolling-window sequences of shape `[Batch, Seq_len, Features]`.

## Evaluation & Metrics
- **Primary Metrics**:
  - **Mean Absolute Error (MAE)**
  - **Root Mean Squared Error (RMSE)**
  - **Symmetric Mean Absolute Percentage Error (sMAPE)**

## Limitations
- Performance depends highly on matching the configured context window (96 steps).
- Checkpoint weights require training on specific real-world target datasets (e.g. Weather, Traffic, Exchange rate) before deployment.
