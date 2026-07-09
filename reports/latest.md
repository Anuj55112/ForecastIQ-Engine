# Benchmark Verification Report: ForecastIQ Engine
- **Generated Timestamp**: 2026-07-09T09:45:33.059735Z
- **Status**: SUCCESS

## Hardware Metadata
- **OS**: Darwin
- **CPU**: Apple M3
- **RAM**: 8 GB
- **GPU**: Apple Metal (MPS)

## Environment Versions
- **Python**: 3.12.7
- **PyTorch**: 2.12.1

## Model Metadata
- **Model**: PatchTST / N-BEATS
- **Dataset**: Simulated Demand Series (500 steps)
- **Batch Size**: 16
- **Sequence Length**: 96
- **Device**: cpu

## Measured Benchmark Results
| Model | Batch Latency (ms) | Inference Throughput |
| :--- | :---: | :---: |
| **PatchTST** | 0.53 ms | 1899.12 batches/sec |
| **N-BEATS** | 0.4 ms | 2492.23 batches/sec |
