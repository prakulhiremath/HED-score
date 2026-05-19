# HED-Bench

Benchmarking Early Detection Reliability in Streaming Systems

HED-Bench is an open benchmark suite for evaluating temporal detection systems under realistic streaming conditions. It focuses on *when* a detector reacts — not just whether it eventually predicts correctly.

Traditional metrics like AUC, accuracy, and F1 often fail to capture delayed detection, slow poisoning, temporal instability, and late-response failures in cyber and time-series environments. HED-Bench introduces reproducible evaluation pipelines built around the HED-Score framework.

---

## Goals

- Evaluate temporal reliability of detection systems
- Benchmark latency-sensitive anomaly detection
- Compare models under delayed and adversarial shifts
- Provide reproducible streaming evaluation pipelines
- Standardize robustness evaluation across domains

---

## Core Features

- Streaming evaluation framework
- HED-FAR analysis
- Delayed detection benchmarking
- Adversarial drift simulations
- Cybersecurity and financial time-series tasks
- Reproducible baselines and configs
- Extensible dataset loader system

---

## Repository Structure

```text
hed-bench/
├── benchmark.py
├── configs/
├── datasets/
├── baselines/
├── experiments/
├── evaluation/
├── plots/
├── results/
├── notebooks/
└── tests/
```

---

## Supported Tasks

### Cybersecurity
- Intrusion detection
- Slow poisoning detection
- Streaming anomaly detection
- Adversarial traffic shifts

### Financial Time-Series
- Market regime change detection
- Volatility transition analysis
- Streaming event detection

### Synthetic Temporal Drift
- Controlled delayed-shift environments
- Same-AUC different-latency scenarios
- Distribution drift generators

---

## Evaluation Metrics

HED-Bench supports:

- HED-Score
- FAR (False Alarm Rate)
- Detection latency
- AUC
- F1-score
- Robustness stability metrics

The benchmark emphasizes temporal correctness over static classification quality.

---

## Installation

```bash
git clone https://github.com/your-org/hed-bench.git

cd hed-bench

pip install -r requirements.txt
```

---

## Quick Start

Run the benchmark suite:

```bash
python benchmark.py --config configs/cyber.yaml
```

Run a synthetic delayed-detection experiment:

```bash
python experiments/delayed_detection.py
```

Generate comparison plots:

```bash
python plots/hed_vs_auc.py
```

---

## Example Research Questions

- Can two models achieve identical AUC while differing significantly in temporal responsiveness?
- How robust are streaming detectors against slow poisoning attacks?
- Which architectures maintain stable early detection under distribution drift?
- How should latency-sensitive anomaly detectors be evaluated beyond static metrics?

---

## Planned Benchmarks

- NSL-KDD
- UNSW-NB15
- CICIDS
- Financial regime datasets
- Sensor anomaly streams

---

## Long-Term Direction

HED-Bench aims to become a standardized evaluation ecosystem for:
- temporal anomaly detection,
- low-latency cybersecurity systems,
- streaming robustness evaluation,
- and adaptive detection research.

---

## Citation

```bibtex
@misc{hedbench2026,
  title={HED-Bench: Benchmarking Early Detection Reliability in Streaming Systems},
  author={Your Name},
  year={2026},
  note={Open-source benchmark framework}
}
```

---

## Status

Early research-stage development.

Contributions, baseline implementations, and benchmark extensions are welcome.
