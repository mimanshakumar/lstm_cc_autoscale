# LSTM-Based Predictive Auto-Scaling

Undergraduate cloud computing research (with Dr. Abhishek Gaur) on LSTM-based predictive auto-scaling for cloud infrastructure, with a focus on domain-shift generalization. Central finding: a frozen, pretrained LSTM fine-tuned on synthetic bursts shows **negative transfer** — it underperforms simple reactive scaling — when evaluated on real financial transaction traces. IEEE paper in progress.

## Pipeline

Notebooks are numbered in the order they're meant to run. Each is self-contained (imports what it needs) and its outputs (figures, printed metrics) are preserved from the last full run, so you don't need a GPU just to look at results.

| # | Notebook | What it does |
|---|----------|---------------|
| 00 | `upi_spike_motivation.ipynb` | Real-world UPI transaction spike analysis (NPCI data) — motivates why predictive auto-scaling matters |
| 01 | `reactive_scaler.ipynb` | Baseline reactive (threshold-based) auto-scaler, generates synthetic workloads (normal/spike/ramp) and saves baseline results |
| 02 | `lstm_training.ipynb` | Trains the LSTM predictor on Google 2019 / Alibaba 2018 cluster traces (pretraining), evaluates zero-shot transfer to Alibaba |
| 03 | `monte_carlo_arima_baseline.ipynb` | 30-seed Monte Carlo evaluation; ARIMA as a naive-transfer baseline for comparison |
| 04 | `finetune_spike_data.ipynb` | Fine-tunes the pretrained model on synthetic spike data |
| 05 | `evaluate_finetuned_model.ipynb` | Evaluates the fine-tuned model |
| 06 | `threshold_sweep.ipynb` | Sweeps scaling thresholds to produce a cost vs. SLA-violation trade-off curve |
| 07 | `horizon_ablation.ipynb` | Ablation over prediction horizon (5s/15s/30s); main strategy comparison table |
| 08 | `master_figures.ipynb` | Builds the consolidated figures used in the paper |
| 09 | `cross_domain_financial.ipynb` | Evaluates the model on real financial transaction traces (Financial1.spc, UMass Trace Repository) — where the negative transfer result shows up |
| 10 | `mixed_domain_finetune.ipynb` | v2 model: fine-tuned on mixed burst durations, testing whether fixing the burst-duration mismatch resolves the negative transfer |

## Model

2-layer LSTM, 64 hidden units, 0.2 dropout, prediction horizon = 3 steps (15s). Pretrained on Google 2019 and Alibaba 2018 cluster traces; evaluated on Alibaba (in-domain) and Financial1.spc (out-of-domain, real financial trace).

## Setup

```bash
git clone https://github.com/mimanshakumar/lstm_cc_autoscale.git
cd lstm_cc_autoscale
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Notebooks 00–01 and 06–08 run standalone. Notebooks that reference Colab-specific paths (e.g. `/content/*.csv`, `google.colab.files`) were originally written for Colab — when running locally, replace those with local file paths.

## Data sources

- Google 2019 and Alibaba 2018 cluster traces (pretraining)
- Financial1.spc — UMass Trace Repository (real financial trace evaluation)
- NPCI monthly UPI statistics, 2022–2024 (motivating spike analysis)
