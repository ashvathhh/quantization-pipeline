"""
app.py
Quantization-Aware Inference Pipeline
CS 5130 — Ashvath Cheppalli & Saketh Kuppili

Run:
    python app.py
"""

import os
import sys

from data_loader  import load_imdb_data
from models       import FP32Model, FP16Model, PTQModel
from evaluator    import evaluate_model, print_results_table
from visualizer   import plot_all
from robustness   import evaluate_robustness
from sensitivity  import run_sensitivity_analysis, save_sensitivity_chart
from ui           import build_ui


def separator(label=""):
    width = 58
    if label:
        pad   = (width - len(label) - 2) // 2
        print(f"\n{'─' * pad} {label} {'─' * pad}")
    else:
        print(f"\n{'─' * width}")


print()
print("Quantization-Aware Inference Pipeline")
print("CS 5130  |  Ashvath Cheppalli  |  Saketh Kuppili")
separator()

# ── Step 1: Data ──────────────────────────────────────────────
separator("DATA")
test_texts, test_labels, calib_texts = load_imdb_data(
    test_size=200,
    calibration_size=512
)

# ── Step 2: Models ────────────────────────────────────────────
separator("MODELS")
fp32_model = FP32Model().load()
fp16_model = FP16Model().load()
ptq_model  = PTQModel().load(calib_texts=calib_texts)

all_models = [fp32_model, fp16_model, ptq_model]

# ── Step 3: Benchmark ─────────────────────────────────────────
separator("BENCHMARK")
results = [
    evaluate_model(model, test_texts, test_labels, max_samples=200)
    for model in all_models
]
print_results_table(results)

# ── Step 4: Charts ────────────────────────────────────────────
separator("CHARTS")
plot_all(results)

# ── Step 5: Robustness ────────────────────────────────────────
separator("ROBUSTNESS")
evaluate_robustness(all_models, test_texts, test_labels, n=100)

# ── Step 6: Sensitivity Analysis ──────────────────────────────
separator("SENSITIVITY ANALYSIS")
print("Testing each layer individually (approx. 10 minutes on CPU)...")
sensitivity_results, baseline_acc = run_sensitivity_analysis(
    test_texts, test_labels, max_samples=100
)
save_sensitivity_chart(sensitivity_results, baseline_acc)

# ── Step 7: Launch UI ─────────────────────────────────────────
separator("UI")
print("Starting server at http://127.0.0.1:7860")
print("Open that address in your browser\n")

demo = build_ui(fp32_model, fp16_model, ptq_model, results)
demo.launch()
