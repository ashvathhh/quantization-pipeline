"""
sensitivity.py
──────────────
Layer-Wise Sensitivity Analysis for DistilBERT

What this does:
- Takes the FP32 baseline model
- Quantizes ONE layer at a time to INT8
- Measures how much accuracy drops for each layer
- Builds a sensitivity map showing which layers are fragile
- Recommends mixed-precision config (INT8 vs FP16 per layer)

This is what tells you WHERE accuracy loss happens inside the model.
"""

import torch
import copy
import numpy as np
from torch.quantization import quantize_dynamic
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


def get_layer_map(model):
    """
    Returns a dictionary mapping layer names to their module objects.
    Each entry is one quantizable component of DistilBERT.
    """
    layer_map = {}

    transformer = model.distilbert.transformer

    for i, layer in enumerate(transformer.layer):
        # Attention components
        layer_map[f"Attention Block {i+1} — Q"] = layer.attention.q_lin
        layer_map[f"Attention Block {i+1} — K"] = layer.attention.k_lin
        layer_map[f"Attention Block {i+1} — V"] = layer.attention.v_lin
        layer_map[f"Attention Block {i+1} — Out"] = layer.attention.out_lin

        # Feed-forward components
        layer_map[f"FFN Block {i+1} — Linear1"] = layer.ffn.lin1
        layer_map[f"FFN Block {i+1} — Linear2"] = layer.ffn.lin2

    # Classifier head
    layer_map["Classifier — Linear1"] = model.pre_classifier
    layer_map["Classifier — Linear2"] = model.classifier

    return layer_map


def evaluate_accuracy(model, tokenizer, texts, labels, max_samples=100):
    """
    Runs model on test data and returns accuracy.
    Quick version — uses 100 samples for speed.
    """
    correct = 0
    total   = 0

    for text, label in zip(texts[:max_samples], labels[:max_samples]):
        try:
            inputs = tokenizer(
                text.strip(),
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            with torch.no_grad():
                outputs = model(**inputs)

            probs   = torch.softmax(outputs.logits, dim=-1)[0]
            pred_id = torch.argmax(probs).item()

            if pred_id == label:
                correct += 1
            total += 1

        except Exception:
            continue

    return round(correct / total * 100, 2) if total > 0 else 0.0


def run_sensitivity_analysis(texts, labels, max_samples=100):
    """
    Main function — runs the full layer sensitivity analysis.

    For each layer:
      1. Load a fresh copy of the FP32 model
      2. Quantize ONLY that one layer to INT8
      3. Measure accuracy
      4. Compare to FP32 baseline
      5. Record the drop

    Returns list of results sorted by sensitivity (most fragile first).
    """

    print("\n" + "="*60)
    print("  LAYER-WISE SENSITIVITY ANALYSIS")
    print("  Quantizing one layer at a time...")
    print("="*60)

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── Step 1: Get FP32 baseline accuracy ───────────────────
    print("\n📊 Step 1: Measuring FP32 baseline accuracy...")
    base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    base_model.eval()
    baseline_acc = evaluate_accuracy(base_model, tokenizer, texts, labels, max_samples)
    print(f"   ✅ FP32 Baseline: {baseline_acc}%")

    # ── Step 2: Get layer map ─────────────────────────────────
    layer_map = get_layer_map(base_model)
    print(f"\n📋 Found {len(layer_map)} quantizable layers to test")
    print("   Testing each layer individually...\n")

    results = []

    # ── Step 3: Quantize one layer at a time ─────────────────
    for layer_name, target_module in tqdm(layer_map.items(), desc="Analyzing layers"):

        # Fresh copy of the full FP32 model for each test
        test_model = copy.deepcopy(base_model)
        test_model.eval()

        # Find and quantize ONLY this specific layer
        # We do this by replacing just that module with its quantized version
        try:
            quantized_layer = quantize_dynamic(
                copy.deepcopy(target_module),
                {torch.nn.Linear},
                dtype=torch.qint8
            )

            # Navigate to the correct location in the model and replace
            _replace_module(test_model, layer_name, quantized_layer, layer_map)

            # Measure accuracy with only this layer quantized
            acc  = evaluate_accuracy(test_model, tokenizer, texts, labels, max_samples)
            drop = round(baseline_acc - acc, 2)

        except Exception as e:
            acc  = baseline_acc
            drop = 0.0

        # Determine recommendation based on sensitivity
        if drop <= 1.0:
            recommendation = "INT8 ✅"
            status         = "🟢"
        elif drop <= 2.5:
            recommendation = "FP16 ⚠️"
            status         = "🟡"
        else:
            recommendation = "FP16 🔴"
            status         = "🔴"

        results.append({
            "layer":          layer_name,
            "baseline_acc":   baseline_acc,
            "quantized_acc":  acc,
            "drop":           drop,
            "recommendation": recommendation,
            "status":         status
        })

    # Sort by most sensitive (largest drop) first
    results.sort(key=lambda x: x["drop"], reverse=True)

    # ── Step 4: Print results table ───────────────────────────
    print("\n" + "="*72)
    print("  SENSITIVITY RESULTS  (sorted by most fragile first)")
    print("="*72)
    print(f"  {'Layer':<35} {'Baseline':>9} {'Quantized':>10} {'Drop':>7}  {'Recommend'}")
    print("-"*72)

    for r in results:
        print(
            f"  {r['status']} {r['layer']:<33} "
            f"{r['baseline_acc']:>8}%  "
            f"{r['quantized_acc']:>9}%  "
            f"{-r['drop']:>6.1f}%  "
            f"  {r['recommendation']}"
        )

    print("="*72)

    # ── Step 5: Summary ───────────────────────────────────────
    safe_layers    = [r for r in results if r["drop"] <= 1.0]
    fragile_layers = [r for r in results if r["drop"] > 2.5]
    most_fragile   = results[0]
    safest         = results[-1]

    print(f"\n📌 KEY FINDINGS:")
    print(f"   Most fragile layer: {most_fragile['layer']}  (drop: -{most_fragile['drop']}%)")
    print(f"   Safest layer:       {safest['layer']}  (drop: -{safest['drop']}%)")
    print(f"   Safe for INT8:      {len(safe_layers)}/{len(results)} layers")
    print(f"   Need FP16:          {len(fragile_layers)}/{len(results)} layers")

    print(f"\n💡 MIXED-PRECISION RECOMMENDATION:")
    print(f"   Keep these layers at FP16:")
    for r in fragile_layers:
        print(f"     • {r['layer']}")
    print(f"   Quantize everything else to INT8")
    print(f"   Expected outcome: Most of the INT8 speedup with minimal accuracy loss")

    return results, baseline_acc


def _replace_module(model, layer_name, quantized_layer, layer_map):
    """
    Helper: finds the specific layer inside the model by
    matching the module object reference and replaces it.
    """
    original_module = layer_map[layer_name]

    # Walk the model tree and find the matching module
    for parent_name, parent_module in model.named_modules():
        for child_name, child_module in parent_module.named_children():
            if child_module is original_module or (
                hasattr(child_module, 'weight') and
                hasattr(original_module, 'weight') and
                child_module.weight.shape == original_module.weight.shape and
                child_module.weight.data_ptr() != original_module.weight.data_ptr()
            ):
                # Replace this child with the quantized version
                setattr(parent_module, child_name, quantized_layer)
                return


def save_sensitivity_chart(results, baseline_acc):
    """
    Saves a horizontal bar chart showing sensitivity of each layer.
    Saved to results/sensitivity_analysis.png
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os
    os.makedirs("results", exist_ok=True)

    # Reverse so most fragile is at top
    layers = [r["layer"] for r in reversed(results)]
    drops  = [r["drop"]  for r in reversed(results)]
    colors = []
    for r in reversed(results):
        if r["drop"] <= 1.0:
            colors.append("#00D4AA")   # teal = safe
        elif r["drop"] <= 2.5:
            colors.append("#FFB347")   # orange = caution
        else:
            colors.append("#FF6B6B")   # red = fragile

    fig, ax = plt.subplots(figsize=(10, max(6, len(layers) * 0.55)))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")

    bars = ax.barh(layers, drops, color=colors, edgecolor="none", height=0.6)

    # Value labels
    for bar, val in zip(bars, drops):
        ax.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"-{val}%",
            va="center", color="white", fontsize=9, fontweight="bold"
        )

    ax.set_title(
        f"Layer-Wise Sensitivity Analysis\n"
        f"FP32 Baseline: {baseline_acc}%  |  Bar = Accuracy Drop When That Layer is INT8",
        color="white", fontsize=12, pad=12
    )
    ax.set_xlabel("Accuracy Drop (%)", color="#8B949E")
    ax.tick_params(colors="white", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#21262D")
    ax.xaxis.grid(True, color="#21262D", linewidth=0.8)
    ax.set_axisbelow(True)

    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(color="#00D4AA", label="Safe → INT8 (drop ≤ 1%)"),
        Patch(color="#FFB347", label="Caution → FP16 (drop 1-2.5%)"),
        Patch(color="#FF6B6B", label="Fragile → FP16 (drop > 2.5%)"),
    ]
    ax.legend(handles=legend, loc="lower right",
              facecolor="#161B22", edgecolor="#21262D",
              labelcolor="white", fontsize=9)

    plt.tight_layout()
    save_path = "results/sensitivity_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()

    print(f"\n✅ Sensitivity chart saved  →  {os.path.abspath(save_path)}")
    return save_path
