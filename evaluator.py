import numpy as np
from tqdm import tqdm


def evaluate_model(model, texts, labels, max_samples=200):
    """
    Runs the model on test data and measures:
    - Accuracy
    - Average latency
    - P99 latency
    - Memory usage
    """
    print(f"\n📊 Evaluating {model.name} on {max_samples} samples...")

    correct, latencies, errors = 0, [], 0

    for text, label in tqdm(
        zip(texts[:max_samples], labels[:max_samples]),
        total=max_samples,
        desc=f"  {model.name}"
    ):
        try:
            result = model.predict(text)
            pred   = 1 if result["label"] == "POSITIVE" else 0
            if pred == label:
                correct += 1
            latencies.append(result["latency_ms"])
        except Exception:
            errors += 1

    total    = max_samples - errors
    accuracy = round(correct / total * 100, 2) if total > 0 else 0

    metrics = {
        "model":          model.name,
        "accuracy":       accuracy,
        "avg_latency_ms": round(float(np.mean(latencies)), 2),
        "p99_latency_ms": round(float(np.percentile(latencies, 99)), 2),
        "memory_mb":      model.get_memory_mb(),
        "errors":         errors,
        "samples":        total,
    }

    print(f"   ✅ Accuracy:  {metrics['accuracy']}%")
    print(f"   ⚡ Latency:   {metrics['avg_latency_ms']}ms avg  |  {metrics['p99_latency_ms']}ms p99")
    print(f"   💾 Memory:    {metrics['memory_mb']} MB")

    return metrics


def print_results_table(results):
    """Prints a clean comparison table to the terminal."""
    baseline = next(r for r in results if r["model"] == "FP32")

    print("\n" + "=" * 70)
    print("  FINAL BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Model':<12} {'Accuracy':>10} {'Drop':>8} "
          f"{'Avg Latency':>13} {'Speedup':>9} {'Memory':>9}")
    print("-" * 70)

    for m in results:
        drop    = f"{m['accuracy'] - baseline['accuracy']:+.1f}%" \
                  if m["model"] != "FP32" else "—"
        speedup = f"{baseline['avg_latency_ms'] / m['avg_latency_ms']:.1f}×"
        print(
            f"{m['model']:<12} {m['accuracy']:>9}%  {drop:>8}  "
            f"{m['avg_latency_ms']:>9}ms  {speedup:>8}  "
            f"{m['memory_mb']:>7}MB"
        )

    print("=" * 70)
