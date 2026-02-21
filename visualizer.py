import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

os.makedirs("results", exist_ok=True)

# Clean, professional color palette — no neon
COLORS = {
    "FP32":     "#DC2626",
    "FP16":     "#D97706",
    "INT8 PTQ": "#059669",
}

def _style_ax(ax):
    """Apply clean minimal styling to an axis."""
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E5E7EB")
    ax.spines["bottom"].set_color("#E5E7EB")
    ax.tick_params(colors="#6B7280", labelsize=9)
    ax.yaxis.grid(True, color="#F3F4F6", linewidth=1)
    ax.set_axisbelow(True)


def plot_all(results):
    print("\nGenerating benchmark charts...")

    models = [r["model"] for r in results]
    colors = [COLORS.get(m, "#6B7280") for m in models]
    base   = results[0]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.patch.set_facecolor("#FFFFFF")
    fig.suptitle(
        "Quantization Benchmark — FP32 vs FP16 vs INT8 PTQ",
        fontsize=11, fontweight="600", color="#1A1D23",
        y=1.01, fontfamily="sans-serif"
    )

    # ── Chart 1: Speedup ──────────────────────────────────────
    ax = axes[0]
    _style_ax(ax)
    speedups = [round(base["avg_latency_ms"] / r["avg_latency_ms"], 2) for r in results]
    bars = ax.bar(models, speedups, color=colors, width=0.45, edgecolor="none")
    for bar, v in zip(bars, speedups):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{v}×",
            ha="center", va="bottom",
            color="#1A1D23", fontsize=9, fontweight="600"
        )
    ax.set_title("Inference Speedup vs FP32", fontsize=10,
                 color="#374151", pad=10, fontweight="500")
    ax.set_ylabel("Speedup (×)", fontsize=9, color="#6B7280")
    ax.set_ylim(0, max(speedups) * 1.25)

    # ── Chart 2: Memory ───────────────────────────────────────
    ax = axes[1]
    _style_ax(ax)
    memory = [r["memory_mb"] for r in results]
    bars   = ax.bar(models, memory, color=colors, width=0.45, edgecolor="none")
    for bar, v in zip(bars, memory):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{v}MB",
            ha="center", va="bottom",
            color="#1A1D23", fontsize=9, fontweight="600"
        )
    ax.set_title("Model Memory Footprint", fontsize=10,
                 color="#374151", pad=10, fontweight="500")
    ax.set_ylabel("Memory (MB)", fontsize=9, color="#6B7280")
    ax.set_ylim(0, max(memory) * 1.2)

    # ── Chart 3: Accuracy vs Latency ─────────────────────────
    ax = axes[2]
    _style_ax(ax)
    for r, c in zip(results, colors):
        ax.scatter(r["avg_latency_ms"], r["accuracy"],
                   color=c, s=120, zorder=5, edgecolors="white", linewidths=1.5)
        ax.annotate(
            r["model"],
            (r["avg_latency_ms"], r["accuracy"]),
            textcoords="offset points", xytext=(8, 3),
            color="#374151", fontsize=8.5, fontweight="500"
        )
    ax.set_title("Accuracy vs Inference Speed", fontsize=10,
                 color="#374151", pad=10, fontweight="500")
    ax.set_xlabel("Avg Latency (ms)  —  lower is better", fontsize=9, color="#6B7280")
    ax.set_ylabel("Accuracy (%)  —  higher is better", fontsize=9, color="#6B7280")
    ax.xaxis.grid(True, color="#F3F4F6", linewidth=1)

    plt.tight_layout(pad=2)
    save_path = "results/benchmark_charts.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="#FFFFFF", edgecolor="none")
    plt.close()

    print(f"Charts saved to {os.path.abspath(save_path)}")
