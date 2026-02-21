# Quantization-Aware Inference Pipeline

Layer-aware benchmarking of FP32, FP16, and INT8 quantization for DistilBERT on sentiment classification.

**CS 5130 — Applied Programming and Data Processing for AI**
Ashvath Cheppalli · Saketh Kuppili

---

## What This Project Does

Deploys DistilBERT for sentiment analysis at three precision levels and systematically answers:

1. When is post-training quantization (PTQ) sufficient vs. quantization-aware training (QAT)?
2. Which transformer layers are most sensitive to quantization?
3. Does INT8 degrade more than FP32 on noisy real-world inputs?

---

## Results

| Model | Accuracy | Latency | Memory | Speedup |
|---|---|---|---|---|
| FP32 | 88.5% | 80ms | 255MB | 1.0× |
| FP16 | 88.5% | 603ms | 128MB | 0.1× (CPU) |
| INT8 PTQ | 86.5% | 35ms | 91MB | 2.3× |

**Key findings:**
- INT8 PTQ is 2.3× faster and uses 64% less memory with only 2% accuracy loss
- FP16 is slower than FP32 on CPU — only beneficial with a GPU
- Attention Value matrices (Blocks 2 and 3) are the most sensitive layers — 72-74% accuracy drop when quantized alone
- FFN Block 1 layers can be safely compressed to INT8 with zero accuracy loss
- All models fail on heavily typo-injected inputs — reversing predictions with high confidence

---

## Project Structure

```
quantization-pipeline/
├── app.py              # Entry point — runs everything
├── models.py           # FP32, FP16, INT8 PTQ model classes
├── data_loader.py      # IMDB dataset download and preparation
├── evaluator.py        # Accuracy, latency, memory benchmarking
├── visualizer.py       # Chart generation
├── robustness.py       # Typo injection and robustness testing
├── sensitivity.py      # Layer-wise sensitivity analysis
├── ui.py               # Gradio web interface
├── requirements.txt    # Python dependencies
└── SETUP.txt           # Detailed setup instructions
```

---

## Setup

### Requirements

- Python 3.9 or higher
- Windows, Mac, or Linux
- No GPU required (CPU inference supported)

### Install

```bash
# Clone the repository
git clone https://github.com/ashvathhh/quantization-pipeline.git
cd quantization-pipeline

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

Then open your browser and go to:
```
http://127.0.0.1:7860
```

---

## What Happens When You Run It

The pipeline runs automatically in this order:

1. Downloads the IMDB dataset (50,000 reviews) from HuggingFace
2. Loads FP32, FP16, and INT8 PTQ versions of DistilBERT
3. Benchmarks all three models on 200 test reviews
4. Saves benchmark charts to `results/benchmark_charts.png`
5. Runs robustness test with typo-injected inputs
6. Runs layer-wise sensitivity analysis across all 38 layers (~10 min on CPU)
7. Saves sensitivity chart to `results/sensitivity_analysis.png`
8. Launches the web UI at `http://127.0.0.1:7860`

**First run takes 5-10 minutes** — the dataset and model download once and are cached after that. Every run after is faster.

---

## Dataset

The project uses the [IMDB Sentiment Dataset](https://huggingface.co/datasets/imdb) from HuggingFace — 50,000 movie reviews labeled positive or negative. Downloaded automatically when you run the project. No manual download required.

---

## Model

[distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) — a fine-tuned DistilBERT model for binary sentiment classification. Downloaded automatically from HuggingFace on first run (~250MB).

---

## Dependencies

All installed automatically via `pip install -r requirements.txt`:

```
torch
transformers
datasets
gradio
matplotlib
pandas
numpy
tqdm
seaborn
```

---

## Notes

- FP16 is slower than FP32 on CPU — this is expected behavior. FP16 only improves speed on GPU hardware with Tensor Cores (e.g. NVIDIA A100, V100, T4).
- The sensitivity analysis takes approximately 10 minutes on CPU because it tests each of the 38 layers individually.
- All results are saved to the `results/` folder automatically.
