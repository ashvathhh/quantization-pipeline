import torch
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.quantization import quantize_dynamic

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


# ─────────────────────────────────────────────────────────────
# FP32 — Full Precision Baseline
# ─────────────────────────────────────────────────────────────
class FP32Model:
    def __init__(self):
        self.name = "FP32"

    def load(self):
        print("🔄 Loading FP32 model...")
        print("   (Downloads ~250MB the first time, cached after)")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"✅ FP32 ready  |  Device: {self.device.upper()}  |  Memory: {self.get_memory_mb()}MB")
        return self

    def predict(self, text):
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            raise ValueError("Input must be a non-empty string")

        inputs = self.tokenizer(
            text.strip(), return_tensors="pt",
            truncation=True, max_length=512, padding=True
        ).to(self.device)

        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(**inputs)
        latency_ms = (time.perf_counter() - start) * 1000

        probs   = torch.softmax(outputs.logits, dim=-1)[0]
        pred_id = torch.argmax(probs).item()

        return {
            "label":      "POSITIVE" if pred_id == 1 else "NEGATIVE",
            "confidence": round(probs[pred_id].item() * 100, 2),
            "latency_ms": round(latency_ms, 2)
        }

    def get_memory_mb(self):
        return round(sum(
            p.numel() * p.element_size()
            for p in self.model.parameters()
        ) / (1024 * 1024), 1)


# ─────────────────────────────────────────────────────────────
# FP16 — Half Precision
# ─────────────────────────────────────────────────────────────
class FP16Model:
    def __init__(self):
        self.name = "FP16"

    def load(self):
        print("🔄 Loading FP16 model...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model     = self.model.half()   # convert weights to 16-bit
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"✅ FP16 ready  |  Device: {self.device.upper()}  |  Memory: {self.get_memory_mb()}MB")
        return self

    def predict(self, text):
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            raise ValueError("Input must be a non-empty string")

        inputs = self.tokenizer(
            text.strip(), return_tensors="pt",
            truncation=True, max_length=512, padding=True
        ).to(self.device)

        start = time.perf_counter()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                outputs = self.model(**inputs)
        latency_ms = (time.perf_counter() - start) * 1000

        probs   = torch.softmax(outputs.logits.float(), dim=-1)[0]
        pred_id = torch.argmax(probs).item()

        return {
            "label":      "POSITIVE" if pred_id == 1 else "NEGATIVE",
            "confidence": round(probs[pred_id].item() * 100, 2),
            "latency_ms": round(latency_ms, 2)
        }

    def get_memory_mb(self):
        return round(sum(
            p.numel() * p.element_size()
            for p in self.model.parameters()
        ) / (1024 * 1024), 1)


# ─────────────────────────────────────────────────────────────
# INT8 PTQ — Post Training Quantization
# ─────────────────────────────────────────────────────────────
class PTQModel:
    def __init__(self):
        self.name   = "INT8 PTQ"
        self.device = "cpu"   # PTQ is optimized for CPU

    def load(self, calib_texts=None):
        print("🔄 Loading INT8 PTQ model...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        base           = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        base.eval()

        print("   ⚙️  Compressing weights to INT8...")
        self.model = quantize_dynamic(
            base,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        print("   ✅ Compression complete!")

        if calib_texts:
            print(f"   📊 Running calibration pass...")
            for t in calib_texts[:32]:
                try:
                    self.predict(t)
                except Exception:
                    continue
            print("   ✅ Calibration done!")

        print(f"✅ INT8 PTQ ready  |  Device: CPU  |  Memory: {self.get_memory_mb()}MB")
        return self

    def predict(self, text):
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            raise ValueError("Input must be a non-empty string")

        inputs = self.tokenizer(
            text.strip(), return_tensors="pt",
            truncation=True, max_length=512, padding=True
        )

        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(**inputs)
        latency_ms = (time.perf_counter() - start) * 1000

        probs   = torch.softmax(outputs.logits, dim=-1)[0]
        pred_id = torch.argmax(probs).item()

        return {
            "label":      "POSITIVE" if pred_id == 1 else "NEGATIVE",
            "confidence": round(probs[pred_id].item() * 100, 2),
            "latency_ms": round(latency_ms, 2)
        }

    def get_memory_mb(self):
        return round(sum(
            p.numel() * p.element_size()
            for p in self.model.parameters()
        ) / (1024 * 1024), 1)
