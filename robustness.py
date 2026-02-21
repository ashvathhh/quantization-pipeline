import random


def inject_typos(text, error_rate=0.08, seed=42):
    """
    Randomly swaps adjacent characters to simulate real-world typos.
    Example: "fantastic" → "fatansitc"
    """
    random.seed(seed)
    chars    = list(text)
    n_errors = max(1, int(len(chars) * error_rate))
    for _ in range(n_errors):
        idx = random.randint(0, len(chars) - 2)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    return "".join(chars)


def evaluate_robustness(models, texts, labels, n=100):
    """
    Compares each model on clean vs typo-injected inputs.
    Shows which model degrades the most under noisy conditions.
    """
    print("\n" + "=" * 55)
    print("  ROBUSTNESS TEST — Typo Injection")
    print("=" * 55)

    clean     = texts[:n]
    perturbed = [inject_typos(t) for t in clean]
    labs      = labels[:n]

    print(f"\n📝 Example perturbation:")
    print(f"   CLEAN:     {clean[0][:75]}")
    print(f"   PERTURBED: {perturbed[0][:75]}\n")

    all_results = []

    for model in models:
        clean_ok = 0
        pert_ok  = 0

        for c, p, label in zip(clean, perturbed, labs):
            # Test on clean text
            try:
                r = model.predict(c)
                if (1 if r["label"] == "POSITIVE" else 0) == label:
                    clean_ok += 1
            except Exception:
                pass

            # Test on perturbed text
            try:
                r = model.predict(p)
                if (1 if r["label"] == "POSITIVE" else 0) == label:
                    pert_ok += 1
            except Exception:
                pass

        clean_acc = round(clean_ok / n * 100, 2)
        pert_acc  = round(pert_ok  / n * 100, 2)
        drop      = round(clean_acc - pert_acc, 2)

        all_results.append({
            "model":     model.name,
            "clean_acc": clean_acc,
            "pert_acc":  pert_acc,
            "drop":      drop
        })

        status = "🟢" if drop < 1.5 else "🟡" if drop < 3 else "🔴"
        print(f"  {status} {model.name:<12}  "
              f"Clean: {clean_acc}%  →  "
              f"Perturbed: {pert_acc}%  "
              f"(drop: -{drop}%)")

    print("\n" + "-" * 55)
    worst = max(all_results, key=lambda x: x["drop"])
    best  = min(all_results, key=lambda x: x["drop"])
    print(f"  Most robust:  {best['model']:<12}  drop: -{best['drop']}%")
    print(f"  Most fragile: {worst['model']:<12}  drop: -{worst['drop']}%")
    print("=" * 55)

    return all_results
