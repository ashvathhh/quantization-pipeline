from datasets import load_dataset
import random

def load_imdb_data(test_size=200, calibration_size=512, seed=42):
    print("📦 Downloading IMDB dataset...")
    print("   (Takes ~30 seconds first time, cached after that)")

    dataset = load_dataset("imdb")

    random.seed(seed)

    test_pool = list(zip(
        dataset["test"]["text"],
        dataset["test"]["label"]
    ))
    random.shuffle(test_pool)

    test_subset = test_pool[:test_size]
    test_texts  = [item[0] for item in test_subset]
    test_labels = [item[1] for item in test_subset]

    calib_pool = list(zip(
        dataset["train"]["text"],
        dataset["train"]["label"]
    ))
    random.shuffle(calib_pool)
    calib_texts = [item[0] for item in calib_pool[:calibration_size]]

    print(f"✅ Test set:        {len(test_texts)} reviews")
    print(f"✅ Calibration set: {len(calib_texts)} reviews")
    print(f"\n📝 Sample review:")
    print(f'   "{test_texts[0][:100]}..."')
    print(f'   Label: {"POSITIVE" if test_labels[0]==1 else "NEGATIVE"}')

    return test_texts, test_labels, calib_texts
