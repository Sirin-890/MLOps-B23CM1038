
import argparse
import gzip
import json
import logging
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import requests
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

LOCAL_MODEL_DIR = "distilbert-uncased-goodreads-genres"
RESULTS_DIR     = "results"
LOGS_DIR        = "logs"
MAX_LENGTH      = 128
SEED            = 42
REVIEWS_PER_GENRE = 1_000
TRAIN_RATIO       = 0.8

GENRE_URLS: Dict[str, str] = {
    "poetry":                 "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz",
    "children":               "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz",
    "comics_graphic":         "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz",
    "fantasy_paranormal":     "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz",
    "history_biography":      "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz",
    "mystery_thriller_crime": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz",
    "romance":                "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz",
    "young_adult":            "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz",
}

os.environ["WANDB_DISABLED"] = "true"

Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOGS_DIR, "eval.log")),
    ],
    force=True,
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_reviews_from_url(url: str, head: int = 10_000, sample_size: int = 2_000) -> List[str]:
    logger.info("Fetching: %s", url)
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    reviews = []
    with gzip.open(response.raw, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if head is not None and i >= head:
                break
            text = json.loads(line).get("review_text", "").strip()
            if text:
                reviews.append(text)
    return random.sample(reviews, min(sample_size, len(reviews)))


def load_all_genres(cache_path: str = "genre_reviews_dict.pickle") -> Dict[str, List[str]]:
    cache = Path(cache_path)
    if cache.exists():
        logger.info("Loading reviews from cache: %s", cache_path)
        return pickle.load(cache.open("rb"))
    genre_reviews: Dict[str, List[str]] = {}
    for genre, url in GENRE_URLS.items():
        logger.info("Loading genre: %s", genre)
        genre_reviews[genre] = load_reviews_from_url(url)
    pickle.dump(genre_reviews, cache.open("wb"))
    return genre_reviews


def make_split(
    genre_reviews: Dict[str, List[str]],
    reviews_per_genre: int = REVIEWS_PER_GENRE,
    train_ratio: float = TRAIN_RATIO,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    train_texts, train_labels, test_texts, test_labels = [], [], [], []
    for genre, reviews in genre_reviews.items():
        sample = random.sample(reviews, min(reviews_per_genre, len(reviews)))
        split = int(len(sample) * train_ratio)
        for t in sample[:split]:
            train_texts.append(t); train_labels.append(genre)
        for t in sample[split:]:
            test_texts.append(t);  test_labels.append(genre)
    return train_texts, train_labels, test_texts, test_labels


def build_label_maps(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    unique = sorted(set(labels))
    label2id = {l: i for i, l in enumerate(unique)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels: List[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def save_json(obj: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
    logger.info("Saved → %s", path)


def save_heatmap(
    true_labels: List[str],
    pred_labels: List[str],
    output_path: str,
    title: str = "Confusion Matrix",
    exclude_diagonal: bool = False,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not installed – skipping heatmap.")
        return

    counts = defaultdict(int)
    for t, p in zip(true_labels, pred_labels):
        if exclude_diagonal and t == p:
            continue
        counts[(t, p)] += 1

    records = [{"True": t, "Predicted": p, "Count": c} for (t, p), c in counts.items()]
    import pandas as pd
    df = pd.DataFrame(records).pivot_table(index="True", columns="Predicted", values="Count")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df, linewidths=0.5, cmap="Purples", ax=ax)
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved heatmap → %s", output_path)


def evaluate_model(
    model_path: str,
    test_dataset: "ReviewDataset",
    test_labels: List[str],
    id2label: Dict[int, str],
    label: str = "local",
) -> dict:
    """Load model from model_path and evaluate on test_dataset."""
    device = get_device()
    logger.info("--- Evaluating '%s' from: %s ---", label, model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

    eval_args = TrainingArguments(
        output_dir=os.path.join(RESULTS_DIR, f"tmp_{label}"),
        per_device_eval_batch_size=32,
        report_to=[],
    )
    trainer = Trainer(model=model, args=eval_args, processing_class=tokenizer)

    pred_output = trainer.predict(test_dataset)
    pred_ids    = pred_output.predictions.argmax(-1).flatten().tolist()
    pred_labels = [id2label[i] for i in pred_ids]

    acc  = accuracy_score(test_labels, pred_labels)
    f1_w = f1_score(test_labels, pred_labels, average="weighted")
    f1_m = f1_score(test_labels, pred_labels, average="macro")
    loss = float(pred_output.metrics.get("test_loss", -1))
    report = classification_report(test_labels, pred_labels, output_dict=True)

    metrics = {
        "model_path":            model_path,
        "accuracy":              round(acc,  4),
        "f1_weighted":           round(f1_w, 4),
        "f1_macro":              round(f1_m, 4),
        "test_loss":             round(loss, 4),
        "classification_report": report,
    }
    save_json(metrics, os.path.join(RESULTS_DIR, f"eval_results_{label}.json"))
    logger.info("Accuracy: %.4f | F1 (weighted): %.4f | Loss: %.4f", acc, f1_w, loss)
    logger.info("\n%s", classification_report(test_labels, pred_labels))

    save_heatmap(test_labels, pred_labels,
                 os.path.join(RESULTS_DIR, f"confusion_{label}.png"),
                 title=f"Confusion Matrix – {label}")
    save_heatmap(test_labels, pred_labels,
                 os.path.join(RESULTS_DIR, f"confusion_{label}_misclass.png"),
                 title=f"Misclassifications – {label}",
                 exclude_diagonal=True)

    return metrics


def push_to_hub(repo_id: str, private: bool = False) -> None:
    """Push locally saved model and tokeniser to the HuggingFace Hub."""
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    logger.info("=== Pushing model to Hub: %s ===", repo_id)

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    model     = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR)

    tokenizer.push_to_hub(repo_id, private=private, token=token)
    model.push_to_hub(repo_id, private=private, token=token)

    metrics_path = Path(RESULTS_DIR) / "train_metrics.json"
    if metrics_path.exists():
        HfApi().upload_file(
            path_or_fileobj=str(metrics_path),
            path_in_repo="train_metrics.json",
            repo_id=repo_id,
            token=token,
        )
        logger.info("Uploaded train_metrics.json to Hub")

    logger.info("Model live at: https://huggingface.co/%s", repo_id)


def print_comparison(local: dict, hub: dict) -> None:
    comparison = {
        "local": {k: local[k] for k in ("accuracy", "f1_weighted", "f1_macro", "test_loss")},
        "hub":   {k: hub[k]   for k in ("accuracy", "f1_weighted", "f1_macro", "test_loss")},
    }
    save_json(comparison, os.path.join(RESULTS_DIR, "comparison.json"))

    print("\n" + "=" * 55)
    print(f"{'Metric':<22} {'Local':>14} {'Hub':>14}")
    print("-" * 55)
    for k in ("accuracy", "f1_weighted", "f1_macro", "test_loss"):
        print(f"{k:<22} {local[k]:>14.4f} {hub[k]:>14.4f}")
    print("=" * 55)
    print(f"\nFull results saved to {RESULTS_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="Evaluate & push Goodreads genre classifier")
    parser.add_argument("--hub_repo",   default=None,
                        help="HF repo ID, e.g. myusername/roberta-goodreads-genres")
    parser.add_argument("--skip_push",  action="store_true",
                        help="Skip Hub push (model already uploaded)")
    parser.add_argument("--local_only", action="store_true",
                        help="Only evaluate local model, no Hub steps")
    parser.add_argument("--private",    action="store_true",
                        help="Make the Hub repo private")
    args = parser.parse_args()

    set_seed()

    logger.info("=== Step 1: Preparing test data ===")
    genre_reviews = load_all_genres()
    train_texts, train_labels, test_texts, test_labels = make_split(genre_reviews)
    label2id, id2label = build_label_maps(train_labels)

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    test_enc  = tokenizer(test_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    test_dataset = ReviewDataset(test_enc, [label2id[y] for y in test_labels])

    logger.info("=== Step 2: Local evaluation ===")
    local_metrics = evaluate_model(
        LOCAL_MODEL_DIR, test_dataset, test_labels, id2label, label="local"
    )

    if args.local_only:
        logger.info("--local_only set. Done.")
        return

    hub_repo = args.hub_repo
    from huggingface_hub import HfApi

    logger.info("=== Step 3: Preparing HuggingFace repo ===")

    token = os.environ.get("HF_TOKEN")
    if token is None:
        raise ValueError("HF_TOKEN not found. Run `huggingface-cli login` or export HF_TOKEN.")

    api = HfApi()
    user_info = api.whoami(token=token)
    username = user_info["name"]

    DEFAULT_REPO_NAME = "mlops-assignment3_final"

    hub_repo = args.hub_repo if args.hub_repo else f"{username}/{DEFAULT_REPO_NAME}"

    api.create_repo(
        repo_id=hub_repo,
        private=args.private,
        exist_ok=True,   
        token=token,
    )

    logger.info("Using repo → %s", hub_repo)

    if not args.skip_push:
        logger.info("=== Pushing model to Hub ===")
        push_to_hub(hub_repo, private=args.private)
    else:
        logger.info("Skipping push (--skip_push).")

    logger.info("=== Step 4: Hub evaluation ===")
    hub_metrics = evaluate_model(
        hub_repo, test_dataset, test_labels, id2label, label="hub"
    )

    logger.info("=== Step 5: Comparison ===")
    print_comparison(local_metrics, hub_metrics)


if __name__ == "__main__":
    main()