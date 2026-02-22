
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
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_NAME      = "distilbert-base-uncased"
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
        logging.FileHandler(os.path.join(LOGS_DIR, "train.log")),
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
    """Stream reviews from a gzipped JSONL URL and return a random sample."""
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
    """Load reviews for every genre, using a pickle cache when available."""
    cache = Path(cache_path)
    if cache.exists():
        logger.info("Loading reviews from cache: %s", cache_path)
        return pickle.load(cache.open("rb"))
    genre_reviews: Dict[str, List[str]] = {}
    for genre, url in GENRE_URLS.items():
        logger.info("Loading genre: %s", genre)
        genre_reviews[genre] = load_reviews_from_url(url)
    pickle.dump(genre_reviews, cache.open("wb"))
    logger.info("Saved cache → %s", cache_path)
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


def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    return {
        "accuracy":    round(accuracy_score(labels, preds), 4),
        "f1_weighted": round(f1_score(labels, preds, average="weighted"), 4),
    }


def save_json(obj: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
    logger.info("Saved → %s", path)


def main():
    set_seed()
    device = get_device()
    logger.info("Device: %s", device)

  
    logger.info("=== Step 1: Loading data ===")
    genre_reviews = load_all_genres()
    train_texts, train_labels, test_texts, test_labels = make_split(genre_reviews)
    logger.info("Train: %d | Test: %d | Genres: %d",
                len(train_texts), len(test_texts), len(set(train_labels)))

    label2id, id2label = build_label_maps(train_labels)
    save_json(label2id, os.path.join(RESULTS_DIR, "label2id.json"))
    save_json({str(k): v for k, v in id2label.items()},
              os.path.join(RESULTS_DIR, "id2label.json"))

   
    logger.info("=== Step 2: Tokenising ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    test_enc  = tokenizer(test_texts,  truncation=True, padding=True, max_length=MAX_LENGTH)
    train_dataset = ReviewDataset(train_enc, [label2id[y] for y in train_labels])
    test_dataset  = ReviewDataset(test_enc,  [label2id[y] for y in test_labels])


    logger.info("=== Step 3: Loading model: %s ===", MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    ).to(device)


    use_gpu = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=RESULTS_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        learning_rate=2e-5,
        warmup_steps=200,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        fp16=use_gpu,
        dataloader_num_workers=0,            
        dataloader_pin_memory=False,
        report_to=[],
        seed=SEED,
    )


    logger.info("=== Step 4: Training ===")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    logger.info("Training metrics: %s", train_result.metrics)
    save_json(train_result.metrics, os.path.join(RESULTS_DIR, "train_metrics.json"))


    logger.info("=== Step 5: Saving model → %s ===", LOCAL_MODEL_DIR)
    trainer.save_model(LOCAL_MODEL_DIR)
    tokenizer.save_pretrained(LOCAL_MODEL_DIR)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()