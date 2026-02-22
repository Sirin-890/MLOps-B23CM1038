# Goodreads Genre Classifier — DistilBERT Fine-Tuning

Fine-tunes **`distilbert-base-uncased`** on Goodreads book reviews from the
[UCSD Book Graph](https://mengtingwan.github.io/data/goodreads.html) to predict
one of **8 genres**: children, comics & graphic, fantasy & paranormal,
history & biography, mystery/thriller/crime, poetry, romance, and young adult.




```bash
docker build -t mlops_assingment3 .
```

docker run --gpus all \
    -v $(pwd)/results:/workspace/results \
    -v $(pwd)/logs:/workspace/logs \
    mlops_assingment3 \
    python train.py

```bash
docker run --gpus all \
    -v $(pwd)/results:/workspace/results \
    -v $(pwd)/logs:/workspace/logs \
    -v $(pwd)/distilbert-uncased-goodreads-genres:/workspace/distilbert-uncased-goodreads-genres \
    mlops_assingment3 \
    python eval.py --local_only
```
```bash
docker run --gpus all \
    -e HF_TOKEN=hf_xxxxxxxxxxxx \
    -v $(pwd)/results:/workspace/results \
    -v $(pwd)/logs:/workspace/logs \
    -v $(pwd)/distilbert-uncased-goodreads-genres:/workspace/distilbert-uncased-goodreads-genres \
    mlops_assingment3 \
    python eval.py 
```


## Model Selection

**Model: `distilbert-base-uncased`**

| Criterion | Detail |
|-----------|--------|
| Parameters | 66M (vs 125M for RoBERTa-base, 110M for BERT-base) |
| GPU memory @ batch 32 | ~4 GB — fits a V100 16GB with room to spare |
| Training time | ~3 min/epoch on V100 |
| Accuracy vs BERT-base | Retains ~97% of performance at 60% of the size |
| Why uncased | Book reviews are informal; capitalisation carries little signal |
| Why not the original | Original notebook used `distilbert-base-**cased**` — this is a distinct model |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `distilbert-base-uncased` |
| Task | 8-class sequence classification |
| Max sequence length | 128 tokens |
| Epochs | 3 |
| Train batch size | 32 |
| Eval batch size | 64 |
| Learning rate | 2e-5 (AdamW, linear decay) |
| Warmup steps | 200 |
| Weight decay | 0.01 |
| Mixed precision | fp16 (GPU only) |
| Dataloader workers | 0 (stable on older kernels) |
| Best model metric | F1 (weighted) |

---

## Training Metrics

| Metric | Value |
|--------|-------|
| Train runtime | 168.45 s (~2.8 min) |
| Samples / second | 113.98 |
| Steps / second | 14.25 |
| Final train loss | **1.1018** |
| Epochs completed | 3 |
| Total FLOPs | 6.36 × 10¹⁴ |

---

## Label Mapping

| Genre | ID |
|-------|----|
| children | 0 |
| comics_graphic | 1 |
| fantasy_paranormal | 2 |
| history_biography | 3 |
| mystery_thriller_crime | 4 |
| poetry | 5 |
| romance | 6 |
| young_adult | 7 |

---

## Evaluation Results

### Per-class breakdown (1,600 test samples — 200 per genre)

| Genre | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| children | 0.745 | 0.745 | 0.745 | 200 |
| comics_graphic | 0.940 | 0.785 | 0.856 | 200 |
| fantasy_paranormal | 0.490 | 0.500 | 0.495 | 200 |
| history_biography | 0.689 | 0.620 | 0.653 | 200 |
| mystery_thriller_crime | 0.651 | 0.710 | 0.679 | 200 |
| poetry | 0.777 | 0.855 | 0.814 | 200 |
| romance | 0.662 | 0.655 | 0.658 | 200 |
| young_adult | 0.404 | 0.430 | 0.416 | 200 |
| **Weighted avg** | **0.670** | **0.663** | **0.665** | 1600 |

### Summary metrics

| Metric | Local model | Hub model |
|--------|-------------|-----------|
| Accuracy | 0.6625 | 0.6625 |
| F1 (weighted) | 0.6646 | 0.6646 |
| F1 (macro) | 0.6646 | 0.6646 |
| Test loss | 0.9893 | 0.9893 |

Local and Hub evaluations are **identical**, confirming correct serialisation.

### Notable observations

- **comics_graphic** achieved the highest F1 (0.856) — distinctive vocabulary (character names, visual storytelling terms) makes it easy to identify.
- **poetry** scored well on recall (0.855) — poetic language is lexically unique and separable from prose reviews.
- **young_adult** and **fantasy_paranormal** are the weakest classes (F1 ~0.42–0.50). This reflects genuine semantic overlap — YA novels frequently feature paranormal elements, making these two genres inherently difficult to separate from review text alone.


