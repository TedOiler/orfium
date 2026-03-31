# 📌 Predicting Future Video Views from Temporal Panel Data

This project builds a machine learning pipeline to predict the number of views a video will receive at a future time point, using repeated observations collected over time.

The model is designed to **improve its predictions as more observations become available**, leveraging the temporal structure of the data.

---

## Problem Description

The dataset consists of ~100k observations across ~10k videos. Each video appears multiple times with updated engagement statistics (views, likes, comments, etc.) collected at different timestamps.

Each row corresponds to a **partial observation of a video’s trajectory**, and the target is the video’s future view count.

The main challenges are:
- Handling the **panel structure** correctly
- Avoiding leakage across videos
- Ensuring predictions **improve with additional observations**

---

## Approach Overview

### Data Processing
- Sorted observations by `VIDEO_ID` and timestamp
- Created a within-video time index (`TIME_IDX`)
- Removed corrupted videos (zero duration with non-zero engagement)
- Engineered time-based features:
  - time since previous scrape
  - time since first scrape
  - video age at scrape
- Applied `log(1 + x)` transformation to reduce skew

### Modelling
- Grouped train/validation/test split by `VIDEO_ID` (prevents leakage)
- Models:
  - Ridge Regression (baseline)
  - Histogram Gradient Boosting (main model)
- Model selection based on **validation RMSE (log scale)**

### Evaluation
- Validation set used for model comparison and selection  
- Test set used **only once** for final performance reporting  
- Metrics computed on both:
  - log scale (primary)
  - original scale (interpretability)

---

## Repository Structure
```
.
├── data/
│ ├── raw/
│ └── processed/
├── outputs/
│ ├── models/
│ ├── tables/
│ └── figures/
├── src/
│ ├── data_processing.py
│ ├── feature_engineering.py
│ ├── split.py
│ ├── train_baseline.py
│ ├── train_boosted.py
│ ├── evaluation.py
│ ├── final_test_evaluation.py
│ ├── prefix_analysis.py
│ └── segment_analysis.py
├── run_pipeline.py
├── requirements.txt
└── README.md
```

---

## How to Run

Run the full pipeline:

```bash
python -m run_pipeline