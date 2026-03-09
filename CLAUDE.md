# CLAUDE.md — Fake News Detection Project

This file provides context and conventions for AI assistants working in this repository.

## Project Overview

This is a **Python machine learning project** implementing a fake news detection pipeline using NLP techniques. The entire project lives in a single Jupyter notebook.

**Primary file:** `Fake_news_detection_using_NLP.ipynb`

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Environment | Jupyter Notebook |
| Data processing | pandas, numpy |
| Visualization | matplotlib, seaborn, wordcloud |
| ML models | scikit-learn (RandomForest, LogisticRegression, SVC) |
| Deep learning | TensorFlow / Keras (LSTM) |
| Text features | TF-IDF (`TfidfVectorizer`), `Tokenizer` |
| Model saving | joblib, pickle, `.h5` (Keras) |

No web framework, database, REST API, or CI/CD pipeline exists in this project.

---

## Repository Structure

```
fake_news/
├── CLAUDE.md                                  # This file
├── README.md                                  # Brief project description
└── Fake_news_detection_using_NLP.ipynb       # Complete ML pipeline
```

The notebook is self-contained. There is no `src/`, `tests/`, or `scripts/` directory.

---

## Data Dependencies

The notebook requires two CSV files that are **not committed to the repository** and must be obtained separately:

| File | Description |
|---|---|
| `Fake.csv` | 23,481 fake news articles |
| `True.csv` | 21,417 true news articles |

**Schema** (both files share the same columns):
```
title      - Article headline (string)
text       - Article body (string)
subject    - News category (string): politics, News, left-news, politicsNews
date       - Publication date (string)
```

A `class` label column is added during preprocessing: `0 = fake`, `1 = true`.

---

## ML Pipeline — Step by Step

The notebook follows this exact workflow:

1. **Load data** — Read `Fake.csv` and `True.csv`, assign class labels, concatenate.
2. **Balance classes** — Upsample the minority class (fake news) to match true news count (42,834 total).
3. **Feature engineering**
   - Encode `subject` column → `subject_encoded` (integer)
   - Compute `text_length` (character count of article text)
   - Build TF-IDF matrix from `text` column
4. **Feature selection** — Pearson correlation filter; keep features with |correlation| > 0.05 against the target.
5. **Scaling** — `StandardScaler` on numeric features.
6. **Train/test split** — 80% train / 20% test, `random_state=42`.
7. **K-Fold CV** — 5-fold cross-validation for all models.
8. **Hyperparameter tuning** — `GridSearchCV` with 5-fold CV.
9. **Evaluation** — accuracy, precision, recall, F1-score via `classification_report`.
10. **Serialization** — Save models to disk (joblib/pickle for sklearn, `.h5` for Keras LSTM).

---

## Models Implemented

### 1. Random Forest Classifier
- Default: `n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2`
- Tuned best params: `criterion='gini', max_depth=None, min_samples_split=2, n_estimators=100`
- Reported test accuracy: **100%**

### 2. Logistic Regression
- Default: `penalty='l2', C=1.0, solver='lbfgs', max_iter=1000`
- Tuned best params: `C=0.01, penalty='l2', solver='lbfgs'`
- Reported test accuracy: **100%**

### 3. Support Vector Machine (SVC)
- Default: `C=1.0, kernel='rbf', gamma='scale'`
- Tuned best params: `C=0.1, kernel='linear', gamma='scale'`
- Reported test accuracy: **100%**

### 4. LSTM Neural Network (Deep Learning)
- Architecture: `Embedding → LSTM(128) → Dense(1, sigmoid)`
- Training: 5 epochs, batch size 64, 20% validation split
- Dropout: 0.2 on LSTM layer

> **Note:** All three classical models report 100% accuracy, which likely reflects the highly separable nature of the dataset's features (subject category is a near-perfect predictor) rather than generalizable performance.

---

## Key Conventions

- **Random seed:** Always use `random_state=42` for reproducibility across train/test splits and model initialization.
- **Label encoding:** `0 = fake`, `1 = true` — do not reverse this convention.
- **Class balancing:** Use upsampling (not downsampling) to handle class imbalance.
- **Feature correlation threshold:** Features with |Pearson r| ≤ 0.05 against the target are dropped.
- **Scaling:** Apply `StandardScaler` after the train/test split (fit on train, transform both).
- **No external data in repo:** CSV data files must be sourced externally; never commit large data files.

---

## Development Workflow

This project has no automated tests, linter, or build system. The development workflow is:

1. Ensure `Fake.csv` and `True.csv` are available in the working directory.
2. Open `Fake_news_detection_using_NLP.ipynb` in Jupyter or a compatible environment (VS Code, JupyterLab, Google Colab).
3. Run all cells in order (`Kernel > Restart & Run All`).
4. Inspect output metrics and saved model files.

### Installing dependencies

There is no `requirements.txt`. Install the necessary libraries manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow wordcloud joblib
```

---

## Git Branches

| Branch | Purpose |
|---|---|
| `master` | Main branch with original uploaded files |
| `claude/claude-md-*` | AI assistant working branches |

---

## What Does Not Exist (yet)

The following are **absent** from this repo — do not assume they exist:

- `requirements.txt` or `pyproject.toml`
- Unit tests or test directory
- CI/CD pipeline (no `.github/workflows/`)
- Docker configuration
- REST API or web application
- Logging configuration
- Environment variable files (`.env`)
- Pre-commit hooks
