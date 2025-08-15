# 🎬 Movie Review Sentiment Classification

This repository contains the notebook `nb_14.ipynb` implementing an end‑to‑end sentiment classification workflow on an IMDB‑style movie reviews dataset (`imdb_reviews.tsv`). The project objective is to build a robust binary classifier that detects negative reviews with a target test F1 ≥ 0.85 while balancing accuracy, interpretability, and operational efficiency.

## 🚀 Key Features

* **Structured Experiment Ladder**: Baseline → TF‑IDF Logistic Regression → Lemmatized TF‑IDF → LightGBM → BERT CLS embeddings.
* **Unified Evaluation**: Single evaluation function outputs F1 vs threshold, ROC, PR curves, and tabular metrics (Accuracy, F1, ROC AUC, Average Precision).
* **Text Normalization & Lemmatization**: Lowercasing, punctuation/digit removal, optional spaCy lemmatization to reduce sparsity.
* **Multiple Representation Paths**: Sparse (TF‑IDF variants) and dense (BERT) feature pipelines for comparative analysis.
* **Custom Review Scoring**: Section to test arbitrary review snippets across trained models.
* **Rich Documentation**: In‑notebook graph and model summaries, running metrics table, comprehensive final conclusion.

## 🧾 Dataset Overview

`imdb_reviews.tsv` (tab‑delimited) includes:

| Column | Description |
|--------|-------------|
| review | Raw text review |
| pos | Binary sentiment label (1=positive, 0=negative) |
| rating | Star / numeric rating (auxiliary) |
| tconst | Movie identifier |
| start_year | Release year (temporal analysis) |
| ds_part | Pre-defined split (train / test) |

Characteristics: temporal polarity stability, long‑tail review count per title, balanced target distribution supporting F1 optimization without resampling.

## 📊 Visual & Analytical Outputs

| Visualization | Purpose |
|---------------|---------|
| Review & movie volume over time | Detect temporal drift & engagement trends |
| Reviews per movie distribution | Identify long‑tail & potential popularity bias |
| Rating distribution (train vs test) | Validate split alignment |
| Polarity over time & per title | Check label stability & per-title variability |
| Model evaluation panels | Compare threshold behavior, ranking quality |
| Prediction comparison table | Qualitative consistency across models |

## 🧪 Modeling Ladder

| Stage | Model | Features | Rationale |
|-------|-------|----------|-----------|
| 0 | Dummy (constant) | N/A | Prevalence baseline floor |
| 1 | Logistic Regression | TF‑IDF (stopwords removed) | Strong sparse baseline |
| 3 | Logistic Regression | Lemmatized TF‑IDF | Reduced sparsity, slight lift |
| 4 | LightGBM | Lemmatized TF‑IDF | Mild non‑linear gains |
| 9 | Logistic Regression | BERT CLS embeddings | Context & semantics upper bound |

(Exact metrics recorded inside notebook Running Summary.)

## 🧠 Key Findings

* Lemmatization yields modest but consistent F1 improvement vs raw tokens.
* LightGBM provides incremental ROC AUC / PR gains without large overfit.
* BERT offers additional semantic robustness; marginal lift may not justify CPU latency unless handling nuanced cases.
* Negation handling (loss of apostrophes) is a tunable improvement area (bigrams or preserving contractions).

## ✅ Recommended Production Baseline

Lemmatized TF‑IDF + Logistic Regression (simplicity, speed, interpretability) unless error analysis shows systematic contextual failures—then evaluate BERT variant with caching.

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Vyncent-vdW/sprint_fourteen_ml_for_text.git
cd sprint_fourteen_ml_for_text

# (Optional) create a virtual environment
python -m venv .venv
./.venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1

# Install core dependencies
pip install -r requirements.txt  # (create file as needed) OR
pip install pandas numpy scikit-learn lightgbm nltk spacy matplotlib seaborn tqdm transformers torch
python -m spacy download en_core_web_sm
```

## 🛠️ Usage

Run the modeling notebook:

```bash
jupyter notebook nb_14.ipynb
```

Typical workflow:
1. Execute cells top-to-bottom (recreates splits & functions if needed).
2. Review EDA outputs for drift / distribution sanity.
3. Train baseline TF‑IDF Logistic Regression.
4. Add lemmatization & LightGBM variant.
5. (Optional) Generate & cache BERT embeddings.
6. Populate Running Summary table with actual metrics.
7. Score custom reviews & compare model outputs.
8. Read final conclusion for deployment recommendations.

## 📐 Evaluation Methodology

* Primary metric: F1 (target ≥ 0.85)
* Supporting: ROC AUC, Average Precision, Accuracy
* Threshold exploration: F1 vs threshold curve; retain tuned threshold if deviating from default 0.5.
* Overfit check: Train vs test gap (< ~0.02 desirable early).

## 🧩 Dependencies

Core libraries:
```
pandas, numpy, scikit-learn, lightgbm, nltk, spacy, matplotlib, seaborn, tqdm, transformers, torch
```
Download model/language assets: `python -m spacy download en_core_web_sm`

## 🔄 Reproducibility & Caching

* Deterministic preprocessing pipeline (single normalization definition).
* Embeddings caching (e.g., `np.savez_compressed('features_9.npz', train_features_9=..., test_features_9=...)`).
* Notebook self-healing guards reinstantiate splits if cells run out of order.

## 🧪 Future Enhancements

| Category | Enhancement |
|----------|-------------|
| Feature Engineering | Add bigrams / retain contractions for negation |
| Modeling | Calibrated threshold selection; ensemble sparse + dense |
| Explainability | Coefficient report, SHAP (sparse baseline) |
| Monitoring | Drift dashboard (rolling polarity & F1) |
| Performance | Embedding cache & optional distillation |
| Quality | Error slicing by length, title frequency, year |

## 🙋 Contributing

Contributions welcome—performance tuning, feature engineering, evaluation extensions, or explainability improvements.

1. Fork the repository
2. Create a branch: `git checkout -b feature-name`
3. Commit: `git commit -m "Describe change"`
4. Push: `git push origin feature-name`
5. Open a Pull Request with before/after metrics if applicable

## 📄 License

Add a LICENSE file (MIT / Apache-2.0 recommended) if external distribution is intended.

## 🙏 Acknowledgments

* IMDB-style review corpus (educational context)
* Open-source NLP ecosystem: spaCy, scikit-learn, LightGBM, Transformers

## 🏁 Summary

An interpretable, high-performing Lemmatized TF‑IDF + Logistic Regression model serves as a pragmatic production baseline, with clear, data-driven pathways for semantic enhancement (BERT) and operational hardening (monitoring, caching, threshold calibration).

---
For questions or suggestions, please open an issue or start a discussion.
