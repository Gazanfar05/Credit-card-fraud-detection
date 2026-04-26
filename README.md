# Credit Card Fraud Detection

A machine learning project for detecting credit card fraud on imbalanced data using Random Forest with sampling strategy comparison.

---

## Project Summary

| Item | Detail |
|------|--------|
| **Dataset** | 6,000 transactions, 117 fraud cases |
| **Fraud Rate** | 1.95% |
| **Features** | 8 numeric (amount, time, f1–f6) |
| **Model** | Random Forest Classifier |
| **Best Strategy** | Baseline RF |
| **Best Threshold** | 0.3214 |
| **ROC-AUC** | 0.9968 |
| **Precision** | 0.9474 |
| **Recall** | 0.6207 |

---

## Pages

| Page | File | Description |
|------|------|-------------|
| Home | `index.html` | Project overview and model snapshot |
| How It Works | `how-it-works.html` | Pipeline explanation |
| Results | `results.html` | Model comparison with metrics |
| Predictor | `predictor.html` | Live transaction prediction |
| Analysis | `analysis.html` | EDA, sampling comparison, feature importance |

Navigate between pages using the top menu bar in the browser.

---

## File Structure

```
Credit-card-fraud-detection/
├── index.html              ← Home page
├── how-it-works.html       ← Pipeline explanation
├── results.html            ← Model evaluation results
├── predictor.html          ← Live fraud predictor
├── analysis.html           ← EDA and sampling analysis
├── style.css               ← Shared stylesheet
├── app.js                  ← Shared scripts
├── server.py               ← Local HTTPS server + prediction API
├── train_model.py          ← Model training script
├── artifacts/
│   ├── fraud_model.joblib  ← Trained model (generated after training)
│   └── model_summary.json  ← Training summary (generated after training)
└── .certs/
    ├── cert.pem            ← Self-signed certificate
    └── key.pem             ← Private key
```

---

## Getting Started

### 1. Install dependencies

```bash
pip install numpy pandas scikit-learn imbalanced-learn joblib
```

### 2. Train the model

```bash
python train_model.py
```

This generates `artifacts/fraud_model.joblib` and `artifacts/model_summary.json`.

### 3. Start the HTTPS server

```bash
python server.py
```

Server runs at `https://127.0.0.1:8443/`

### 4. Trust the self-signed certificate

Open `https://127.0.0.1:8443/` in your browser, click **Advanced → Proceed**, then navigate to the predictor.

### 5. Open the app

```
https://127.0.0.1:8443/index.html
```

> **Important:** Always open pages via the server URL, not by double-clicking HTML files. The predictor needs the server to run predictions.

---

## Model Results

### Sampling Strategy Comparison

| Strategy | Recall | Notes |
|----------|--------|-------|
| Baseline | 0.6207 | Best precision (0.9474), lowest noise |
| Undersampling | 0.9655 | Highest recall, more false positives |
| Oversampling | 0.6207 | Similar to baseline |
| SMOTE | 0.8276 | Balanced middle ground |

### Best Strategy — Baseline RF

| Metric | Value |
|--------|-------|
| Precision | 0.9474 |
| Recall | 0.6207 |
| ROC-AUC | 0.9968 |
| Threshold | 0.3214 |
| True Negatives | 1,469 |
| False Positives | 2 |
| False Negatives | 14 |
| True Positives | 25 |

### Feature Importance

| Feature | Importance |
|---------|-----------|
| amount | 0.4670 |
| f6 | 0.1782 |
| f4 | 0.0867 |
| f1 | 0.0816 |
| f3 | 0.0727 |

---

## Threshold Guidance

| Priority | Threshold | Outcome |
|----------|-----------|---------|
| Precision (low noise) | Higher than 0.3214 | Fewer false alerts, may miss some fraud |
| Balanced | 0.3214 (current) | Best precision/recall tradeoff |
| Recall (catch more fraud) | Lower than 0.3214 | More alerts, higher capture rate |

The right threshold depends on the cost of missed fraud versus the capacity for manual review.

---

## Live Predictor

Enter values for `amount`, `time`, and `f1`–`f6` on the Predictor page. The model returns:

- **Fraud** or **Genuine** label
- Fraud probability score
- Risk score (%)
- Threshold used for the decision

The server must be running for predictions to work.

---

## Key Concepts

**Why not use accuracy?**  
With only 1.95% fraud, a model that always predicts "Genuine" scores 98% accuracy but catches zero fraud. Precision, recall, and ROC-AUC are the meaningful metrics here.

**Why SMOTE and undersampling?**  
The fraud class is rare. Sampling techniques help the model learn from the minority class. Each technique makes a different precision/recall tradeoff.

**Why a tuned threshold?**  
The default 0.5 threshold is not optimal for imbalanced data. The trained threshold of 0.3214 was chosen to maximise useful fraud detection at manageable false positive rates.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `No module named 'imblearn'` | `pip install imbalanced-learn` |
| `ModuleNotFoundError: numpy._core` | `pip install --upgrade numpy scikit-learn joblib`, then retrain |
| `numpy.dtype size changed` | `pip uninstall numpy pandas -y && pip install numpy pandas`, then retrain |
| Predictor shows "Prediction unavailable" | Ensure server is running and cert is trusted in browser |
| `artifacts/fraud_model.joblib` missing | Run `python train_model.py` first |

---

**Status:** Complete  
**Last trained:** 1 day ago  
**Recommended action:** Use Baseline RF at threshold 0.3214 for production
