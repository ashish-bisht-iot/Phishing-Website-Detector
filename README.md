# 🎣 Phishing Website Detector

A machine learning-powered REST API that analyzes URLs and predicts whether they are **phishing**, **suspicious**, or **safe** — with a detailed feature breakdown for each prediction.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Train the Model](#1-train-the-model)
  - [Run the API](#2-run-the-api)
  - [API Endpoints](#3-api-endpoints)
- [Feature Engineering](#feature-engineering)
- [Model Architecture](#model-architecture)
- [Example Response](#example-response)
- [Deployment](#deployment)
- [Dependencies](#dependencies)
- [License](#license)

---

## Overview

**Phishing Website Detector** extracts 29 URL-based features (protocol, domain age, TLD suspiciousness, keyword presence, DNS resolution, etc.) and runs them through a trained **Random Forest + Gradient Boosting ensemble** to return a risk score and verdict in real time.

The project ships with:
- A **Flask REST API** (`app.py`) ready for production via Gunicorn
- A **feature extractor** (`feature_extractor.py`) with 29 hand-crafted signals
- A **training pipeline** (`train_model.py`) supporting the UCI Phishing Dataset or synthetic fallback data
- An optional **frontend** served from `templates/index.html`

---

## Features

- ✅ Classifies URLs as **Safe**, **Suspicious**, or **Phishing**
- 📊 Returns a **risk score (0–100)** and phishing probability
- 🔍 Provides a **human-readable feature breakdown** per prediction
- 🌐 Optional **WHOIS domain age** lookup
- 🔒 HTTPS detection, IP-in-URL check, suspicious TLD matching
- ⚡ Sub-second inference after model load

---

## Project Structure

```
phishing-detector/
│
├── app.py                  # Flask REST API
├── feature_extractor.py    # URL feature engineering (29 features)
├── train_model.py          # Model training pipeline
├── requirements.txt        # Python dependencies
│
├── model/
│   └── phishing_model.pkl  # Saved model (generated after training)
│
├── data/
│   └── phishing_dataset.csv  # UCI dataset (optional — see Training)
│
└── templates/
    └── index.html          # Optional frontend UI
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/phishing-detector.git
cd phishing-detector

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Train the Model

```bash
python train_model.py
```

This will:
- Load `data/phishing_dataset.csv` if present (UCI Phishing Dataset)
- Fall back to **synthetic training data** if the file is not found
- Save the trained model to `model/phishing_model.pkl`
- Print classification report, confusion matrix, ROC-AUC, and 5-fold CV F1

> **Using the real dataset (recommended):**  
> Download from [UCI ML Repository](https://archive.ics.uci.edu/dataset/327/phishing+websites) and place the CSV at `data/phishing_dataset.csv`.

---

### 2. Run the API

```bash
# Development
python app.py

# Production (Gunicorn)
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

The server starts at `http://localhost:5000`.

---

### 3. API Endpoints

#### `POST /predict`

Analyzes a URL and returns a phishing verdict.

**Request:**
```json
{
  "url": "http://paypa1-secure-login.xyz/account/verify?user=abc"
}
```

**Response:**
```json
{
  "url": "http://paypa1-secure-login.xyz/...",
  "prediction": 1,
  "risk_level": "phishing",
  "risk_score": 91.3,
  "phishing_probability": 0.9130,
  "verdict_message": "High phishing risk! Multiple red flags detected — do not enter credentials.",
  "features": [ ... ],
  "raw_features": { ... }
}
```

| `risk_level` | Score Range | Meaning |
|---|---|---|
| `safe` | 0 – 29 | No significant signals |
| `suspicious` | 30 – 64 | Some red flags — proceed with caution |
| `phishing` | 65 – 100 | High risk — do not trust |

---

#### `GET /health`

Returns API and model status.

```json
{
  "status": "ok",
  "model_ready": true,
  "model_path": "/path/to/model/phishing_model.pkl"
}
```

#### `GET /`

Serves the frontend UI (if `templates/index.html` exists), otherwise returns a JSON welcome message.

---

## Feature Engineering

The extractor derives **29 numeric features** from each URL across five categories:

| Category | Features |
|---|---|
| **URL structure** | Length, digit ratio, special char count, slash/dot/hyphen counts |
| **Protocol & security** | HTTPS presence, `@` symbol, double-slash in path, redirect params |
| **Domain signals** | IP-in-URL, subdomain depth, suspicious TLD, trusted domain match |
| **Keyword signals** | Phishing keyword count, brand keyword presence |
| **Live signals** | DNS resolution, WHOIS domain age (days) |

See `feature_extractor.py` for the full list and logic.

---

## Model Architecture

The classifier is a **soft-voting ensemble** trained via scikit-learn:

| Component | Algorithm | Weight |
|---|---|---|
| Primary | Random Forest (200 trees, depth 15) | 2 |
| Secondary | Gradient Boosting (150 estimators, lr=0.08) | 1 |

The pipeline applies `StandardScaler` before the ensemble. Training uses stratified 80/20 split with 5-fold cross-validation reporting.

---

## Example Response

```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"url": "https://google.com"}'
```

```json
{
  "risk_level": "safe",
  "risk_score": 4.2,
  "verdict_message": "This URL appears legitimate with no major phishing signals detected."
}
```

---

## Deployment

### Docker (recommended)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN python train_model.py
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

```bash
docker build -t phishing-detector .
docker run -p 5000:5000 phishing-detector
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `FLASK_ENV` | `production` | Set to `development` for debug mode |
| `PORT` | `5000` | Port for Gunicorn |

---

## Dependencies

| Package | Purpose |
|---|---|
| `flask` | REST API framework |
| `flask-cors` | Cross-origin request support |
| `scikit-learn` | ML model training & inference |
| `numpy` / `pandas` | Data processing |
| `python-whois` | Domain age lookup (optional) |
| `tldextract` | Accurate TLD parsing |
| `gunicorn` | Production WSGI server |

Install all with:
```bash
pip install -r requirements.txt
```

---

## License

This project is licensed under the **MIT License**. See `LICENSE` for details.

---

> Built with ❤️ using Flask, scikit-learn, and 29 hand-crafted URL features.
