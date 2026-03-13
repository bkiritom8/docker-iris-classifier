# Docker Lab — Iris Classifier with Random Forest

This lab demonstrates a multi-stage Docker workflow that trains a **scikit-learn Random Forest** classifier on the Iris dataset and serves predictions through a Flask web application.

## What Changed from the Original

The original lab used a TensorFlow/Keras neural network. This version replaces it with a **scikit-learn RandomForestClassifier**, which:

- Removes the heavy TensorFlow dependency (no GPU/CUDA required)
- Trains faster with no epoch tuning needed
- Persists the model with `joblib` (`.pkl`) instead of Keras's `.keras` format
- Bundles the `StandardScaler` with the model so training and inference preprocessing are always in sync

## Dataset

**Iris** (from `sklearn.datasets.load_iris`)

| Feature | Description |
|---|---|
| Sepal Length | Length of the sepal in cm |
| Sepal Width | Width of the sepal in cm |
| Petal Length | Length of the petal in cm |
| Petal Width | Width of the petal in cm |

**Classes:** Setosa, Versicolor, Virginica

## Model

| Property | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Library | scikit-learn |
| n_estimators | 100 |
| random_state | 42 |
| Saved as | `my_model.pkl` (via joblib) |

The scaler (`StandardScaler`) is saved together with the model inside the `.pkl` file so that inference always uses identical preprocessing.

## Project Structure

```
Docker_Labs/
├── dockerfile              # Multi-stage build: train → serve
├── docker-compose.yml      # Orchestrates training and serving services
├── requirements.txt        # Python dependencies (no TensorFlow)
├── HOWTO                   # Quick-start commands
├── README.md
└── src/
    ├── model_training.py   # Trains RandomForest, saves my_model.pkl
    ├── main.py             # Flask app serving predictions on port 4000
    ├── templates/
    │   └── predict.html    # Web UI for entering flower measurements
    └── statics/
        ├── setosa.jpeg
        ├── versicolor.jpeg
        └── virginica.jpeg
```

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- Verify Docker is running: `docker info`

## How to Run

### Option 1 — Single Docker image (recommended)

This uses a multi-stage Dockerfile: stage 1 trains the model, stage 2 serves it.

**Step 1: Clone the repo**
```bash
git clone https://github.com/bkiritom8/docker-iris-classifier.git
cd docker-iris-classifier
```

**Step 2: Build the image**
```bash
docker build -t iris-rf-app .
```

**Step 3: Run the container**
```bash
docker run -p 4000:4000 iris-rf-app
```

**Step 4: Open the web UI**

Go to `http://localhost:4000/predict` in your browser, enter the flower measurements, and click **ANALYZE**.

To stop the container: `Ctrl+C`

---

### Option 2 — Docker Compose

Runs training and serving as separate containers with a shared volume.

**Step 1: Clone the repo**
```bash
git clone https://github.com/bkiritom8/docker-iris-classifier.git
cd docker-iris-classifier
```

**Step 2: Start all services**
```bash
docker compose up
```

This spins up two services in order:
1. **model-training** — trains the Random Forest and writes `my_model.pkl` to the shared `model_exchange` volume
2. **serving** — waits for training to finish, then starts the Flask API on port 4000

**Step 3: Open the web UI**

Go to `http://localhost:4000/predict`

To stop: `Ctrl+C`, then `docker compose down`

---

### Option 3 — Test via curl (no browser needed)

```bash
curl -X POST http://localhost:4000/predict \
  -d "sepal_length=5.1&sepal_width=3.5&petal_length=1.4&petal_width=0.2"
```

**Sample values for each class:**

| Class | sepal_length | sepal_width | petal_length | petal_width |
|---|---|---|---|---|
| Setosa | 5.1 | 3.5 | 1.4 | 0.2 |
| Versicolor | 6.0 | 2.7 | 5.1 | 1.6 |
| Virginica | 6.3 | 3.3 | 6.0 | 2.5 |

**Expected response:**
```json
{"predicted_class": "Setosa"}
```

## API Reference

### `GET /predict`
Returns the prediction web UI.

### `POST /predict`

| Field | Type | Example |
|---|---|---|
| sepal_length | float | 5.1 |
| sepal_width | float | 3.5 |
| petal_length | float | 1.4 |
| petal_width | float | 0.2 |

**Response:**
```json
{"predicted_class": "Setosa"}
```

## Dependencies

```
scikit-learn
joblib
Flask
requests
```

No TensorFlow or Keras required.
