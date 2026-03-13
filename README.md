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

## Running with Docker (single image)

```bash
# Build the multi-stage image (stage 1 trains, stage 2 serves)
docker build -t iris-rf-app .

# Run the serving container
docker run -p 4000:4000 iris-rf-app
```

Open your browser at `http://localhost:4000/predict`.

## Running with Docker Compose

```bash
docker compose up
```

This spins up two services:
1. **model-training** — trains the Random Forest and writes `my_model.pkl` to the shared `model_exchange` volume
2. **serving** — waits for training to complete, then loads the model and starts the Flask API on port 4000

## API

### `GET /predict`
Returns the prediction web UI.

### `POST /predict`

**Form fields:**

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
