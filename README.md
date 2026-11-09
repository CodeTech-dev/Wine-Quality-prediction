# Wine Quality Prediction

## ML Zoomcamp Midterm-Project

### Overview
This project predicts whether a wine will be rated **"good"** by tasters (quality score ≥ 7 out of 10) using only lab measurements such as alcohol, acidity and sulfur dioxide. It covers data preparation, exploratory data analysis (EDA), model training and tuning, and deployment as a web service with **FastAPI** and **Docker**.

The final model is a **Random Forest Classifier**, used with a tuned decision threshold of **0.32**.

**Problem and who benefits.** Wine quality is normally judged by human tasters, which is slow, expensive and subjective. A model that screens samples from cheap lab measurements helps **producers and quality-control teams** decide which batches to prioritize for expert tasting. It supports tasters; it does not replace them.

**Evaluation metric.** Only about 19% of wines are "good", so accuracy is misleading (predicting "not good" for every wine scores about 81%). The project uses **ROC-AUC** and **PR-AUC** to compare models, and **precision, recall and F1** at a tuned threshold.

---

### Features

**Input features (12):**
- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol
- Wine type (`red` or `white`)

**Output:**
- Probability that the wine is "good" (a value between 0 and 1)
- Prediction: `good` or `not good` (probability ≥ 0.32)

---

### Dataset
- **Source:** [UCI Machine Learning Repository: Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) (Vinho Verde wines from Portugal), also on Kaggle.
- **Files:** `winequality-red.csv` and `winequality-white.csv` (semicolon-separated). Download both and save them in the `data/` folder.
- **Size:** 6,497 wines (1,599 red + 4,898 white). After removing 1,177 duplicates, **5,320 wines** remain.
- **Target:** `good = 1` if `quality >= 7`, else `0`. Class balance: 81% not good / 19% good.
- **Split:** stratified 60 / 20 / 20 train / validation / test.

---

### EDA Summary
- **Alcohol is the strongest signal** (correlation +0.42 with `good`). Good wines have about 1.3 points more alcohol in both red and white.
- **Other effects depend on wine type.** Volatile acidity, citric acid, sulphates and total SO2 separate good from not-good *red* wines, while residual sugar and chlorides matter more for *white* wines.
- **Correlated features:** alcohol and density (-0.67), free and total sulfur dioxide (0.72).
- **No other single feature is strong** (correlations with `good` mostly below 0.16), so the achievable accuracy has a modest ceiling.

---

### Modeling and Results

Three models were compared: logistic regression (baseline), Random Forest and XGBoost, each tuned one parameter at a time, with Random Forest also tuned by `GridSearchCV`.

**5-fold cross-validation (training data):**

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic regression | 0.834 ± 0.004 | 0.529 ± 0.024 |
| **Random Forest** | **0.840 ± 0.010** | **0.557 ± 0.020** |
| XGBoost | 0.830 ± 0.010 | 0.534 ± 0.011 |

The models are close on ROC-AUC; the Random Forest was chosen because it leads on PR-AUC, the metric that matters most for imbalanced data.

**Final model:** Random Forest (`n_estimators=200`, `max_depth=None`, `min_samples_leaf=5`), trained on train + validation data.

**Test-set results** (evaluated once, `train.py` output):
- **ROC-AUC: 0.867**
- **PR-AUC: 0.605** (a model with no skill scores about 0.19)

**Decision threshold.** At the default 0.5 the model finds only about 30% of good wines. Tuning on the validation set gave **0.32**, which on the test set finds about **64%** of good wines, with about **53%** of its "good" flags correct (confusion matrix from the notebook: 130 found, 72 missed, 115 false alarms, 747 correct "not good").

**Feature importance.** Both Random Forest and XGBoost rank **alcohol first** by a wide margin. The rest contribute small, similar amounts, so quality depends on many weak signals.

---

### Project Structure

- **`notebook.ipynb`**:
  - Data preparation and cleaning
  - Exploratory Data Analysis (EDA) and feature importance
  - Model selection and hyperparameter tuning

- **`train.py`**:
  - Trains the final Random Forest
  - Saves the encoder, model and threshold to `model.bin` with `pickle`

- **`predict.py`**:
  - Loads `model.bin`
  - Function to predict the probability that a wine is good
  - FastAPI web service (`GET /health`, `POST /predict`)

- **`Dockerfile`**:
  - Instructions for containerizing the service

- **`pyproject.toml`** and **`uv.lock`**:
  - Project dependencies, managed with uv

- **`data/`**:
  - The two CSV files (see the Dataset section for the download steps)

---

### Prerequisites
- **Python** (version >= 3.11)
- **[uv](https://docs.astral.sh/uv/)** for dependency management
- **Docker** (for containerization)
- Dataset from the **[UCI Wine Quality page](https://archive.ics.uci.edu/dataset/186/wine+quality)**

---

### Setup and Execution

#### A. Operation Locally
##### Step 1: Clone the Repository
```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

##### Step 2: Install Dependencies
```bash
uv sync
```

##### Step 3: Add the Data
Download the two CSV files and save them as `data/winequality-red.csv` and `data/winequality-white.csv`.

##### Step 4: Train the Model
```bash
uv run python train.py
```
This prints the test ROC-AUC and PR-AUC and creates `model.bin`.

##### Step 5: Run the Service
```bash
uv run uvicorn serve:app --host 0.0.0.0 --port 9696
```
Interactive API docs are available at `http://localhost:9696/docs`.

#### B. Using Docker
##### Step 1: Build the Docker Image
```bash
docker build -t wine-quality .
```
##### Step 2: Run the Docker Container
```bash
docker run -it --rm -p 9696:9696 wine-quality
```
The image uses `python:3.11-slim-bookworm` and installs the exact versions in `uv.lock`. If you retrain the model, rebuild the image.

#### C. Cloud Deployment
> **TODO:** add the deployment URL (or a link to a video or screenshots of the deployed service), plus teardown notes if the platform charges for it.

---

### API Usage

- `GET /health` returns `{"status": "ok"}`.
- `POST /predict` accepts a JSON wine sample and returns the probability of "good" and the prediction.

Input is validated: a missing or misspelled field, a non-numeric value, an impossible value (such as a negative acidity) or a `wine_type` other than `red` / `white` returns a `422` error instead of a prediction.

**Example request:**
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{
           "fixed_acidity": 7.4,
           "volatile_acidity": 0.7,
           "citric_acid": 0.0,
           "residual_sugar": 1.9,
           "chlorides": 0.076,
           "free_sulfur_dioxide": 11.0,
           "total_sulfur_dioxide": 34.0,
           "density": 0.9978,
           "ph": 3.51,
           "sulphates": 0.56,
           "alcohol": 9.4,
           "wine_type": "red"
         }' \
     http://localhost:9696/predict
```

**Example response**
```json
{
  "good_probability": 0.641,
  "prediction": "good",
  "threshold": 0.32
}
```

---

### Project Deliverables
- A FastAPI web service for wine quality prediction.
- Docker container for easy deployment.
- Data preparation, EDA and model selection in `notebook.ipynb`.
- Trained model and scripts (`train.py`, `predict.py`, `serve.py`).

