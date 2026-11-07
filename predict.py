import pickle

MODEL_FILE = 'model.bin'

with open(MODEL_FILE, 'rb') as f_in:
    dv, model, threshold = pickle.load(f_in)


def predict_single(wine: dict) -> dict:
    X = dv.transform([wine])
    probability = float(model.predict_proba(X)[0, 1])
    return {
        'good_probability': round(probability, 4),
        'prediction': 'good' if probability >= threshold else 'not good',
        'threshold': threshold,
    }