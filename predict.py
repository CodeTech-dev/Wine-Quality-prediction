import pickle
from typing import Literal
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_FILE = 'model.bin'

with open(MODEL_FILE, 'rb') as f_in:
    dv, model, threshold = pickle.load(f_in)

app = FastAPI(title='Wine Quality Prediction')


class Wine(BaseModel):
    fixed_acidity: float = Field(ge=0)
    volatile_acidity: float = Field(ge=0)
    citric_acid: float = Field(ge=0)
    residual_sugar: float = Field(ge=0)
    chlorides: float = Field(ge=0)
    free_sulfur_dioxide: float = Field(ge=0)
    total_sulfur_dioxide: float = Field(ge=0)
    density: float = Field(gt=0)
    ph: float = Field(gt=0, lt=14)
    sulphates: float = Field(ge=0)
    alcohol: float = Field(ge=0, le=100)
    wine_type: Literal['red', 'white']


def predict_single(wine: dict) -> dict:
    X = dv.transform([wine])
    probability = float(model.predict_proba(X)[0, 1])
    return {
        'good_probability': round(probability, 3),
        'prediction': 'good' if probability >= threshold else 'not good',
        'threshold': threshold,
    }


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict')
def predict(wine: Wine):
    return predict_single(wine.model_dump())


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=9696)