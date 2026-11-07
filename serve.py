from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

import uvicorn
from predict import predict_single

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


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict')
def predict(wine: Wine):
    return predict_single(wine.model_dump())


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=9696)