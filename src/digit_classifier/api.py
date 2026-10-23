"""Demo inference API. Not intended for production or sensitive use."""
from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
from sklearn.datasets import load_digits
from .modeling import validate_pixels

ROOT=Path(__file__).resolve().parents[2]
MODEL_PATH=Path(os.getenv('DIGITS_MODEL_PATH',str(ROOT/'models'/'model.joblib')))

class InputPixels(BaseModel):
    pixels: list[float] = Field(min_length=64,max_length=64,description='Row-major 8×8 grayscale intensities, from 0 to 16')

    @field_validator('pixels')
    @classmethod
    def check(cls,v):
        validate_pixels(v)
        return v

@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError('Model artifact not found. Run python -m digit_classifier.train first.')
    return joblib.load(MODEL_PATH)

app=FastAPI(title='Handwritten Digits — ML Portfolio Demo',version='1.0.0',
            description='Model comparison, error analysis, and an illustrative prediction API.')

@app.get('/')
def home():
    return FileResponse(ROOT/'web'/'index.html')

@app.get('/health')
def health():
    return {'status':'ok' if MODEL_PATH.exists() else 'model_not_trained'}

@app.get('/examples')
def examples():
    data=load_digits()
    return [{'label':int(i),'pixels':data.data[np.where(data.target==i)[0][0]].tolist()} for i in range(10)]

@app.post('/predict')
def predict(req:InputPixels):
    try:
        obj=load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503,detail=str(e)) from e
    x=np.asarray(req.pixels,dtype=float).reshape(1,64)
    probs=obj['estimator'].predict_proba(x)[0]
    label=int(obj['estimator'].classes_[np.argmax(probs)])
    return {'digit':label,'confidence':round(float(probs.max()),4),
            'probabilities':{str(int(k)):round(float(v),4) for k,v in zip(obj['estimator'].classes_,probs)},
            'model':obj['name'], 'caution':'Scores are not independently calibrated.'}
