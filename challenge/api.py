
import joblib

import pandas as pd

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request

from challenge.api_schemas import PredictionRequest
from challenge.settings import MODEL_NAME


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.delay_model = joblib.load(f'models/{MODEL_NAME}')
    print("Model loaded")

    yield

    print("App finished")

app = FastAPI(lifespan=lifespan)

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(request: Request, body: PredictionRequest) -> dict:

    delay_model = request.app.state.delay_model

    flights = body.flights
    pred_df = pd.DataFrame([f.model_dump() for f in flights])

    X = delay_model.preprocess(pred_df)

    # Realizar predicción
    y_pred = delay_model.predict(X)

    return {'predict' : y_pred}
