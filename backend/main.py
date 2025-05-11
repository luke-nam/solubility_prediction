from typing import Union

from fastapi import FastAPI, HTTPException

from app.schemas.prediction import PredictionRequest, ShapResponse, SolubilityResponse

app = FastAPI()

model_registry = {
    "solubility_model": "",
    "shap_model": "",
}

@app.get("/")
def root():
    return "Welcome to Solubility Predictor"

@app.post(
    "/predict/{model_name}", 
    response_model=Union[SolubilityResponse, ShapResponse]
)
def predict(model_name: str, request: PredictionRequest) -> None: 
    if model_name not in model_registry:
        raise HTTPException(status_code=404, detail="Model not found")

    match model_name:
        case "solubility_model": 
            #"Do something here"
            return SolubilityResponse(prediction=1)
        case "shap_model": 
            #"Do another thing here"
            return ShapResponse(prediction=1) 


