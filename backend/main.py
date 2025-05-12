from typing import Union, List

from fastapi import FastAPI, HTTPException

from app.schemas.prediction import PredictionRequest, ShapResponse, SolubilityResponse
from app.services.predictor import predict_label, predict_probs, predict_shap

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
    response_model=Union[SolubilityResponse, List[ShapResponse]]
)
def predict(model_name: str, request: PredictionRequest) -> Union[SolubilityResponse, List[ShapResponse]]: 
    if model_name not in model_registry:
        raise HTTPException(status_code=404, detail="Model not found")

    match model_name:
        case "solubility_model": 
            pred = predict_label(request.smiles)
            probs = predict_probs(request.smiles)
            return SolubilityResponse(prediction=pred, probabilities=probs)
        
        case "shap_model": 
            pred = predict_shap(request.smiles)
            return [ShapResponse(**row) for row in pred] 


