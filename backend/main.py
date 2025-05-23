from io import BytesIO
from typing import List, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.schemas.prediction import PredictionRequest, ShapResponse, SolubilityResponse
from app.services.predictor import (
    display_structure,
    predict_label,
    predict_probs,
    predict_shap,
)

app = FastAPI()

# Allow your frontend origin during development
origins = [
    "http://localhost:5173",  # Vite dev server
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # Can also use ["*"] to allow all
    allow_credentials=True,
    allow_methods=["*"],              # Allow all HTTP methods
    allow_headers=["*"],              # Allow all headers
)

model_registry = {
    "solubility_model": "",
    "shap_model": "",
}

@app.get("/")
def root():
    return "Welcome to Solubility Predictor"

@app.get("/structure")
def get_structure(smiles: str) -> StreamingResponse: 
    img = display_structure(smiles)    

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")

@app.post(
    "/predict/{model_name}", 
    response_model=Union[SolubilityResponse, List[ShapResponse]]
)
def predict(
    model_name: str, 
    request: PredictionRequest
) -> Union[SolubilityResponse, List[ShapResponse]]: 
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


