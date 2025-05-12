from pydantic import BaseModel


class PredictionRequest(BaseModel):
    smiles: str

class SolubilityResponse(BaseModel): 
    prediction: str 
    probabilities: list[float]

class ShapResponse(BaseModel):
    Feature: str
    SHAP_Value: float
    Feature_Value: float
    Abs_SHAP_Value: float