from pydantic import BaseModel


class PredictionRequest(BaseModel):
    smiles: str

class SolubilityResponse(BaseModel): 
    prediction: str 
    probabilities: list[float]

class ShapResponse(BaseModel):
    Feature: str
    Feature_Value: float
    Influence: float
    Contribution: str 
