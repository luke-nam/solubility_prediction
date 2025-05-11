from pydantic import BaseModel


class PredictionRequest(BaseModel):
    input: str

class SolubilityResponse(BaseModel): 
    prediction: int

class ShapResponse(BaseModel):
    prediction: float 