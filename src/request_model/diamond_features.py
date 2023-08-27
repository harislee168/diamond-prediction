from pydantic import BaseModel

class DiamondFeatures(BaseModel):
    carat: float
    y: float
    clarity: str
    color: str
