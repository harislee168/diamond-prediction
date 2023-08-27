from pydantic import BaseModel
from typing import List

class DiamondInitialValue(BaseModel):
    min_carat: float
    max_carat: float
    min_y: float
    max_y: float
    clarities: List
    colors: List
