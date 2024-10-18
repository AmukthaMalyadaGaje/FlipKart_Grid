from pydantic import BaseModel


class LabelResponse(BaseModel):
    label: str

class ExpiryResponse(BaseModel):
    expiry_date: str
    raw_text: str



class BrandRecognitionResponse(BaseModel):
    brand: str


class CountResponse(BaseModel):
    count: int


class FreshnessResponse(BaseModel):
    freshness_score: float


class BrandResponseModel(BaseModel):
    brand: str
