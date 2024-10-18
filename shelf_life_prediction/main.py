from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.expiry_extraction import router as expiry_router  # Expiry extraction router
from api.freshness_prediction import router as freshness_router  # Freshness prediction router
from api.label_extraction import router as label_router  # Label extraction router

app = FastAPI()

# Allow CORS (adjust based on your needs)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to specific domains if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the routers
app.include_router(expiry_router, prefix="/expiry")
app.include_router(freshness_router, prefix="/freshness")
app.include_router(label_router, prefix="/label")  # Make sure this matches the label router in label_extraction.py

@app.get("/")
def read_root():
    return {"message": "Welcome to the expiry extraction API!"}
