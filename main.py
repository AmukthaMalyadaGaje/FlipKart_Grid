from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from api.expiry_extraction import router as expiry_router  # Import the expiry extraction router

app = FastAPI()

# Allow CORS (you can adjust this based on your needs)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the expiry extraction router
app.include_router(expiry_router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the expiry extraction API!"}
