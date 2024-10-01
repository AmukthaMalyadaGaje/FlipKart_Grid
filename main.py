from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from api.label_extraction import extract_labels
import tempfile

app = FastAPI()

# Allow CORS (you can adjust this based on your needs)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/extract-labels/")
async def extract_labels_from_image(file: UploadFile = File(...)):
    # Create a temporary file to store the uploaded image
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        tmp.write(await file.read())
        tmp.flush()  # Ensure the data is written to disk
        
        try:
            labels = extract_labels(tmp.name)
            return JSONResponse(content={"labels": labels})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# To run the application, use the command:
# uvicorn main:app --reload
