from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import Json
import os
import shutil
from src.models.predict import predict


app = FastAPI(title="Grapelead Disease Detection")

@app.get("/health")
def health():
    return { "status": "ok"}


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    temp_path = f"temp:{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict(temp_path)
    os.remove(temp_path)

    return JSONResponse(content=result)