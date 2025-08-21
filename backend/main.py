from fastapi.responses import FileResponse
import os
from fastapi import FastAPI, UploadFile, File

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    return {"info": f"file '{file.filename}' saved at '{file_location}'"}

@app.get("/image/{filename}")
def get_image(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"error": "File not found"}
    


