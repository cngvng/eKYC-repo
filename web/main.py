from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
from utils import process_image


app = FastAPI()

origin_img = cv2.imread("/home/nyanmaruk/AIoT_Lab/eKYC/eKYC_flutter/khanh1.jpg")  # Replace with your image path

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post('/upload')
async def upload(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        result = process_image(origin_img, image_np)
        return {'result': result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
