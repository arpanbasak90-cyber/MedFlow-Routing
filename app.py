from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.get("/")
def home():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))
