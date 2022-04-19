from flask import Flask, request, flash, redirect, jsonify, url_for
# migrate: vanilla flask -> fastapi flask

from fastapi import FastAPI
from typing import Optional

import uvicorn
from fastapi import File, UploadFile
from starlette.responses import RedirectResponse
from PIL import Image
from io import BytesIO

from process_image import process_image
import numpy as np


UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


app = FastAPI(title='Navi', description='Backend?')


@app.get('/', include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")


async def read_image(image_encoded):
    image = np.array(Image.open(BytesIO(image_encoded)))
    print(image)
    return image


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.post('/api/process')
async def process(file: UploadFile = File(...)):
    if allowed_file(file.filename):
        img = read_image(file.read())
        return img.shape
        img = process_image(img)
        return Image.fromarray(img)


if __name__ == "__main__": 
    # uvicorn.run(app, port=8080, host='0.0.0.0')
    uvicorn.run(app, debug=True)