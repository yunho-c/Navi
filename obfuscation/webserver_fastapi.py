from flask import Flask, request, flash, redirect, jsonify, url_for
# migrate: vanilla flask -> fastapi flask

from fastapi import FastAPI
from typing import Optional

import uvicorn
from fastapi import File, UploadFile
from starlette.responses import RedirectResponse, StreamingResponse
from PIL import Image
from io import BytesIO

from process_image import process_image
import numpy as np

import cv2


UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


app = FastAPI(title='Navi', description='Backend?')


@app.get('/', include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")


def read_image(image_encoded):
    image = np.array(Image.open(BytesIO(image_encoded)))[:,:,:3]
    print(image.shape)
    # print(image)
    return image


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.post('/api/process')
async def process(file: UploadFile = File(...)):
    if allowed_file(file.filename):
        img = read_image(await file.read())
        # return img.shape
        try: img = process_image(img)
        except Exception as e:
            print('ERROR:', e)
        # return Image.fromarray(img)
        # image_stream = image_bytes
        res, im_png = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return StreamingResponse(content=BytesIO(im_png), media_type="image/png")
        # return Response(content=).fromarray(img)


if __name__ == "__main__": 
    # uvicorn.run(app, port=8080, host='0.0.0.0')
    uvicorn.run(app, debug=True)