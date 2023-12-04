import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

FOODS = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot',
 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic',
 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion',
 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato',
 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato',
 'turnip', 'watermelon']

model_path = os.path.join('food-classification', 'INPUT_model_path', 'food-cnn')
model = load_model(model_path)

@app.get("/")
async def root():
    return {"message": "Goeindag"}

@app.post('/upload/image')
async def uploadImage(img: UploadFile = File(...)):
    original_image = Image.open(img.file)
    original_image = original_image.resize((100, 100))
    # If image is png, convert it to jpg
    if original_image.mode == 'RGBA':
        original_image = original_image.convert('RGB')
    images_to_predict = np.expand_dims(np.array(original_image), axis=0)
    predictions = model.predict(images_to_predict)
    classification = predictions.argmax(axis=1)

    return FOODS[classification.tolist()[0]]

@app.get("/healthcheck")
def healthcheck():
    return {"status": "Healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
