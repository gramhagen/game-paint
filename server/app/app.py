import json
import uvicorn
from fastapi import FastAPI
# from image_generator import ImageGenerator

app = FastAPI()


@app.get("/")
def hello():
    """ Main page of the app. """
    return "Hello World!"


@app.get("/predict")
async def predict(input_text: str):
    """ Return JSON serializable output from the model """
    payload = {'input_text': input_text}
    # classifier = PythonPredictor("")
    # return classifier.predict(payload)
    return None

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)