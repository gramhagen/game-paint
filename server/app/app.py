from io import BytesIO
import json
from pathlib import Path
from uuid import uuid4

from typing import Optional

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, status
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel
import uvicorn

from model.vqgan_clip import load_model, load_perceptor, generate


IMAGE_PATH = Path("/images")
TOKEN = "910350ecee704db58c6a8abe6bb96fb1"


class TextPrompt(BaseModel):
    prompt: str


class ImageRef(BaseModel):
    image_id: str


app = FastAPI()
model = load_model()
perceptor = load_perceptor()


def validate_token(token):
    if token != TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect Token"
        )

@app.get("/")
def get_status():
    return "Server is up!"


@app.post("/predict")
async def predict(text: TextPrompt, tasks: BackgroundTasks, token: Optional[str] = Header(None)):
    """ Return image id for model output """
    validate_token(token)
    ref = ImageRef(image_id=uuid4().hex)
    image_dir = IMAGE_PATH.joinpath(ref.image_id)
    image_dir.mkdir()
    image_path = image_dir.joinpath("1.png")
    tasks.add_task(
        generate,
        model=model,
        perceptor=perceptor,
        output_path=image_path.as_posix(),
        prompts=f"{text.prompt}|unreal engine",
        cuda_device="cuda:0",
    )
    return ref


@app.post("/retrieve",
    # Set what the media type will be in the autogenerated OpenAPI specification.
    # fastapi.tiangolo.com/advanced/additional-responses/#additional-media-types-for-the-main-response
    responses = {
        200: {
            "content": {"image/png": {}}
        }
    },
    # Prevent FastAPI from adding "application/json" as an additional
    # response media type in the autogenerated OpenAPI specification.
    # https://github.com/tiangolo/fastapi/issues/3258
    response_class=Response,
)
async def retrieve(ref: ImageRef, token: Optional[str] = Header(None)):
    """ Return generated image from the model """
    validate_token(token)
    image_byte_array = BytesIO()
    image = Image.open(IMAGE_PATH.joinpath(ref.image_id, "1.png"))
    image.save(image_byte_array, format=image.format)
    return Response(content=image_byte_array.getvalue(), media_type="image/png")


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)