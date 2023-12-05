import uvicorn
import argparse
from typing import Callable
from fastapi import FastAPI
from app.router import router
from app.config import (
    IMAGE_PATH_PREFIX
)
from app import logic
import logging
from fastapi.logger import logger as fastapi_logger
from contextlib import asynccontextmanager
import time

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="Model type")


def load_model(model_load_fn: Callable, model_type: str):
    start = time.time()
    model_load_fn()
    end = time.time()
    with open(f"model_load_time.txt", "a") as f:
        f.write(f"Model type: {model_type}, Load time: {end - start}\n")


@asynccontextmanager
async def lifespan(app: FastAPI):

    for i in range (1, 7):
        logic._get_transformed_image(IMAGE_PATH_PREFIX + f"{i}.jpg")

    model = parser.parse_args().model

    match model:
        case 'torch':
            load_model(logic._get_torch_model, model)
        case 'onnx':
            load_model(logic._get_onnx_model, model)
        case 'keras':
            load_model(logic._get_keras_model, model)
        case _:
            raise Exception("Model type not supported")
    yield
    logic._get_torch_model.cache_clear()
    logic._get_onnx_model.cache_clear()
    logic._get_keras_model.cache_clear()
    logic._get_transformed_image.cache_clear()


app = FastAPI(lifespan=lifespan)
app.include_router(router)

gunicorn_logger = logging.getLogger("gunicorn")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8081)
else:
    logging.info("------------------ Application Started -------------------")
    fastapi_logger.setLevel(gunicorn_logger.level)