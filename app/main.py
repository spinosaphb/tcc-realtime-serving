import uvicorn
from fastapi import FastAPI
from app.router import router
from app import logic
import logging
from fastapi.logger import logger as fastapi_logger
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    logic._get_torch_model()
    logic._get_onnx_model()
    logic._get_keras_model()
    yield
    logic._get_torch_model.cache_clear()
    logic._get_onnx_model.cache_clear()
    logic._get_keras_model.cache_clear()


app = FastAPI(lifespan=lifespan)
app.include_router(router)

gunicorn_logger = logging.getLogger("gunicorn")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8081)
else:
    logging.info("------------------ Application Started -------------------")
    fastapi_logger.setLevel(gunicorn_logger.level)