import uvicorn
from fastapi import FastAPI
from app.router import router
import logging
from fastapi.logger import logger as fastapi_logger


app = FastAPI()
app.include_router(router)

gunicorn_logger = logging.getLogger("gunicorn")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8081)
else:
    logging.info("------------------ Application Started -------------------")
    fastapi_logger.setLevel(gunicorn_logger.level)