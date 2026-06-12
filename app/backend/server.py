import sys
import asyncio
import logging
from pathlib import Path

#add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel

from model import My_Translator_Model

#logging
LOG_PATH = Path("data/logs/log_file.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

#app setup
app = FastAPI(title="Akkadian Translator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#load model once at startup
translator = My_Translator_Model()

@app.on_event("startup")
async def startup_event():
    logger.info("Loading translation model...")
    translator._load_model()
    logger.info("Model ready.")


#request schema
class TranslateRequest(BaseModel):
    text: str


#endpoints
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/translate")
async def translate(request: TranslateRequest):
    """
    Translate Akkadian text, stream tokens via SSE.
    Frontend reads 'data: {"token": "word "}' events.
    """
    text = request.text.strip()
    logger.info(f"Translate request: {text[:80]}")

    async def token_generator():
        try:
            #run model inference in thread pool (it's blocking)
            loop = asyncio.get_event_loop()

            def run_translation():
                return list(translator.predict(text, stream=True))

            tokens = await loop.run_in_executor(None, run_translation)

            for token in tokens:
                yield {"data": f'{{"token": "{token.strip()}"}}'}
                await asyncio.sleep(0.03)  #small delay for visible streaming effect

            yield {"data": "[DONE]"}

        except Exception as e:
            logger.error(f"Translation error: {e}")
            yield {"data": f'{{"error": "{str(e)}"}}'}
            yield {"data": "[DONE]"}

    return EventSourceResponse(token_generator())


#serve frontend
frontend_path = Path(__file__).parent.parent / "frontend"

if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(frontend_path / "index.html"))

#run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )