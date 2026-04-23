from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from pathlib import Path

from .asr import transcribe
from .tts import synthesize

app = FastAPI(title="Voice Agent API")

# mount static frontend
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def index():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return HTMLResponse(index_file.read_text(encoding='utf-8'))
    return HTMLResponse("<h1>Voice Agent API</h1><p>No frontend available.</p>")


@app.post("/asr")
async def asr_endpoint(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".flac")):
        raise HTTPException(status_code=400, detail="Unsupported audio format")
    tmp = Path("/tmp") / f"agent_upload_{file.filename}"
    with tmp.open("wb") as f:
        f.write(await file.read())
    text = transcribe(str(tmp))
    try:
        tmp.unlink()
    except Exception:
        pass
    return JSONResponse({"text": text})


@app.post("/tts")
async def tts_endpoint(payload: dict):
    text = payload.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' field")
    try:
        wav = synthesize(text)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    return Response(content=wav, media_type="audio/wav")


@app.post("/agent")
async def agent_endpoint(file: UploadFile = File(...)):
    # simple pipeline: audio -> asr -> echo reply -> tts
    tmp = Path("/tmp") / f"agent_upload_{file.filename}"
    with tmp.open("wb") as f:
        f.write(await file.read())
    text = transcribe(str(tmp))
    try:
        tmp.unlink()
    except Exception:
        pass
    # simple agent logic
    reply = "I heard: " + text
    wav = synthesize(reply)
    return Response(content=wav, media_type="audio/wav")


if __name__ == "__main__":
    uvicorn.run("agent.server:app", host="0.0.0.0", port=8000, reload=False)
