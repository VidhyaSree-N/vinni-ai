from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from vinni.agent import ask_vinni
from vinni.voice import transcribe, speak
from dotenv import load_dotenv
import tempfile
import os
import uuid
import time
import asyncio
import base64
import urllib.parse
import httpx

load_dotenv()

DID_API_KEY  = os.getenv("DID_API_KEY")
DID_PHOTO_URL = os.getenv("DID_PHOTO_URL")  # https://i.imgur.com/KCZDEcA.jpeg

app = FastAPI(
    title="Vinni AI",
    description="Personal AI agent for Vidhya Sree Narayanappa",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Transcription", "X-Reply", "X-DID-Video"]
)

conversations: dict = {}

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

class AskRequest(BaseModel):
    question: str
    session_id: str = "default"

class AskResponse(BaseModel):
    answer: str
    session_id: str

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    if request.session_id not in conversations:
        conversations[request.session_id] = []
    reply, updated = ask_vinni(request.question, conversations[request.session_id])
    conversations[request.session_id] = updated
    return AskResponse(answer=reply, session_id=request.session_id)

@app.post("/voice")
async def voice(
    audio: UploadFile = File(...),
    session_id: str = Form(default="default"),
    use_did: str = Form(default="false")
):
    t0 = time.time()

    # save + transcribe
    suffix = os.path.splitext(audio.filename)[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    question = transcribe(tmp_path)
    os.unlink(tmp_path)
    print(f"⏱ Whisper: {time.time()-t0:.2f}s | You said: {question}")

    # agent
    t1 = time.time()
    if session_id not in conversations:
        conversations[session_id] = []
    reply, updated = ask_vinni(question, conversations[session_id])
    conversations[session_id] = updated
    print(f"⏱ Agent: {time.time()-t1:.2f}s | Vinni: {reply[:80]}")

    # TTS — Resemble generates your voice
    t2 = time.time()
    out = f"output_{uuid.uuid4().hex}.wav"
    speak(reply, output_path=out)
    with open(out, "rb") as f:
        audio_bytes = f.read()
    os.unlink(out)
    print(f"⏱ TTS: {time.time()-t2:.2f}s")

    # D-ID — animate your face with your voice
    did_video_url = None
    if use_did == "true" and DID_API_KEY and DID_PHOTO_URL:
        try:
            t3 = time.time()
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            async with httpx.AsyncClient(timeout=60) as c:
                # create talk
                res = await c.post(
                    "https://api.d-id.com/talks",
                    headers={
                        "Authorization": f"Basic {DID_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "source_url": DID_PHOTO_URL,
                        "script": {
                            "type": "audio",
                            "audio_url": f"data:audio/wav;base64,{audio_b64}"
                        },
                        "config": {"fluent": True, "pad_audio": 0}
                    }
                )
                talk = res.json()
                print(f"D-ID full response: {talk}")  # ADD THIS LINE
                talk_id = talk.get("id")
                print(f"D-ID talk created: {talk_id}")

                # poll for result
                if talk_id:
                    for _ in range(30):
                        await asyncio.sleep(1)
                        sr = await c.get(
                            f"https://api.d-id.com/talks/{talk_id}",
                            headers={"Authorization": f"Basic {DID_API_KEY}"}
                        )
                        s = sr.json()
                        if s.get("status") == "done":
                            did_video_url = s.get("result_url")
                            break
                        elif s.get("status") == "error":
                            print(f"D-ID error: {s}")
                            break
            print(f"⏱ D-ID: {time.time()-t3:.2f}s → {did_video_url}")
        except Exception as e:
            print(f"⚠️ D-ID failed: {e}")

    print(f"⏱ Total: {time.time()-t0:.2f}s")

    headers = {
        "X-Transcription": urllib.parse.quote(question),
        "X-Reply": urllib.parse.quote(reply),
    }
    if did_video_url:
        headers["X-DID-Video"] = urllib.parse.quote(did_video_url)

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers=headers
    )

@app.get("/health")
def health():
    return {"status": "online", "agent": "Vinni AI", "version": "1.0.0"}