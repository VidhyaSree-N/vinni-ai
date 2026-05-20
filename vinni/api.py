from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from vinni.agent import ask_vinni
from vinni.voice import transcribe, speak
from dotenv import load_dotenv
import tempfile
import os
import uuid
from fastapi.staticfiles import StaticFiles
import urllib.parse



load_dotenv()

app = FastAPI(
    title="Vinni AI",
    description="Personal AI agent for Vidhya Sree Narayanappa",
    version="1.0.0"
)

# CORS — allows your portfolio and frontend to call this API
# Without this browsers will block requests from other domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # in production replace * with your actual domain
    allow_methods=["*"],
    allow_headers=["*"]
)
# In-memory conversation store
# key = session_id, value = conversation history list
conversations: dict = {}

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")

class AskRequest(BaseModel):
    question: str
    session_id: str = "default"


class AskResponse(BaseModel):
    answer: str
    session_id: str


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    # Get or create conversation history for this session
    if request.session_id not in conversations:
        conversations[request.session_id] = []

    history = conversations[request.session_id]

    # Ask Vinni
    reply, updated_history = ask_vinni(request.question, history)

    # Save updated history
    conversations[request.session_id] = updated_history

    return AskResponse(answer=reply, session_id=request.session_id)

import time

@app.post("/voice")
async def voice(
    audio: UploadFile = File(...),
    session_id: str = Form(default="default")
):
    t0 = time.time()

    # save audio
    suffix = os.path.splitext(audio.filename)[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    # transcribe
    question = transcribe(tmp_path)
    os.unlink(tmp_path)
    print(f"⏱ Whisper: {time.time()-t0:.2f}s")

    t1 = time.time()
    if session_id not in conversations:
        conversations[session_id] = []
    reply, updated_history = ask_vinni(question, conversations[session_id])
    conversations[session_id] = updated_history
    print(f"⏱ Agent: {time.time()-t1:.2f}s")

    t2 = time.time()
    output_path = f"output_{uuid.uuid4().hex}.wav"
    speak(reply, output_path=output_path)
    print(f"⏱ TTS: {time.time()-t2:.2f}s")
    print(f"⏱ Total: {time.time()-t0:.2f}s")

    with open(output_path, "rb") as f:
        audio_bytes = f.read()
    os.unlink(output_path)

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "X-Transcription": urllib.parse.quote(question),
            "X-Reply": urllib.parse.quote(reply),
            "Access-Control-Expose-Headers": "X-Transcription, X-Reply"
        }
    )

@app.get("/health")
def health():
    return {
        "status": "online",
        "agent": "Vinni AI",
        "version": "1.0.0"
    }