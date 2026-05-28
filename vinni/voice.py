import os
import torch
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 512
THRESHOLD = 0.5
SILENCE_LIMIT = 0.8
INPUT_WAV = "input.wav"
OUTPUT_MP3 = "output.mp3"

# VAD model — loaded lazily only when voice mode is selected
vad_model = None


def load_vad_model():
    global vad_model
    if vad_model is None:
        print("Loading VAD model...")
        vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False
        )
        print("VAD model ready.")


def record_audio() -> str:
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    """
    Records audio using Voice Activity Detection.
    Automatically starts when you speak and stops after silence.
    Returns path to saved wav file.
    """
    print("\n🎙  Listening... speak when ready.")

    audio_buffer = []
    silence_chunks = 0
    speaking = False
    max_silence_chunks = int(SILENCE_LIMIT * SAMPLE_RATE / CHUNK_SIZE)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=np.float32,
        blocksize=CHUNK_SIZE
    ) as stream:

        while True:
            chunk, _ = stream.read(CHUNK_SIZE)
            audio_tensor = torch.from_numpy(chunk.flatten())

            with torch.no_grad():
                speech_prob = vad_model(audio_tensor, SAMPLE_RATE).item()

            if speech_prob > THRESHOLD:
                if not speaking:
                    print("🔴 Speaking detected — recording...")
                    speaking = True
                silence_chunks = 0
                audio_buffer.append(chunk)

            elif speaking:
                silence_chunks += 1
                audio_buffer.append(chunk)

                if silence_chunks > max_silence_chunks:
                    print("✅ Done speaking.")
                    break

    if not audio_buffer:
        return None

    audio = np.concatenate(audio_buffer, axis=0)
    sf.write(INPUT_WAV, audio, SAMPLE_RATE)
    return INPUT_WAV


def transcribe(audio_path: str) -> str:
    """
    Sends audio file to OpenAI Whisper and returns transcribed text.
    """
    print("📝 Transcribing...")

    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="en"
        )

    text = result.text.strip()
    print(f"You said: {text}")
    return text


def speak(text: str, output_path: str = "output.wav") -> None:
    use_resemble = os.getenv("RESEMBLE_API_KEY") and os.getenv("RESEMBLE_VOICE_UUID")

    if use_resemble:
        try:
            print("🔊 Speaking (Resemble)...")
            response = requests.post(
                os.getenv("RESEMBLE_URL"),
                headers={
                    "Authorization": f"Bearer {os.getenv('RESEMBLE_API_KEY')}",
                    "Content-Type": "application/json"
                },
                json={
                    "voice_uuid": os.getenv("RESEMBLE_VOICE_UUID"),
                    "data": text
                },
                timeout=30
            )

            if response.status_code == 200:
                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    # synthesize endpoint — base64 encoded
                    import base64
                    data = response.json()
                    audio_bytes = base64.b64decode(data["audio_content"])
                else:
                    # stream endpoint — raw audio bytes
                    audio_bytes = response.content

                with open(output_path, "wb") as f:
                    f.write(audio_bytes)
                if output_path == "output.wav":
                    os.system(f"afplay {output_path}")
                return

            else:
                print(f"⚠️  Resemble failed ({response.status_code}), falling back to OpenAI TTS")

        except Exception as e:
            print(f"⚠️  Resemble error: {e}, falling back to OpenAI TTS")

    # Fallback — OpenAI TTS
    print("🔊 Speaking (OpenAI TTS)...")
    response = client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=text,
        speed=1.0
    )
    with open(output_path, "wb") as f:
        f.write(response.content)
    # only play locally
    if output_path == "output.wav":
        os.system(f"afplay {output_path}")


def voice_loop() -> None:
    """
    Full voice conversation loop.
    Listen → Transcribe → Ask Vinni → Speak → repeat.
    """
    from vinni.agent import ask_vinni

    load_vad_model()

    print("\n" + "=" * 40)
    print("  Vinni AI — Voice Mode")
    print("  Say 'goodbye' or 'exit' to quit")
    print("=" * 40)

    conversation_history = []

    while True:
        path = record_audio()
        if not path:
            print("No speech detected, try again.")
            continue

        question = transcribe(path)
        if not question:
            print("Could not transcribe, try again.")
            continue

        if any(word in question.lower() for word in ["goodbye", "exit", "quit", "bye"]):
            speak("Goodbye! Feel free to reach out anytime.")
            break

        reply, conversation_history = ask_vinni(question, conversation_history)
        print(f"\nVinni: {reply}\n")

        speak(reply)


if __name__ == "__main__":
    voice_loop()