import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import torch
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 512       # silero requirement — 32ms per chunk
THRESHOLD = 0.5        # VAD sensitivity — above = speech
SILENCE_LIMIT = 0.8    # seconds of silence before stopping
INPUT_WAV = "input.wav"
OUTPUT_MP3 = "output.mp3"

# Load Silero VAD model once at startup
print("Loading VAD model...")
vad_model, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False
)
print("VAD model ready.")


def record_audio() -> str:
    """
    Records audio using Voice Activity Detection.
    Automatically starts when you speak and stops after silence.
    Returns path to saved wav file.
    """
    print("\n🎙  Listening... speak when ready.")

    audio_buffer = []          # stores all speech chunks
    silence_chunks = 0         # counts consecutive silent chunks
    speaking = False           # tracks if user has started speaking
    max_silence_chunks = int(SILENCE_LIMIT * SAMPLE_RATE / CHUNK_SIZE)
    # max_silence_chunks = 0.8 * 16000 / 512 = ~25 chunks of silence before stopping

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=np.float32,
        blocksize=CHUNK_SIZE   # deliver exactly 512 samples per callback
    ) as stream:

        while True:
            # Read one chunk (512 samples = 32ms of audio)
            chunk, _ = stream.read(CHUNK_SIZE)

            # Convert to tensor for Silero VAD
            audio_tensor = torch.from_numpy(chunk.flatten())

            # Get VAD probability — is this speech?
            with torch.no_grad():
                speech_prob = vad_model(audio_tensor, SAMPLE_RATE).item()

            if speech_prob > THRESHOLD:
                # Speech detected
                if not speaking:
                    print("🔴 Speaking detected — recording...")
                    speaking = True
                silence_chunks = 0
                audio_buffer.append(chunk)

            elif speaking:
                # Was speaking but now silence
                silence_chunks += 1
                audio_buffer.append(chunk)  # include silence in buffer (natural pauses)

                if silence_chunks > max_silence_chunks:
                    print("✅ Done speaking.")
                    break

    if not audio_buffer:
        return None

    # Join all chunks and save
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
            language="en"        # force English — faster and more accurate
        )

    text = result.text.strip()
    print(f"You said: {text}")
    return text

def speak(text: str) -> None:
    """
    Converts text to speech using OpenAI TTS and plays it.
    """
    print("🔊 Speaking...")

    response = client.audio.speech.create(
        model="tts-1",        # tts-1 = faster, tts-1-hd = higher quality
        voice="onyx",         # onyx = deep, professional sounding
        input=text,
        speed=1.0             # 0.25 to 4.0 — 1.0 is normal speed
    )

    # Save audio to file
    with open(OUTPUT_MP3, "wb") as f:
        f.write(response.content)

    # Play on Mac
    os.system(f"afplay {OUTPUT_MP3}")

def voice_loop() -> None:
    """
    Full voice conversation loop.
    Listen → Transcribe → Ask Vinni → Speak → repeat.
    """
    from vinni.agent import ask_vinni

    print("\n" + "=" * 40)
    print("  Vinni AI — Voice Mode")
    print("  Say 'goodbye' or 'exit' to quit")
    print("=" * 40)

    conversation_history = []

    while True:
        # Step 1 — record
        path = record_audio()
        if not path:
            print("No speech detected, try again.")
            continue

        # Step 2 — transcribe
        question = transcribe(path)
        if not question:
            print("Could not transcribe, try again.")
            continue

        # Step 3 — check for exit
        if any(word in question.lower() for word in ["goodbye", "exit", "quit", "bye"]):
            speak("Goodbye! Feel free to reach out anytime.")
            break

        # Step 4 — ask Vinni
        reply, conversation_history = ask_vinni(question, conversation_history)
        print(f"\nVinni: {reply}\n")

        # Step 5 — speak reply
        speak(reply)

if __name__ == "__main__":
    voice_loop()