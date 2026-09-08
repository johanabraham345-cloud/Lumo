import os
import queue
import threading
import time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import google.generativeai as genai
import subprocess

GEMINI_MODEL = "models/gemini-2.0-flash-lite"

MIC_DEVICE = 1
SAMPLE_RATE = 48000
BLOCK_DURATION = 0.5
LANGUAGE = "en"

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_KEY not set.")
    raise SystemExit
genai.configure(api_key=api_key)
gemini = genai.GenerativeModel(GEMINI_MODEL)

MODEL_DIR = "/home/sarun/whisper_models"
print("Loading Whisper model from:", MODEL_DIR)
asr = WhisperModel(MODEL_DIR, device="cpu", compute_type="int8")

def speak(text: str):
    print("AI:", text)
    subprocess.run(["espeak", "-s", "160", text])

def ask_gemini(prompt: str) -> str:
    try:
        resp = gemini.generate_content(prompt)
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()
        return "I am having trouble generating a response."
    except Exception as e:
        print("Gemini error:", e)
        return "Sorry, I had trouble talking to Gemini."

audio_q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio status:", status)
    audio_q.put(indata[:, 0].copy())

def transcribe_loop():
    CHUNK_SECONDS = 2.0
    samples_per_chunk = int(SAMPLE_RATE * CHUNK_SECONDS)
    buffer = np.zeros(0, dtype=np.float32)
    print("\nVoice AI Ready.")
    print("Listening continuously. Ctrl + C to stop.\n")
    while True:
        block = audio_q.get()
        if block.dtype != np.float32:
            block = block.astype(np.float32) / 32768.0
        buffer = np.concatenate((buffer, block))
        if len(buffer) >= samples_per_chunk:
            segment_audio = buffer[:samples_per_chunk]
            buffer = buffer[samples_per_chunk:]
            segments, _ = asr.transcribe(segment_audio, language=LANGUAGE)
            text = " ".join(seg.text for seg in segments).strip()
            if not text:
                continue
            print("\nYOU SAID:", text)
            if len(text) < 3:
                continue
            answer = ask_gemini(text)
            speak(answer)

def main():
    t = threading.Thread(target=transcribe_loop, daemon=True)
    t.start()
    with sd.InputStream(
        device=MIC_DEVICE,
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=int(SAMPLE_RATE * BLOCK_DURATION),
        dtype="int16",
        callback=audio_callback,
    ):
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping...")

if __name__ == "__main__":
    main()
