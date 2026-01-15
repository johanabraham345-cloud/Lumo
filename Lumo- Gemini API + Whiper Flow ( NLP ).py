import os
import queue
import threading
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import google.generativeai as genai
import subprocess

# ==========================
# CONFIG
# ==========================

GEMINI_MODEL = "models/gemini-2.0-flash-lite"  # light & cheap

# Mic config (these are from your working test)
MIC_DEVICE = 1        # "USB PnP Sound Device"
SAMPLE_RATE = 48000   # this one is supported
BLOCK_DURATION = 0.5  # seconds per block
LANGUAGE = "en"

# ==========================
# INIT GEMINI
# ==========================

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_KEY not set.")
    raise SystemExit

genai.configure(api_key=api_key)
gemini = genai.GenerativeModel(GEMINI_MODEL)

# ==========================
# INIT WHISPER
# ==========================

MODEL_DIR = "/home/sarun/whisper_models"  # folder with ggml-base.bin from before

print("Loading Whisper model from:", MODEL_DIR)
asr = WhisperModel(MODEL_DIR, device="cpu", compute_type="int8")

# ==========================
# TEXT TO SPEECH
# ==========================

def speak(text: str):
    print("AI:", text)
    # Use espeak (modify -s 160 for speed, -p for pitch if you like)
    subprocess.run(["espeak", "-s", "160", text])


# ==========================
# GEMINI CHAT FUNCTION
# ==========================

def ask_gemini(prompt: str) -> str:
    try:
        resp = gemini.generate_content(prompt)
        # Newer API returns .text
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()
        # Fallback
        return "I am having trouble generating a response."
    except Exception as e:
        print("Gemini error:", e)
        return "Sorry, I had trouble talking to Gemini."


# ==========================
# AUDIO STREAM + WHISPER LOOP
# ==========================

# A queue to hold raw audio blocks
audio_q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    # This runs in audio thread
    if status:
        print("Audio status:", status)
    # Copy to avoid referencing the buffer
    audio_q.put(indata[:, 0].copy())

def transcribe_loop():
    """
    Pulls audio from the queue, builds chunks of ~2 seconds,
    sends to Whisper, then sends text to Gemini + speak.
    """
    CHUNK_SECONDS = 2.0
    samples_per_chunk = int(SAMPLE_RATE * CHUNK_SECONDS)
    buffer = np.zeros(0, dtype=np.float32)

    print("\nVoice AI Ready.")
    print("Listening continuously. Ctrl + C to stop.\n")

    while True:
        # Get next block from queue
        block = audio_q.get()
        # Convert int16 -> float32 in [-1, 1]
        if block.dtype != np.float32:
            block = block.astype(np.float32) / 32768.0

        # Append to buffer
        buffer = np.concatenate((buffer, block))

        # If enough samples for one segment
        if len(buffer) >= samples_per_chunk:
            segment_audio = buffer[:samples_per_chunk]
            buffer = buffer[samples_per_chunk:]

            # Whisper expects a numpy float32 array
            segments, _ = asr.transcribe(segment_audio, language=LANGUAGE)

            text = " ".join(seg.text for seg in segments).strip()
            if not text:
                continue

            print("\nYOU SAID:", text)

            # Very short garbage like "uh", "ah"
            if len(text) < 3:
                continue

            # Send text to Gemini
            answer = ask_gemini(text)
            speak(answer)


def main():
    # Start transcription thread
    t = threading.Thread(target=transcribe_loop, daemon=True)
    t.start()

    # Start sounddevice stream
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

