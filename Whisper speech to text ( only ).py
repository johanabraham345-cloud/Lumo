import sounddevice as sd
import numpy as np
import scipy.signal
import tempfile
import wave
from faster_whisper import WhisperModel

# ------------------ CONFIG ------------------
MIC_DEVICE = 1        # USB mic (from sd.query_devices())
INPUT_RATE = 48000    # Your mic supports this
WHISPER_RATE = 16000  # Whisper expects this
RECORD_SECONDS = 4
MODEL_SIZE = "base"   # base / small / medium
# --------------------------------------------

print("Loading Whisper model...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

print("Whisper STT ready.")
print("Speak normally. Transcribing every", RECORD_SECONDS, "seconds.\n")

while True:
    try:
        print("Listening...")

        audio = sd.rec(
            int(RECORD_SECONDS * INPUT_RATE),
            samplerate=INPUT_RATE,
            channels=1,
            dtype="int16",
            device=MIC_DEVICE
        )
        sd.wait()

        audio = audio.flatten().astype(np.float32) / 32768.0

        # Resample 48k -> 16k
        audio_16k = scipy.signal.resample(
            audio,
            int(len(audio) * WHISPER_RATE / INPUT_RATE)
        )

        # Save temp WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            with wave.open(f.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(WHISPER_RATE)
                wf.writeframes((audio_16k * 32767).astype(np.int16).tobytes())

            segments, _ = model.transcribe(f.name, language="en")

        text = " ".join([seg.text.strip() for seg in segments])

        if text.strip():
            print("You said:", text)
        else:
            print("(No speech detected)")

    except KeyboardInterrupt:
        print("\nStopping Whisper test.")
        break
