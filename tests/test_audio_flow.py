"""
End-to-End test for Audio Modality via Realtime API.
Generates a synthetic WAV file, sends it through the RealtimeClient pipeline,
and reports the results.
"""
import sys
import os
import asyncio
import io
import requests
import numpy as np
import scipy.io.wavfile

# Ensure shared module is importable
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "services", "frontend", "src"))

from dotenv import load_dotenv
load_dotenv()

from shared.src.config import settings, OPENAI_API_KEY
from services.frontend.src.realtime_client import RealtimeClient

# Use localhost for local testing (not Docker hostname)
ORCHESTRATOR_URL = "http://localhost:8000/query"


def generate_test_wav(duration_s=2.0, sample_rate=24000):
    """Generate a short 440Hz tone WAV for testing at 24kHz."""
    num_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, num_samples, dtype=np.float64)
    data = (np.sin(2 * np.pi * 440 * t) * 1000).astype(np.int16)
    wav_io = io.BytesIO()
    scipy.io.wavfile.write(wav_io, sample_rate, data)
    return wav_io.getvalue()


def orchestrator_callback(text_query):
    """Call the Orchestrator service with the transcribed text."""
    print(f"  [ORCHESTRATOR] Querying: {text_query}")
    try:
        resp = requests.post(ORCHESTRATOR_URL, json={"query": text_query}, timeout=120)
        if resp.status_code == 200:
            data = resp.json()
            final = data.get("final_answer", "No answer.")
            print(f"  [ORCHESTRATOR] Response length: {len(final)} chars")
            return final
        else:
            msg = f"Error {resp.status_code}: {resp.text[:200]}"
            print(f"  [ORCHESTRATOR] {msg}")
            return msg
    except Exception as e:
        msg = f"Orchestrator Error: {e}"
        print(f"  [ORCHESTRATOR] {msg}")
        return msg


def status_callback(msg):
    print(f"  [STATUS] {msg}")


async def run_test():
    print("=" * 60)
    print("AUDIO FLOW E2E TEST")
    print("=" * 60)

    api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    endpoint = settings.GPT_REALTIME_ENDPOINT

    print(f"\n[CONFIG] ORCHESTRATOR_URL: {ORCHESTRATOR_URL}")
    print(f"[CONFIG] GPT_REALTIME_ENDPOINT: {endpoint}")
    print(f"[CONFIG] OPENAI_API_KEY: {'SET' if api_key else 'MISSING'}")

    if not endpoint:
        print("\n[ERROR] GPT_REALTIME_ENDPOINT is not configured!")
        return

    # Generate test audio
    print("\n[STEP 1] Generating test WAV (2s, 440Hz tone)...")
    wav_bytes = generate_test_wav()
    print(f"  WAV size: {len(wav_bytes)} bytes")

    # Initialize client
    print("\n[STEP 2] Initializing RealtimeClient...")
    client = RealtimeClient()
    print(f"  Endpoint: {client.endpoint}")

    # Run the flow
    print("\n[STEP 3] Running audio flow (WS connect -> STT -> Orchestrator -> TTS)...")
    try:
        result_audio, transcript, orchestrator_resp, assistant_transcript = await client.run_flow(
            wav_bytes,
            status_callback,
            orchestrator_callback
        )

        print("\n[RESULTS]")
        print(f"  Transcript (User): {transcript}")
        print(f"  Orchestrator: {orchestrator_resp}")
        print(f"  Transcript (AI): {assistant_transcript}")
        if result_audio:
            print(f"  Response audio size: {len(result_audio)} bytes")
            with open("/tmp/test_audio_response.wav", "wb") as f:
                f.write(result_audio)
            print(f"  Saved response audio to /tmp/test_audio_response.wav")
        else:
            print(f"  No audio response generated.")

    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_test())
