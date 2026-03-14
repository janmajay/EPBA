import os
import json
import base64
import asyncio
import websockets
import io
import scipy.io.wavfile
import scipy.signal
import numpy as np
from shared.src.config import settings, OPENAI_API_KEY
from shared.src.prompts.voice_summary import VOICE_SUMMARY_SYSTEM_PROMPT
import datetime
from pathlib import Path


class RealtimeClient:
    def __init__(self):
        self.api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        self.endpoint = settings.GPT_REALTIME_ENDPOINT

        # Audio config from settings
        self.sample_rate = settings.AUDIO_SAMPLE_RATE          # default 16000
        self.input_format = settings.AUDIO_INPUT_FORMAT        # default pcm16
        self.output_format = settings.AUDIO_OUTPUT_FORMAT      # default pcm16
        self.voice = settings.AUDIO_VOICE                      # default alloy

        # VAD config from settings
        self.vad_type = settings.VAD_TYPE                      # server_vad | none
        self.vad_threshold = settings.VAD_THRESHOLD            # 0.0 – 1.0
        self.vad_prefix_padding_ms = settings.VAD_PREFIX_PADDING_MS
        self.vad_silence_duration_ms = settings.VAD_SILENCE_DURATION_MS

        # Ensure correct protocol
        if self.endpoint.startswith("https://"):
            self.endpoint = self.endpoint.replace("https://", "wss://")
        elif not self.endpoint.startswith("wss://"):
            if not self.endpoint.startswith("ws://"):
                self.endpoint = f"wss://{self.endpoint}"

    def _build_turn_detection(self):
        """Build the turn_detection config from env-based settings."""
        if self.vad_type == "none" or not self.vad_type:
            return None
        return {
            "type": self.vad_type,
            "threshold": self.vad_threshold,
            "prefix_padding_ms": self.vad_prefix_padding_ms,
            "silence_duration_ms": self.vad_silence_duration_ms,
        }

    def _convert_audio_to_pcm16(self, audio_bytes):
        """Convert input audio (e.g. WAV) to PCM16 at configured sample rate (24kHz), Mono."""
        try:
            sr, data = scipy.io.wavfile.read(io.BytesIO(audio_bytes))
            
            # 1. Convert to float32 [-1.0, 1.0] for processing
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            elif data.dtype == np.uint8:
                data = (data.astype(np.float32) - 128) / 128.0
            elif np.issubdtype(data.dtype, np.floating):
                pass # Already float
            else:
                 data = data.astype(np.float32)

            # 2. Stereo -> Mono
            if len(data.shape) > 1:
                data = data.mean(axis=1)

            # 3. Resample if needed
            if sr != self.sample_rate:
                num_samples = int(len(data) * float(self.sample_rate) / sr)
                data = scipy.signal.resample(data, num_samples)

            # 4. Normalize Volume (Target 95% peak)
            max_val = np.max(np.abs(data))
            if max_val > 0.001: # Avoid boosting pure silence
                target = 0.95
                gain = target / max_val
                # Cap max gain to 30dB (approx 31x) to avoid explosive noise
                gain = min(gain, 31.0) 
                data = data * gain

            # 5. Convert to int16
            data = np.clip(data * 32767, -32768, 32767).astype(np.int16)
            return data.tobytes()
        except Exception as e:
            print(f"Error converting audio with scipy: {e}")
            return None

    async def _save_audio_log(self, session_id, prefix, pcm_data):
        """Asynchronously save audio to logs directory."""
        if not session_id or not pcm_data:
            return

        def _write():
            try:
                # directory: logs/realtime_client/{session_id}
                log_dir = Path("logs/realtime_client") / str(session_id)
                log_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = log_dir / f"{prefix}_{timestamp}.wav"
                
                scipy.io.wavfile.write(filename, self.sample_rate, np.frombuffer(pcm_data, dtype=np.int16))
                print(f"Saved audio log: {filename}")
            except Exception as e:
                print(f"Error saving audio log: {e}")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _write)

    async def run_flow(self, audio_input_bytes, status_callback, orchestrator_callback, session_id=None):
        """
        Executes the full S2S flow:
        1. Connect to Realtime API
        2. Send User Audio -> Get Transcript (STT)
        3. Call Orchestrator -> Get Context
        4. Send Context -> Get Audio Response (TTS)
        
        Returns: (wav_bytes, user_transcript_text, orchestrator_response_text, assistant_transcript_text)
        """
        pcm_data = self._convert_audio_to_pcm16(audio_input_bytes)
        if not pcm_data:
            return None, "Audio conversion failed.", None, None
        
        # Async Log Input
        if session_id:
            asyncio.create_task(self._save_audio_log(session_id, "input", pcm_data))

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }

        async with websockets.connect(self.endpoint, additional_headers=headers) as ws:
            # 1. Initialize Session with VAD + audio config
            session_update = {
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "instructions": VOICE_SUMMARY_SYSTEM_PROMPT,
                    "voice": self.voice,
                    "input_audio_format": self.input_format,
                    "output_audio_format": self.output_format,
                    "input_audio_transcription": {
                        "model": "whisper-1"
                    },
                    # For file-based input (Push-to-Talk), disable VAD to prevent race conditions
                    # with manual commit. We want to process the full clip as one turn.
                    "turn_detection": None,
                }
            }
            await ws.send(json.dumps(session_update))

            # Wait for session.created and session.updated before sending audio
            session_ready = False
            while not session_ready:
                msg = await ws.recv()
                event = json.loads(msg)
                ev_type = event.get("type")
                if ev_type == "session.updated":
                    session_ready = True
                elif ev_type == "error":
                    print(f"Session init error: {event}")
                    return None, f"Session Error: {event.get('error', {}).get('message', 'Unknown')}", None, None

            # 2. Send Audio
            status_callback("Transcribing...")

            base64_audio = base64.b64encode(pcm_data).decode("utf-8")
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64_audio
            }))

            # Commit audio to trigger processing
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

            transcript_text = ""
            orchestrator_response = ""
            assistant_transcript = ""
            audio_chunks = []

            # Event Loop
            processing_stage = "transcribing"  # transcribing -> querying -> synthesizing

            async for message in ws:
                event = json.loads(message)
                event_type = event.get("type")

                # --- STT Handling ---
                if event_type == "conversation.item.input_audio_transcription.completed":
                    transcript_text = event.get("transcript", "")
                    status_callback(f"Heard: {transcript_text}")

                    # Call Orchestrator
                    status_callback("Consulting Agents...")
                    processing_stage = "querying"

                    orchestrator_response = await asyncio.to_thread(orchestrator_callback, transcript_text)

                    # Send Response Create with Context
                    status_callback("Synthesizing Answer...")
                    processing_stage = "synthesizing"

                    response_create = {
                        "type": "response.create",
                        "response": {
                            "modalities": ["text", "audio"],
                            "instructions": f"{VOICE_SUMMARY_SYSTEM_PROMPT}\n\nContext Data from Database:\n{orchestrator_response}",
                        }
                    }
                    await ws.send(json.dumps(response_create))

                # --- TTS Handling ---
                elif event_type == "response.audio.delta":
                    delta = event.get("delta", "")
                    if delta:
                        audio_chunks.append(base64.b64decode(delta))

                elif event_type == "response.audio_transcript.done":
                    assistant_transcript = event.get("transcript", "")

                elif event_type == "response.done":
                    if processing_stage == "synthesizing":
                        break

                elif event_type == "error":
                    print(f"Realtime API Error: {event}")
                    if "message" in event.get("error", {}):
                        return None, f"API Error: {event['error']['message']}", None, None

            # Combine Audio
            full_audio = b"".join(audio_chunks)
            
            # Async Log Output
            if session_id and full_audio:
                asyncio.create_task(self._save_audio_log(session_id, "output", full_audio))

            if not full_audio:
                return None, "No audio generated.", orchestrator_response, None

            # Convert raw PCM16 back to WAV for Streamlit playback
            try:
                wav_io = io.BytesIO()
                audio_array = np.frombuffer(full_audio, dtype=np.int16)
                scipy.io.wavfile.write(wav_io, self.sample_rate, audio_array)
                return wav_io.getvalue(), transcript_text, orchestrator_response, assistant_transcript
            except Exception as e:
                return None, f"Audio export error: {e}", orchestrator_response, None
