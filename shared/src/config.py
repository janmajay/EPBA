import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load .env
root_dir = Path(__file__).parent.parent.parent
load_dotenv(root_dir / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class Settings:
    def __init__(self):
        self._config = self._load_yaml()

    def _load_yaml(self):
        # Allow overriding config path
        config_path = os.getenv("CONFIG_PATH", str(root_dir / "config" / "settings.yaml"))
        if not os.path.exists(config_path):
            # Fallback for Docker where structure might differ slightly or if running from different cwd
            # defaulting to standard location if not found
             config_path = "config/settings.yaml"

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        return {}

    @property
    def LLM_MODEL_NAME(self):
        return self._config.get("llm", {}).get("model_name", "gpt-4o")

    @property
    def LLM_TEMPERATURE(self):
        return self._config.get("llm", {}).get("temperature", 0)

    @property
    def EMBEDDING_MODEL_NAME(self):
        return self._config.get("llm", {}).get("embedding_model", "text-embedding-3-small")

    @property
    def VECTOR_STORE_DIR(self):
        # Robust path handling - absolute or relative to execution
        # In Docker, we might mount /app/data.
        # Check defaults
        raw_path = self._config.get("vector_store", {}).get("dir", "data/chroma_db")
        # In this simple setup, we just return the raw path as we expect CWD to be correct or Volume mapped
        return raw_path

    @property
    def VECTOR_SEARCH_K(self):
        return self._config.get("vector_store", {}).get("search_k", 5)

    @property
    def CHUNK_SIZE(self):
        return self._config.get("vector_store", {}).get("chunk_size", 600)
    
    @property
    def CHUNK_OVERLAP(self):
        return self._config.get("vector_store", {}).get("chunk_overlap", 50)

    @property
    def VECTOR_SOURCE_PATH(self):
        # Allow Env override
        env_source = os.getenv("SOURCE_PATH")
        if env_source:
             return env_source
             
        # Config value
        cfg_source = self._config.get("vector_store", {}).get("source_path", "data/patient_reports")
        
        # Smart Resolution if relative path doesn't exist mainly for local vs docker context
        # But cleanest is to rely on the config being set correctly for the environment
        # or standardizing CWD.
        # Docker: WORKDIR /app. data/patient_reports exists.
        # Local Root: data/patient_reports exists.
        # Local Subdir: This breaks.
        # Let's keep the legacy fallback check for now
        if not os.path.exists(cfg_source) and os.path.exists(f"../../{cfg_source}"):
             return f"../../{cfg_source}"
        
        return cfg_source

    @property
    def SQL_DB_PATH(self):
        # Handle /app/data vs data/
         path = self._config.get("database", {}).get("path", "data/patients.db")
         # If running in docker without volume adjustment, path might need /app prefix if raw path is relative
         # But usually we mount to /app in Docker so /app/data/patients.db is consistent if working dir is /app
         return f"/app/{path}" if path.startswith("data/") and os.path.exists("/app") else path

    @property
    def SQL_AGENT_URL(self):
        return os.getenv("SQL_AGENT_URL", self._config.get("services", {}).get("sql_agent_url"))
    
    @property
    def VECTOR_AGENT_URL(self):
        return os.getenv("VECTOR_AGENT_URL", self._config.get("services", {}).get("vector_agent_url"))
    
    @property
    def SUMMARIZER_AGENT_URL(self):
        return os.getenv("SUMMARIZER_AGENT_URL", self._config.get("services", {}).get("summarizer_agent_url"))

    @property
    def ORCHESTRATOR_URL(self):
        return os.getenv("ORCHESTRATOR_URL", self._config.get("services", {}).get("orchestrator_url"))

    @property
    def AGENT_REGISTRY_URL(self):
        return os.getenv("AGENT_REGISTRY_URL", self._config.get("services", {}).get("agent_registry_url", "http://agent_registry:8004"))

    @property
    def GPT_REALTIME_ENDPOINT(self):
        return os.getenv("GPT_REALTIME_ENDPOINT", self._config.get("llm", {}).get("realtime_endpoint"))

    # ── Audio / VAD Configuration ─────────────────────────────────
    @property
    def AUDIO_SAMPLE_RATE(self):
        return int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))

    @property
    def AUDIO_INPUT_FORMAT(self):
        return os.getenv("AUDIO_INPUT_FORMAT", "pcm16")

    @property
    def AUDIO_OUTPUT_FORMAT(self):
        return os.getenv("AUDIO_OUTPUT_FORMAT", "pcm16")

    @property
    def AUDIO_VOICE(self):
        return os.getenv("AUDIO_VOICE", "alloy")

    @property
    def VAD_TYPE(self):
        return os.getenv("VAD_TYPE", "server_vad")

    @property
    def VAD_THRESHOLD(self):
        return float(os.getenv("VAD_THRESHOLD", "0.5"))

    @property
    def VAD_PREFIX_PADDING_MS(self):
        return int(os.getenv("VAD_PREFIX_PADDING_MS", "300"))

    @property
    def VAD_SILENCE_DURATION_MS(self):
        return int(os.getenv("VAD_SILENCE_DURATION_MS", "500"))

    # ── Langfuse Configuration ──────────────────────────────────
    @property
    def LANGFUSE_SECRET_KEY(self):
        return os.getenv("LANGFUSE_SECRET_KEY")

    @property
    def LANGFUSE_PUBLIC_KEY(self):
        return os.getenv("LANGFUSE_PUBLIC_KEY")

    @property
    def LANGFUSE_HOST(self):
        return os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
    
    @property
    def LANGFUSE_PROJECT_ID(self):
        return os.getenv("LANGFUSE_PROJECT_ID")
    
    @property
    def LANGFUSE_SESSION_ID(self):
        # Optional: can be passed from frontend
        return os.getenv("LANGFUSE_SESSION_ID")

# Singleton instance
settings = Settings()
