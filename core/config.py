"""
Core: Settings
All configuration is read from environment variables (or a .env file).
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ── Dataset ────────────────────────────────────────────────────────
    csv_path: str = "data/mental_with_multilabels.csv"

    # ── ML Models ──────────────────────────────────────────────────────
    bi_encoder_model: str = "all-MiniLM-L6-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    llm_max_new_tokens: int = 200

    # ── Use-case defaults ──────────────────────────────────────────────
    default_n_tips: int = 5
    default_tag_filter: str = "All"

    # ── API ────────────────────────────────────────────────────────────
    api_title: str = "Mental Health Support API"
    api_version: str = "1.0.0"
    api_description: str = (
        "RAG-powered mental health support assistant. "
        "Returns community-sourced suggestions synthesised by an LLM."
    )

    # ── Lightning AI / deployment ──────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 7860           # Lightning AI default exposed port
    workers: int = 1           # Keep at 1 when loading large models in memory
    use_fallback_llm: bool = False  # Set to True to skip loading Qwen (faster cold start)

    # ── Security ────────────────────────────────────────────────────────
    secret_key: str = "change-me-in-production"
    allowed_hosts: list[str] = ["*"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
