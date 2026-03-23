import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Production RAG System"
    API_V1_STR: str = "/api/v1"
    
    # Gemini API Key overrides
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Storage settings
    UPLOAD_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "uploads")
    CHROMA_PERSIST_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db")

    # Redis Config
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Retrieval Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    PARENT_CHUNK_SIZE: int = 2000
    CHILD_CHUNK_SIZE: int = 400

    class Config:
        env_file = ".env"

settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
