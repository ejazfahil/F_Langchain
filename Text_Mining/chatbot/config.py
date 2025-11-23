"""
Configuration management for the RAG chatbot application.
Uses pydantic-settings for type-safe environment variable loading.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # API Keys
    openai_api_key: str = Field(..., description="OpenAI API key")
    groq_api_key: Optional[str] = Field(None, description="Groq API key (optional)")
    cohere_api_key: Optional[str] = Field(None, description="Cohere API key for reranking (optional)")
    
    # Model Configuration
    default_model: str = Field("gpt-3.5-turbo", description="Default LLM model to use")
    model_temperature: float = Field(0.1, ge=0.0, le=2.0, description="Model temperature")
    
    # Embedding Configuration
    embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name"
    )
    
    # RAG Configuration
    chunk_size: int = Field(1000, gt=0, description="Text chunk size for splitting")
    chunk_overlap: int = Field(200, ge=0, description="Overlap between chunks")
    retrieval_k: int = Field(4, gt=0, description="Number of documents to retrieve")
    
    # Vector Store Configuration
    vector_store_path: str = Field("./chroma_store", description="Path to vector store")
    persist_directory: str = Field("./chroma_store", description="Persistence directory")
    
    # Application Configuration
    app_title: str = Field("Enhanced RAG ChatBot Using LangChain", description="Application title")
    max_upload_size_mb: int = Field(10, gt=0, description="Max upload size in MB")
    enable_streaming: bool = Field(True, description="Enable streaming responses")
    enable_reranking: bool = Field(False, description="Enable document reranking")
    
    def get_llm_provider(self) -> str:
        """Determine which LLM provider to use based on model name."""
        if "gpt" in self.default_model.lower():
            return "openai"
        elif "llama" in self.default_model.lower() or "mixtral" in self.default_model.lower():
            return "groq"
        return "openai"
    
    def validate_api_keys(self) -> bool:
        """Validate that required API keys are present."""
        provider = self.get_llm_provider()
        if provider == "openai" and not self.openai_api_key:
            return False
        if provider == "groq" and not self.groq_api_key:
            return False
        if self.enable_reranking and not self.cohere_api_key:
            return False
        return True


# Global settings instance
settings = Settings()
