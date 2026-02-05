"""Configuration for Deep Research Agent."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    aws_region: str = "us-west-2"
    # model_id: str = "anthropic.claude-opus-4-5-20251101-v1:0"
    model_id: str = "global.anthropic.claude-opus-4-5-20251101-v1:0"
    temperature: float = 0.0
    max_tokens: int = 4096 # max 200K tokens

    class Config:
        env_file = ".env"
