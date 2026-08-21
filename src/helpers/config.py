from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    APP_NAME: str
    APP_VERSION: str
    GENERATION_BACKEND: str
    EMBEDDING_BACKEND: str
    FILE_ALLOWED_TYPES: list
    FILE_MAX_SIZE: int
    FILE_DEFAULT_CHUNK_SIZE: int
    DEFAULT_GEN_TEMPERATURE: float
    DEFAULT_MAX_OUTPUT_TOKENS: int
    DEFAULT_MAX_INPUT_TOKENS: int
    OPENAI_API_KEY: str
    COHERE_API_KEY: str
    OPENAI_API_URL: str
    MONGODB_URL: str
    MONGODB_DATABASE: str

    

   

    class Config:
        env_file = ".env"

def get_settings():
    return Settings()
