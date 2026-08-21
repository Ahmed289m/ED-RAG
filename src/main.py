from fastapi import FastAPI
from routes import base, data
from motor.motor_asyncio import AsyncIOMotorClient
from helpers.config import get_settings
from contextlib import asynccontextmanager
from store.LLM import LLMProviderFactory


async def startup_db_client():
    settings = get_settings()
    app.mongo_conn = AsyncIOMotorClient(settings.MONGODB_URL)
    app.db_client = app.mongo_conn[settings.MONGODB_DATABASE]
    llm_provider_factory = LLMProviderFactory(config=settings.dict())

    ##GEN
    app.gen_client = llm_provider_factory.create_llm_provider(provider_name=settings.GENERATION_BACKEND)
    app.gen_client.set_gen_model(model_id=settings.GENERATION_MODEL_ID)
    ##EMBED
    app.embed_client = llm_provider_factory.create_llm_provider(provider_name=settings.EMBEDDING_BACKEND)
    app.embed_client.set_embed_model(model_id=settings.EMBEDDING_MODEL_ID, embedding_size=settings.EMBEDDING_MODEL_SIZE)


async def shutdown_db_client():
    app.mongo_conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup_db_client()
    yield
    await shutdown_db_client()

app = FastAPI(lifespan=lifespan)


    



app.include_router(base.base_router)
app.include_router(data.data_router)

