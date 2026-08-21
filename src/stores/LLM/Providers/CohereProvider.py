import cohere
import logging
from ..LLMInterface import LLMInterface
from ..LLMEnums import CohereMessageRoleEnum, CohereInputTypeEnum,DocTypeEnum


class CohereProvider(LLMInterface):
    def __init__(self, api_key: str, default_gen_temperature: float = 0.1, default_max_output_tokens: int = 1000):
        self.api_key = api_key
        self.gen_model_id = None
        self.embed_model_id = None
        self.embedding_size = None
        self.temperature = default_gen_temperature
        self.max_output_tokens = default_max_output_tokens
        self.logger = logging.getLogger(__name__)
        self.client = cohere.ClientV2(api_key=self.api_key)

    
    def set_gen_model(self, model_id: str):
        self.gen_model_id = model_id

    
    def set_embed_model(self, model_id: str, embedding_size: int):
        self.embed_model_id = model_id
        self.embedding_size = embedding_size


    def construct_prompt(self,prompt: str, role:str):
        return {
            "role": role,
            "content": prompt
        }


    def generate_text(self, prompt: str, chat_history: list=[], max_output_tokens: int = None, temperature: float = None):

        max_output_tokens = max_output_tokens if max_output_tokens else self.max_output_tokens
        temperature = temperature if temperature else self.temperature

        if not self.client:
            self.logger.error("Cohere client is not initialized.")
            return None

        if not self.gen_model_id:
            self.logger.error("Generation Model ID is not set up.")
            return None


        chat_history.append(self.construct_prompt(prompt=prompt, role=CohereMessageRoleEnum.USER.value))
        

        response = self.client.chat(
            model=self.gen_model_id,
            messages=chat_history,
)

        if not response or not response.message or len(response.message.content) == 0 or not response.message.content[0].text:
            self.logger.error("No response received from Cohere API. Error while generating text.")
            return None

        return response.message.content[0].text



    def text_to_embedding(self, text:str, doc_type:str=None):

        if not self.client:
            self.logger.error("Cohere client is not initialized.")

        if not self.embed_model_id:
            self.logger.error("Embedding Model ID is not set up.")


        input_type = CohereInputTypeEnum.DOCUMENT.value if doc_type == CohereInputTypeEnum.DOCUMENT.value else CohereInputTypeEnum.QUERY.value

        response = self.client.embed(
    texts=[text],
    model=self.embed_model_id,
    input_type=input_type,
    output_dimension=self.embedding_size,
    embedding_types=["float"],
)

        if not response or not response.embeddings or len(response.embeddings.float) == 0:
            self.logger.error("No response received from Cohere API. Error while generating embedding.")
            return None

        return response.embeddings.float