from ..LLMInterface import LLMInterface
from ..LLMEnums import OpenAIMessageRoleEnum
from openai import OpenAI
import logging

class OpenaiProvider(LLMInterface):
    def __init__(self, api_key: str, url:str=None, default_gen_temperature: float = 0.1, default_max_output_tokens: int = 1000, default_max_input_tokens: int = 1000):
        self.api_key = api_key
        self.url = url
        self.gen_model_id = None
        self.embed_model_id = None
        self.embedding_size = None
        self.temperature = default_gen_temperature
        self.max_output_tokens = default_max_output_tokens
        self.max_input_tokens = default_max_input_tokens
        self.logger = logging.getLogger(__name__)
        self.client = OpenAI(api_key=self.api_key, api_key=self.url)

        




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
            self.logger.error("OpenAI client is not initialized.")

        if not self.gen_model_id:
            self.logger.error("Generation Model ID is not set up.")

        


        chat_history.append(self.construct_prompt(prompt=prompt, role=OpenAIMessageRoleEnum.USER.value))
        

        response = self.client.chat.completions.create(
            model=self.gen_model_id,
            messages=chat_history,
            max_output_tokens=max_output_tokens,
            temperature=temperature
        )

        if not response or not response.choices or len(response.choices) == 0:
            self.logger.error("No response received from OpenAI API. Error while generating text.")
            return None

        return response.choices[0].message.content



    def text_to_embedding(self, text:str, doc_type:str=None):

        if not self.client:
            self.logger.error("OpenAI client is not initialized.")
            return None

        if not self.embed_model_id:
            self.logger.error("Embedding Model ID is not set up.")
            return None

        response = self.client.embeddings.create(
        input=text, model=self.embed_model_id
        )

        if not response or not response.data or len(response.data) == 0:
            self.logger.error("No response received from OpenAI API. Error while generating embedding.")
            return None

        return response.data[0].embedding