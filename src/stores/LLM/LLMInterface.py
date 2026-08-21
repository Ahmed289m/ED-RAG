from abc import ABC, abstractmethod

class LLMFactory(ABC):

    @abstractmethod
    def set_gen_model(self, model_id: str):
        pass

    @abstractmethod
    def set_embed_model(self, model_id: str, embedding_size: int):
        pass

    @abstractmethod
    def construct_prompt(self,prompt: str, role:str):
        pass

    @abstractmethod
    def generate_text(self, prompt: str, chat_history: list=[], max_output_tokens: int = None, temperature: float = None):
        pass

    @abstractmethod
    def text_to_embedding(self, text:str, doc_type:str=None):
        pass
        