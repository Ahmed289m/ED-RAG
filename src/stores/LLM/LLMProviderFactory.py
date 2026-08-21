from Providers import OpenaiProvider, CohereProvider
from .LLMEnums import LLMProviderEnum

class LLMProviderFactory:

    def __init__(self, config: dict):
        self.config = config
    
    def create_llm_provider(self, provider_name: str):
        if provider_name.lower() == LLMProviderEnum.OPENAI.value.lower():
            
            return OpenaiProvider(api_key=self.config["OPENAI_API_KEY"], url=self.config["OPENAI_API_URL"], default_gen_temperature=self.config["DEFAULT_GEN_TEMPERATURE"], default_max_output_tokens=self.config["DEFAULT_MAX_OUTPUT_TOKENS"], default_max_input_tokens=self.config["DEFAULT_MAX_INPUT_TOKENS"])
        elif provider_name.lower() == LLMProviderEnum.COHERE.value.lower():
            return CohereProvider(api_key=self.config["COHERE_API_KEY"], default_gen_temperature=self.config["DEFAULT_GEN_TEMPERATURE"], default_max_output_tokens=self.config["DEFAULT_MAX_OUTPUT_TOKENS"])
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")
            