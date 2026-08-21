from enum import Enum

class LLMProviderEnum(Enum):
    
    OPENAI = "OPENAI"
    COHERE = "COHERE"


class OpenAIMessageRoleEnum(Enum):

    USER = "user"
    DEVELOPER = "developer"
    ASSISTANT = "assistant"


class CohereMessageRoleEnum(Enum):

    USER = "user"
    DEVELOPER = "developer"
    SYSTEM = "system"
    TOOL = "tool"


class DocTypeEnum(Enum):

    DOCUMENT = "document"
    QUERY = "query"

class CohereInputTypeEnum(Enum):

    DOCUMENT = "search_document"
    QUERY = "search_query"
