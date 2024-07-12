from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Optional

import instructor


class CaseInsensitiveEnum(StrEnum):
    @classmethod
    def _missing_(cls, value):
        value_str = str(value)  # Ensure the value is a string
        for member in cls:
            if member.value.lower() == value_str.lower():
                return member
        raise ValueError(f"{value} is not a valid {cls.__name__}")


class APIType(CaseInsensitiveEnum):
    OTHER = "Other"
    ANTHROPIC = "Anthropic"
    OPENAI = "OpenAI"

class APIModel(CaseInsensitiveEnum):
    pass

class AnthropicModels(APIModel):
    CLAUDE_3_OPUS_20240229 = "claude-3-opus-20240229"
    CLAUDE_3_HAIKU_20240307 = "claude-3-haiku-20240307"
    CLAUDE_3_5_SONNET_20240620 = "claude-3-5-sonnet-20240620"
    CLAUDE_3_SONNET_20240229 = "claude-3-sonnet-20240229"


class OpenAIModels(APIModel):
    GPT_3_5_TURBO_0125 = "gpt-3.5-turbo-0125"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_3_5_TURBO_INSTRUCT = "gpt-3.5-turbo-instruct"
    GPT_4 = "gpt-4"
    GPT_4O = "gpt-4o"
    GPT_4O_2024_05_13 = "gpt-4o-2024-05-13"
    DAVINCI = "davinci"
    CURIE = "curie"


def get_api_service(model_name: str) -> APIType:
    model_name = model_name.lower()
    if model_name in AnthropicModels._value2member_map_:
        return APIType.ANTHROPIC
    elif model_name in OpenAIModels._value2member_map_:
        return APIType.OPENAI
    model_name = model_name.replace("-", "_").replace(".", "_").upper()
    if model_name in AnthropicModels._member_names_:
        return APIType.ANTHROPIC
    elif model_name in OpenAIModels._member_names_:
        return APIType.OPENAI
    else:
        raise NotImplementedError(f"Model {model_name} not supported")


@dataclass
class FuncApiWrapper:
    """
    Wrapper for the API function call and class
    """

    from_class: type[Any]
    from_func: Callable[..., Any]


apis: dict[APIType, FuncApiWrapper] = {}
try:
    from openai import OpenAI

    apis[APIType.OPENAI] = FuncApiWrapper(OpenAI, instructor.from_openai)

except ImportError:
    pass

try:
    from anthropic import Anthropic

    apis[APIType.ANTHROPIC] = FuncApiWrapper(Anthropic, instructor.from_anthropic)

except ImportError:
    pass


def get_client(
    model: Optional[str | APIModel ] = None,
    llm_type: Optional[str] = None,
    client_params: Optional[dict[str, Any]] = None,
    func_params: Optional[dict[str, Any]] = None,
) -> instructor.Instructor:
    if model and llm_type:
        raise ValueError(f"Cannot specify both model and llm_type. Got {model} and {llm_type}")
    _llm_type = APIType.OPENAI
    if model:
        _llm_type = get_api_service(str(model))
    elif llm_type:
        _llm_type = APIType(llm_type)
    if client_params is None:
        client_params = {}
    if func_params is None:
        func_params = {}
    wrap = apis[_llm_type]
    instance = wrap.from_class(**client_params)
    client = wrap.from_func(instance, **func_params)
    return client
