from typing import Any, Iterable, Optional, TypeVar, Union

import instructor
from pydantic import BaseModel

from qrev_instructor.apis import get_api_service, get_apis
from qrev_instructor.models import APIModel, APIType

T = TypeVar("T", bound=Union[BaseModel, "Iterable[Any]"])

apis = get_apis()


def _get_client(
    model: Optional[str | APIModel] = None,
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


def get_client(
    model: Optional[str | APIModel] = None,
    llm_type: Optional[str] = None,
    client_params: Optional[dict[str, Any]] = None,
    func_params: Optional[dict[str, Any]] = None,
) -> instructor.Instructor:
    client = _get_client(model, llm_type, client_params, func_params)
    return client
