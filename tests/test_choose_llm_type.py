import pytest

if __name__ == "__main__":
    pytest.main([__file__])
from typing import Any

from anthropic import Anthropic
from openai import OpenAI
from pi_conf import load_config
from pydantic import BaseModel

from qrev_instructor.client import get_client

# Load configuration
load_config().to_env()


# Define the User model
class User(BaseModel):
    name: str
    age: int


def get_create_params(model: str, include_response_model: bool = True) -> dict[str, Any]:
    d = {
        "model": model,
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": "Extract Jason is 25 years old.",
            }
        ],
    }
    if include_response_model:
        d["response_model"] = User
    return d


@pytest.fixture
def openai_model():
    return "gpt-3.5-turbo"


@pytest.fixture
def anthropic_model():
    return "claude-3-haiku-20240307"


@pytest.fixture
def openai_client_model(openai_model):
    return get_client(model=openai_model), openai_model


@pytest.fixture
def anthropic_client_model(anthropic_model):
    return get_client(model=anthropic_model), anthropic_model


def test_anthropic_from_client(anthropic_model):
    client = get_client(model=anthropic_model, client=Anthropic())
    # Create a response using the client
    resp = client.messages.create(**get_create_params(anthropic_model))

    # Assertions to check the response
    assert isinstance(resp, User), "Response is not an instance of User"
    assert resp.name == "Jason", "The extracted name is incorrect"
    assert resp.age == 25, "The extracted age is incorrect"


def test_anthropic_from_wrong_client(anthropic_model, openai_model):
    """Trying to use the wrong client with the anthropic model"""
    with pytest.raises(ValueError):
        get_client(model=anthropic_model, client=OpenAI())
    with pytest.raises(ValueError):
        get_client(model=openai_model, client=Anthropic())


def test_anthropic_extract_user_info_messages(openai_client_model):
    client, model = openai_client_model

    # Create a response using the client
    resp = client.messages.create(**get_create_params(model))

    # Assertions to check the response
    assert isinstance(resp, User), "Response is not an instance of User"
    assert resp.name == "Jason", "The extracted name is incorrect"
    assert resp.age == 25, "The extracted age is incorrect"


def test_anthropic_extract_user_info(anthropic_client_model):
    client, model = anthropic_client_model

    # Create a response using the client
    resp = client.chat.completions.create(**get_create_params(model))

    # Assertions to check the response
    assert isinstance(resp, User), "Response is not an instance of User"
    assert resp.name == "Jason", "The extracted name is incorrect"
    assert resp.age == 25, "The extracted age is incorrect"


def test_openai_extract_user_info_messages(openai_client_model):
    client, model = openai_client_model

    # Create a response using the client
    resp = client.messages.create(**get_create_params(model))

    # Assertions to check the response
    assert isinstance(resp, User), "Response is not an instance of User"
    assert resp.name == "Jason", "The extracted name is incorrect"
    assert resp.age == 25, "The extracted age is incorrect"


def test_openai_extract_user_info_create(openai_client_model):
    client, model = openai_client_model

    # Create a response using the client
    resp = client.chat.completions.create(**get_create_params(model))

    # Assertions to check the response
    assert isinstance(resp, User), "Response is not an instance of User"
    assert resp.name == "Jason", "The extracted name is incorrect"
    assert resp.age == 25, "The extracted age is incorrect"


if __name__ == "__main__":
    pytest.main([__file__])
