import pytest

if __name__ == "__main__":
    pytest.main([__file__])
from typing import Any

from anthropic import Anthropic
from pi_conf import load_config
from pydantic import BaseModel

from qrev_instructor.client import get_client

# Load configuration
load_config().to_env()


# Define the User model
class User(BaseModel):
    name: str
    age: int


@pytest.fixture
def openai_client_model():
    model = "gpt-3.5-turbo"
    return get_client(model=model), model


@pytest.fixture
def anthropic_client_model():
    model = "claude-3-haiku-20240307"
    return get_client(model=model), model


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
