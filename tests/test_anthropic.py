import pytest

if __name__ == "__main__":
    pytest.main([__file__])


import instructor
from anthropic import Anthropic
from pi_conf import load_config
from pydantic import BaseModel

from qrev_instructor import AnthropicModel

load_config().to_env()


class User(BaseModel):
    name: str
    age: int


def test_anthropic_extract_user_info():
    client = instructor.from_anthropic(Anthropic())

    # note that client.chat.completions.create will also work
    resp = client.messages.create(
        model=AnthropicModel.CLAUDE_3_HAIKU_20240307,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Extract Jason is 25 years old.",
            }
        ],
        response_model=User,
    )

    assert isinstance(resp, User)
    assert resp.name == "Jason"
    assert resp.age == 25
