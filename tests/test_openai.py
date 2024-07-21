import pytest

if __name__ == "__main__":
    pytest.main([__file__])


import instructor
from openai import OpenAI
from pi_conf import load_config
from pydantic import BaseModel

from qrev_instructor import OpenAIModel

load_config().to_env()


class User(BaseModel):
    name: str
    age: int

def test_anthropic_extract_user_info():
    client = instructor.from_openai(OpenAI())

    # note that client.chat.completions.create will also work
    resp = client.messages.create(
        model=OpenAIModel.GPT_4O_MINI,
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
