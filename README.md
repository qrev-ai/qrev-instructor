# QRev Instructor

QRev Instructor is a Python wrapper around the [`instructor`](https://github.com/jxnl/instructor) module, providing a unified interface for working with different language models from OpenAI and Anthropic.

## Features

- Supports both OpenAI and Anthropic models
- Easy-to-use client initialization
- Automatic model type detection
- Case-insensitive enum handling
- Extensible for other API types

## Installation

To install QRev Instructor, use pip:

```pip install qrev-instructor```

For Anthropic (Claude models)

```pip install qrev-instructor[anthropic]```

## Usage

Here's a basic example of how to use QRev Instructor:

```python
from qrev_instructor import get_client
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

model_name="gpt-3.5-turbo"
# Initialize the client
client = get_client(model=model_name)

# Use the client to create a response
response = client.messages.create(
    model=model_name,
    messages=[
        {
            "role": "user",
            "content": "Jason is 25 years old.",
        }
    ],
    response_model=User
)

print(f"Name: {response.name}, Age: {response.age}")
# prints "Name: Jason, Age: 25"
```

## Supported Models

OpenAI Models:
- GPT-3.5-Turbo (various versions)
- GPT-4 (various versions)
- Davinci
- Curie

Anthropic Models:
- Claude-3-Opus
- Claude-3-Haiku
- Claude-3.5-Sonnet

## Testing

The package includes pytest-based tests for both OpenAI and Anthropic clients. To run the tests:

pytest path/to/test_file.py

## Dependencies

- `instructor`
- `openai` (optional, for OpenAI models)
- `anthropic` (optional, for Anthropic models)
- `pydantic`

## License

[Your chosen license]

## Contributing

[Your contribution guidelines]

## Support

[Your support information]