import pytest

from qrev_instructor import APIType


def test_api_type_from_model_anthropic():
    # Test models in AnthropicModel
    assert APIType.OTHER.from_model("claude-3-opus-20240229") == APIType.ANTHROPIC
    assert APIType.OTHER.from_model("claude-3-haiku-20240307") == APIType.ANTHROPIC
    assert APIType.OTHER.from_model("claude-3-5-sonnet-20240620") == APIType.ANTHROPIC
    assert APIType.OTHER.from_model("claude-3-sonnet-20240229") == APIType.ANTHROPIC


def test_api_type_from_model_openai():
    # Test models in OpenAIModel
    assert APIType.OTHER.from_model("davinci") == APIType.OPENAI
    assert APIType.OTHER.from_model("curie") == APIType.OPENAI
    assert APIType.OTHER.from_model("gpt-3.5-turbo-0125") == APIType.OPENAI
    assert APIType.OTHER.from_model("gpt-3.5-turbo") == APIType.OPENAI
    assert APIType.OTHER.from_model("gpt-4") == APIType.OPENAI
    assert APIType.OTHER.from_model("gpt-4o") == APIType.OPENAI
    assert APIType.OTHER.from_model("gpt-4o-mini") == APIType.OPENAI
    assert APIType.OTHER.from_model("gpt-4o-2024-05-13") == APIType.OPENAI
    assert APIType.OTHER.from_model("o1-mini") == APIType.OPENAI
    assert APIType.OTHER.from_model("o1-preview") == APIType.OPENAI


def test_api_type_from_model_other():
    # Test unknown models that should default to OTHER
    assert APIType.OTHER.from_model("unknown-model") == APIType.OTHER
    assert APIType.OTHER.from_model("invalid-model") == APIType.OTHER
    assert APIType.OTHER.from_model("non-existent-model") == APIType.OTHER


if __name__ == "__main__":
    pytest.main([__file__])
