import pytest
from pydantic import BaseModel
from dataenvgym.llm_utils import (
    AzureOpenAIParallelRequestor,
    StructuredCompletionRequest,
    CompletionRequest,
)
from typing import Iterable


class Recipe(BaseModel):
    name: str
    ingredients: list[str]
    instructions: str


Recipes = Iterable[Recipe]


@pytest.fixture
def azure_helper():
    return AzureOpenAIParallelRequestor(max_workers=2)


def test_execute_structured_requests(azure_helper):
    requests = [
        StructuredCompletionRequest(
            prompt="Give me 3 recipes for pancakes.",
            model="gpt-4o",
            response_model=Recipes,
        ),
        StructuredCompletionRequest(
            prompt="Give me 4 recipes for cheesecake.",
            model="gpt-4o",
            response_model=Recipes,
        ),
    ]
    responses = azure_helper.execute_structured_requests(requests)
    assert len(responses) == len(requests)


def test_execute_requests(azure_helper):
    requests = [
        CompletionRequest(prompt="Tell me a joke.", model="gpt-4o"),
        CompletionRequest(prompt="What's the weather like today?", model="gpt-4o"),
        CompletionRequest(prompt="Give me a recipe for pancakes.", model="gpt-4o"),
        CompletionRequest(prompt="Explain quantum computing.", model="gpt-4o"),
        CompletionRequest(prompt="What is the capital of France?", model="gpt-4o"),
        CompletionRequest(prompt="How do I improve my coding skills?", model="gpt-4o"),
    ]
    responses = azure_helper.execute_requests(requests)
    assert len(responses) == len(requests)
