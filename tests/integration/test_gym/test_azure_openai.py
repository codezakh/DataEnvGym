from openai import AzureOpenAI
import os
import instructor
from pydantic import BaseModel


def test_chat_completion():
    # Setting up the deployment name
    deployment_name = "gpt-4o"

    # The API key for your Azure OpenAI resource.
    api_key = os.environ["AZURE_OPENAI_API_KEY"]

    # The base URL for your Azure OpenAI resource. e.g. "https://<your resource name>.openai.azure.com"
    azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]

    # Currently Chat Completion API have the following versions available: 2023-03-15-preview
    api_version = "2023-03-15-preview"  # os.environ["OPENAI_API_VERSION"]

    client = AzureOpenAI(
        api_key=api_key, azure_endpoint=azure_endpoint, api_version=api_version
    )

    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a poem that mentions hippos."},
        ],
    )

    assert response.choices[0].message.content
    assert "hippos" in response.choices[0].message.content


def test_structured_output_with_instructor():

    class SuperHero(BaseModel):
        name: str
        power: str

    # Setting up the deployment name
    deployment_name = "gpt-4o"

    # The API key for your Azure OpenAI resource.
    api_key = os.environ["AZURE_OPENAI_API_KEY"]

    # The base URL for your Azure OpenAI resource. e.g. "https://<your resource name>.openai.azure.com"
    azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]

    # Currently Chat Completion API have the following versions available: 2023-03-15-preview
    api_version = "2023-03-15-preview"  # os.environ["OPENAI_API_VERSION"]

    client = instructor.patch(
        AzureOpenAI(
            api_key=api_key, azure_endpoint=azure_endpoint, api_version=api_version
        )
    )

    response: SuperHero = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {
                "role": "user",
                "content": "Create a superhero named SuperHippo. Provide a detailed description of his powers.",
            },
        ],
        response_model=SuperHero,  # type: ignore
    )

    assert response.name == "SuperHippo"
