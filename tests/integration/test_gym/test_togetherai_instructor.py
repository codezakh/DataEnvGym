import os
import openai
from pydantic import BaseModel
import instructor


def test_llama3_works_with_together_instructor():
    client = openai.OpenAI(
        base_url="https://api.together.xyz/v1",
        api_key=os.environ["TOGETHER_API_KEY"],
    )

    # By default, the patch function will patch the ChatCompletion.create and ChatCompletion.create methods to support the response_model parameter
    client = instructor.from_openai(client, mode=instructor.Mode.TOOLS)

    # Now, we can use the response_model parameter using only a base model
    # rather than having to use the OpenAISchema class
    class UserExtract(BaseModel):
        name: str
        age: int

    user: UserExtract = client.chat.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        response_model=UserExtract,
        messages=[
            {"role": "user", "content": "Extract jason is 25 years old"},
        ],
    )

    assert isinstance(user, UserExtract), "Should be instance of UserExtract"
    assert user.name.lower() == "jason"
    assert user.age == 25
