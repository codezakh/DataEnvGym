from dataenvgym.gym.domain_models import (
    CodeGenerationPredictorInterface,
    CodeGenerationTaskInstance,
    implements,
)
from dataenvgym.gym.tasks.code.livecodebench.lm_styles import LMStyle
from dataenvgym.gym.tasks.code.livecodebench.prompts.code_generation import (
    format_prompt_generation,
    PromptConstants,
)
from openai import OpenAI, AzureOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from typing import Sequence, Optional
from tqdm.auto import tqdm
from dataenvgym.gym.tasks.code.livecodebench.utils.extraction_utils import extract_code
from tenacity import retry, stop_after_attempt, wait_random_exponential


def get_generic_question_template_answer(
    question: str, starter_code: Optional[str] = None
) -> str:
    prompt = f"### Question:\n{question}\n\n"
    if starter_code:
        prompt += (
            f"### Format: {PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        )
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def format_prompt(
    task_instance: CodeGenerationTaskInstance,
) -> Sequence[ChatCompletionMessageParam | ChatCompletionSystemMessageParam]:
    chat_messages = [
        ChatCompletionSystemMessageParam(
            role="system",
            content=PromptConstants.SYSTEM_MESSAGE_GENERIC,
        ),
    ]
    chat_messages += [
        ChatCompletionUserMessageParam(
            role="user",
            content=get_generic_question_template_answer(
                question=task_instance.instruction,
                starter_code=task_instance.starter_code,
            ),
        ),
    ]
    return chat_messages


READ_FROM_STDIN_WARNING = """
If told to read from stdin and write to stdout, DO NOT write a function.
Instead, call input() to read from stdin and print() to write to stdout as a standalone code block.
Do not put your code under an if __name__ == "__main__": block.
Example:
```python
# If told to read from stdin and write to stdout
# DO NOT WRITE A FUNCTION!
my_inputs = input() # read from stdin
output = ... # do some computations
print(output) # write to stdout
```
"""


@retry(
    wait=wait_random_exponential(min=1, max=30),
    stop=stop_after_attempt(3),
)
def answer_code_generation_question(
    client: OpenAI | AzureOpenAI,
    question: str,
    starter_code: Optional[str] = None,
) -> str:

    prompt = get_generic_question_template_answer(
        question=question,
        starter_code=starter_code,
    )

    chat_messages = [
        ChatCompletionSystemMessageParam(
            role="system",
            content=PromptConstants.SYSTEM_MESSAGE_GENERIC
            + "\n"
            + READ_FROM_STDIN_WARNING,
        ),
    ]
    chat_messages += [
        ChatCompletionUserMessageParam(
            role="user",
            content=prompt,
        ),
    ]
    response = client.chat.completions.create(
        messages=chat_messages,
        model="gpt-4o",
        temperature=0.2,
        max_tokens=2000,
        top_p=0.95,
        n=1,
    )
    if response.choices[0].message.content is None:
        raise ValueError("No response from OpenAI")
    return response.choices[0].message.content


class OpenAICodeGenerationPredictor:
    def __init__(self):
        self.client = OpenAI()

    def predict(
        self, task_instances: Sequence[CodeGenerationTaskInstance]
    ) -> list[str]:
        prompts = [format_prompt(task_instance) for task_instance in task_instances]
        code_generations = []
        for prompt in tqdm(prompts):
            response = self.client.chat.completions.create(
                messages=prompt,
                model="gpt-4o-mini",
                temperature=0.2,
                max_tokens=2000,
                top_p=0.95,
                n=1,
            )
            if response.choices[0].message.content is None:
                code = "raise ValueError('No response from OpenAI')"
                code_generations.append(code)
            else:
                code = extract_code(
                    model_output=response.choices[0].message.content,
                    lmstyle=LMStyle.OpenAIChat,
                )
                code_generations.append(code)
        return code_generations
