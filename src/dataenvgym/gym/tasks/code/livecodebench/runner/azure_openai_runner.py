import os
import json
from abc import ABC, abstractmethod

from tqdm import tqdm
from typing import Callable

from ..lm_styles import LanguageModel
from ..utils.path_utils import get_cache_path
from ..utils.multiprocess import run_tasks_in_parallel
from ..runner.scenario_router import Scenario
import openai
from openai import OpenAI
from time import sleep
from typing_extensions import TypedDict
from typing import Any

Prompt = list[dict[str, str]]


class RunSingleArgs(TypedDict):
    prompt: Prompt
    call_method: Callable[[Prompt], list[str]]


class OpenAiRunner:
    def __init__(
        self,
        model: LanguageModel,
        multiprocess: int = 0,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        top_p: float = 0.95,
        n: int = 10,
    ):
        self.model = model
        self.client = OpenAI()
        self.model_name = model.model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.n = n
        self.multiprocess = multiprocess

    def _run_single(self, prompt: Prompt) -> list[str]:
        assert isinstance(prompt, list)

        try:
            response = self.client.chat.completions.create(
                messages=prompt,  # type: ignore
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                n=self.n,
            )
        except (
            openai.APIError,
            openai.RateLimitError,
            openai.InternalServerError,
            openai.OpenAIError,
            openai.APIStatusError,
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.APIConnectionError,
        ) as e:
            print("Exception: ", repr(e))
            print("Sleeping for 30 seconds...")
            print("Consider reducing the number of parallel processes.")
            sleep(30)
            return self._run_single(prompt)
        except Exception as e:
            print(f"Failed to run the model for {prompt}!")
            print("Exception: ", repr(e))
            raise e
        return [c.message.content for c in response.choices]

    @staticmethod
    def run_single(combined_args: RunSingleArgs) -> list[str]:
        """
        Run the model for a single prompt and return the output
        Static method to be used in multiprocessing
        Calls the _run_single method with the combined arguments
        """

        prompt = combined_args["prompt"]
        call_method = combined_args["call_method"]

        result = call_method(prompt)

        return result

    def run_batch(self, prompts: list[Prompt]) -> list[list[str]]:
        outputs = []
        arguments: list[RunSingleArgs] = [
            {
                "prompt": prompt,
                "call_method": self._run_single,
            }
            for prompt in prompts
        ]
        if self.multiprocess > 1:
            parallel_outputs = run_tasks_in_parallel(
                self.run_single,
                arguments,
                self.multiprocess,
                use_progress_bar=True,
            )
            for output in parallel_outputs:
                if output.is_success():
                    outputs.append(output.result)
                else:
                    print("Failed to run the model for some prompts")
                    print(output.status)
                    print(output.exception_tb)
                    outputs.extend([""] * self.n)
        else:
            outputs = [self.run_single(argument) for argument in tqdm(arguments)]

        return outputs

    def prompts_to_outputs(self, prompts: list[Prompt]) -> list[list[str]]:
        outputs = self.run_batch(prompts)
        return outputs

    def run_main(self, benchmark: list, format_prompt: Callable) -> list[list[str]]:
        prompts = [
            format_prompt(problem, self.model.model_style) for problem in benchmark
        ]
        outputs = self.prompts_to_outputs(prompts)
        return outputs
