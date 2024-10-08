from typing import Protocol, Type, Generic, TypeVar, Optional
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import AzureOpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential
from dataenvgym.gym.domain_models import implements
import instructor
from loguru import logger
import time
from dataclasses import dataclass
from openai._types import NotGiven
from openai.types.chat import ChatCompletionMessageParam
from typing import Sequence

T = TypeVar("T")


# Note: This _has_ to be a dataclass, it cannot be a Pydantic model.
# This is because of the type annotation on the response_model field.
# Pydantic does not accept generic types like Iterable[SomethingModel],
# but Instructor requires you to wrap your response model in a generic type
# if you're requesting something like a list. You can't do list[SomethingModel].
@dataclass
class StructuredCompletionRequest(Generic[T]):
    prompt: str
    model: str
    response_model: Type[T]
    temperature: float | NotGiven = NotGiven()
    max_tokens: int | NotGiven = NotGiven()
    top_p: float | NotGiven = NotGiven()
    n: int | NotGiven = NotGiven()


class CompletionRequest(BaseModel):
    messages: Sequence[ChatCompletionMessageParam]
    model: str
    temperature: float | NotGiven = NotGiven()
    max_tokens: int | NotGiven = NotGiven()
    top_p: float | NotGiven = NotGiven()
    n: int | NotGiven = NotGiven()


class ParallelLLMRequestor(Protocol, Generic[T]):
    def execute_structured_requests(
        self, requests: list[StructuredCompletionRequest]
    ) -> list[T]: ...

    def execute_requests(
        self, requests: list[CompletionRequest]
    ) -> list[Optional[str]]: ...


class AzureOpenAIParallelRequestor(Generic[T]):
    def __init__(self, max_workers: int = 5):
        self.client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version="2023-03-15-preview",
        )
        self.patched_client = instructor.patch(
            AzureOpenAI(
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_version="2023-03-15-preview",
            )
        )
        self.max_workers = max_workers

    @retry(
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(3),
    )
    def _make_structured_request(self, request: StructuredCompletionRequest) -> T:
        # Implement the actual API call here
        response = self.patched_client.chat.completions.create(
            model="gpt-4o",
            response_model=request.response_model,  # type: ignore
            messages=[{"role": "user", "content": request.prompt}],
        )
        return response

    @retry(
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(3),
    )
    def _make_request(self, request: CompletionRequest) -> Optional[str]:
        response = self.client.chat.completions.create(
            model=request.model,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            n=request.n,
        )
        return response.choices[0].message.content

    def execute_structured_requests(
        self, requests: list[StructuredCompletionRequest]
    ) -> list[T]:
        logger.info(
            f"Executing {len(requests)} structured requests with {self.max_workers} workers."
        )
        responses: list[T] = []
        start_time = time.time()  # Start time for the entire batch
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_request = {
                executor.submit(self._make_structured_request, req): req
                for req in requests
            }
            for future in as_completed(future_to_request):
                try:
                    response = future.result()
                    responses.append(response)
                except Exception:
                    logger.opt(exception=True).error(
                        "Failed to get structured response for request."
                    )
        end_time = time.time()  # End time for the entire batch
        total_time = end_time - start_time
        average_time = total_time / len(requests) if requests else 0
        logger.info(
            f"Finished executing {len(requests)} structured requests with {self.max_workers} workers. Average time per request: {average_time:.2f} seconds."
        )
        return responses

    def execute_requests(
        self, requests: list[CompletionRequest]
    ) -> list[Optional[str]]:
        logger.info(
            f"Executing {len(requests)} unstructured requests with {self.max_workers} workers."
        )
        responses: list[Optional[str]] = []
        start_time = time.time()  # Start time for the entire batch
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_request = {
                executor.submit(self._make_request, req): req for req in requests
            }
            for future in as_completed(future_to_request):
                try:
                    response = future.result()
                    responses.append(response)
                except Exception:
                    logger.opt(exception=True).error(
                        "Failed to get response for request."
                    )
        end_time = time.time()  # End time for the entire batch
        total_time = end_time - start_time
        average_time = total_time / len(requests) if requests else 0
        logger.info(
            f"Finished executing {len(requests)} unstructured requests with {self.max_workers} workers. Average time per request: {average_time:.2f} seconds."
        )
        return responses


implements(ParallelLLMRequestor)(AzureOpenAIParallelRequestor)
