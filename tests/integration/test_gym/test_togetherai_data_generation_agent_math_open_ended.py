from dataenvgym.gym.data_generation_agents.math.baselines.open_ended import (
    DataGenerationAgent,
)
from pathlib import Path

from dataenvgym.gym.domain_models import (
    MathPredictorInterface,
    CompletedMathTaskInstance,
    MathTaskInstance,
    ULID,
)


def test_data_generation_agent_works(
    tmp_path: Path, stub_math_predictor: MathPredictorInterface
):
    agent = DataGenerationAgent(
        model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        logging_folder=tmp_path,
        data_specs_per_llm_call=1,
        num_training_data_per_invocation=1,
        num_examples_for_datagen_prompt=1,
    )

    completed_task_instances = [
        CompletedMathTaskInstance(
            ulid=ULID(),
            task_instance=MathTaskInstance(
                task_name="logarithm_equation",
                instance_id="log_eq_001",
                instruction=r"""Find all values of x that satisfy the equation:
            \[\log_2(x + 3) + \log_2(x - 1) = 3\]
            Express your answer as a comma-separated list of values in ascending order.""",
                ground_truth_label="2, 5",
            ),
            predictor_response="2, 5",
            was_correct=True,
        ),
        CompletedMathTaskInstance(
            ulid=ULID(),
            task_instance=MathTaskInstance(
                task_name="logarithm_equation",
                instance_id="log_eq_001",
                instruction=r"""Find all values of x that satisfy the equation:
            \[\log_2(x + 3) + \log_2(x - 1) = 3\]
            Express your answer as a comma-separated list of values in ascending order.""",
                ground_truth_label="2, 5",
            ),
            predictor_response="16",
            was_correct=False,
        ),
    ]

    generated_training_data = agent(
        completed_task_instances=completed_task_instances,
        predictor=stub_math_predictor,
    )

    assert len(generated_training_data) == 1
