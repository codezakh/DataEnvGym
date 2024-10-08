from dataenvgym.gym.tasks.math.MATH.task import MATHTask, prepare_few_shot_prompt
from dataenvgym.gym.trainable_predictors.math.local_llm import (
    LlamaFactoryTrainer,
    ParallelLlmPredictor,
    ParallelLlmTrainablePredictor,
)
from pathlib import Path
from dataenvgym.gym.skill_discovery.math_gold_skills import AssignAnnotatedTopicsAsSkills
from dataenvgym.gym.data_generation_agents.math.bandit_data_strategy import (
    MathBanditDataStrategy,
    JsonExperienceCheckpointer,
    Explore,
)
from dataenvgym.gym.data_generation_agents.math.bandit_components import (
    ProposeSubskillsWithAzureOpenAI,
    GenerateSubskillDataWithAzureOpenAI,
    AzureOpenAIActionPolicy,
)
from vllm import SamplingParams
from dataenvgym.utils import JSONLinesKeyValueCache
from dataenvgym.gym.data_generation_agents.math.baselines.skill_list import (
    MathTrainingDatumWithSkillCategory,
)
from dataenvgym.gym.domain_models import MathTrainingDatumQualityCheck
from dataenvgym.gym.quality_checking.math.minimal import MathTrainingDataQualityChecker
import ray


def test_bandit_data_strategy_with_real_components(tmp_path: Path) -> None:
    num_gpus = 2
    ray.init(ignore_reinit_error=True, num_gpus=num_gpus)
    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    experience_checkpointer = JsonExperienceCheckpointer(
        output_path=experiment_dir / "checkpoints"
    )

    training_data_cache = JSONLinesKeyValueCache(
        file_path=experiment_dir / "training_data_cache.jsonl",
        model=MathTrainingDatumWithSkillCategory,
    )

    quality_check_cache = JSONLinesKeyValueCache(
        file_path=experiment_dir / "quality_check_cache.jsonl",
        model=MathTrainingDatumQualityCheck,
    )

    # WARNING: You should set always_apply_chat_template=True for gemma-2b-it
    # since it is an instruction tuned model with a chat template.
    predictor = ParallelLlmPredictor(
        sampling_params=SamplingParams(temperature=0.0, max_tokens=350),
        prompt_formatter_for_base_model=prepare_few_shot_prompt,
        model_name_or_path="google/gemma-2-2b-it",
        num_workers=num_gpus,
    )
    trainer = LlamaFactoryTrainer(
        working_directory=experiment_dir / "llama_factory",
        cuda_visible_devices=[1],
        overrides=["model_name_or_path=google/gemma-2-2b-it", "template=gemma"],
    )

    trainable_predictor = ParallelLlmTrainablePredictor(predictor, trainer)

    skill_discovery_module = AssignAnnotatedTopicsAsSkills()

    subskill_proposal_policy = ProposeSubskillsWithAzureOpenAI()
    subskill_data_generator = GenerateSubskillDataWithAzureOpenAI()
    action_policy = AzureOpenAIActionPolicy()
    quality_checker = MathTrainingDataQualityChecker()

    training_data_production_strategy = MathBanditDataStrategy(
        skill_discovery_module=skill_discovery_module,
        logging_folder=experiment_dir / "data_strategy_outputs",
        subskill_data_generator=subskill_data_generator,
        action_policy=action_policy,
        subskill_proposal_policy=subskill_proposal_policy,
        experience_checkpointer=experience_checkpointer,
        training_data_cache=training_data_cache,
        initial_explore_action=Explore(
            num_new_skills=3, data_allocation_for_new_skills=10
        ),
        quality_check_cache=quality_check_cache,
        training_data_quality_checker=quality_checker,
    )

    task = MATHTask(split="algebra_test_500")

    completed_task_instances = task.evaluate(trainable_predictor)

    training_data = []

    for _ in range(3):
        training_data_for_iteration = training_data_production_strategy(
            completed_task_instances=completed_task_instances,
            predictor=trainable_predictor,
        )
        training_data.extend(training_data_for_iteration)

    assert len(training_data) > 0
