from pydantic import BaseModel, Field
from ulid import ULID
from typing import (
    NewType,
    Collection,
    Sequence,
    Optional,
    Literal,
    Union,
    Annotated,
    Protocol,
    MutableMapping,
    Mapping,
    Iterator,
    TypeVar,
    Generic,
)
from dataenvgym.gym.domain_models import (
    TrainingDatum,
    TaskInstance,
    CompletedTaskInstance,
    TrainingDatumCovariant,
    PredictorInterface,
    SkillDiscoveryInterface,
    QualityCheckerInterface,
    TrainingDataQualityCheckCovariant,
    VqaTrainingDatum,
    CompletedVqaTaskInstance,
    VqaTrainingDataQualityCheck,
    MathTrainingDatum,
    CompletedMathTaskInstance,
    MathTaskInstance,
    MathTrainingDatumQualityCheck,
    CodeGenerationTrainingDatum,
    CodeGenerationCompletedTaskInstance,
    CodeGenerationTrainingDataQualityCheck,
    OpenEndedVqaTaskInstance,
    CodeGenerationTaskInstance,
)
import numpy as np
from typing_extensions import Self
from collections import defaultdict
from copy import deepcopy
from loguru import logger

Subskill = NewType("Subskill", str)
Skill = NewType("Skill", str)


class SkillTree(BaseModel):
    subskills: list[Subskill]
    data_allocation: dict[Subskill, int]
    training_data: dict[Subskill, list[ULID]]
    skill: Skill
    quality_checks: dict[Subskill, list[ULID]]
    # This can have a value of None for a subskill because it is
    # possible that we have no data for a subskill due to a failure
    # in the data generation process or the data was too badly malformed
    # and we skipped quality checking.
    perf_on_training_data: dict[Subskill, float | None]

    @property
    def total_data_allocated(self) -> int:
        # Count the actual data allocated, not just the sum of the data allocation.
        return sum(len(data) for data in self.training_data.values())


class PartialSkillTree(BaseModel):
    subskills: list[Subskill]
    data_allocation: dict[Subskill, int]
    skill: Skill
    # This is optional because when we start we don't have any training data
    # but we still want to be able to propose new subskills.
    training_data: Optional[dict[Subskill, list[ULID]]] = None
    # Since we don't have any training data yet, we don't have any quality checks.
    quality_checks: Optional[dict[Subskill, list[ULID]]] = None
    # This is generated once we have training data and quality checks.
    perf_on_training_data: Optional[dict[Subskill, float | None]] = None

    def finalize(self) -> SkillTree:
        assert self.training_data is not None
        assert self.quality_checks is not None
        assert self.perf_on_training_data is not None
        return SkillTree(
            subskills=self.subskills,
            data_allocation=self.data_allocation,
            training_data=self.training_data,
            skill=self.skill,
            quality_checks=self.quality_checks,
            perf_on_training_data=self.perf_on_training_data,
        )

    @classmethod
    def copy_from_skill_tree(cls, skill_tree: SkillTree) -> Self:
        """
        Create a new PartialSkillTree that is a deep copy of the given SkillTree
        but with no training data.
        """
        return cls(
            subskills=deepcopy(skill_tree.subskills),
            data_allocation=deepcopy(skill_tree.data_allocation),
            skill=skill_tree.skill,
            training_data=None,
            quality_checks=None,
        )

    @property
    def total_data_allocated(self) -> int:
        return sum(self.data_allocation.values())


class Exploit(BaseModel):
    action_type: Literal["exploit"] = "exploit"
    data_allocation_delta: dict[Subskill, int]
    reasoning: Optional[str] = None


class Explore(BaseModel):
    action_type: Literal["explore"] = "explore"
    num_new_skills: int
    data_allocation_for_new_skills: int
    reasoning: Optional[str] = None


Action = Annotated[Union[Exploit, Explore], Field(discriminator="action_type")]


class SkillState(BaseModel):
    skill_tree: SkillTree
    past_performance: float

    @property
    def training_accuracy(self) -> float:
        total_weight = 0
        weighted_accuracy_sum = 0.0

        for subskill, allocation in self.skill_tree.data_allocation.items():
            performance = self.skill_tree.perf_on_training_data.get(subskill)
            if performance is not None:
                weighted_accuracy_sum += performance * allocation
                total_weight += allocation

        if total_weight == 0:
            return 0.0  # Return 0 if no data is allocated

        return weighted_accuracy_sum / total_weight


class SkillExperience(BaseModel):
    state: SkillState
    reward: float
    action: Action
    skill: Skill

    def get_performance(self) -> float:
        return self.state.past_performance + self.reward


class PartialSkillExperience(BaseModel):
    state: SkillState
    skill: Skill
    action: Action
    reward: Optional[float] = None

    def finalize(self) -> SkillExperience:
        assert self.reward is not None
        return SkillExperience(
            state=self.state,
            reward=self.reward,
            action=self.action,
            skill=self.skill,
        )


SkillHistory = Sequence[Mapping[Skill, SkillExperience]]


def sum_data_generated_over_forest_history(
    skill_history: SkillHistory,
) -> dict[Skill, dict[Subskill, int]]:

    # The number of skills never change over time so it is safe
    # to prefill the dictionary with the skills in the first experience.
    data_generated: dict[Skill, dict[Subskill, int]] = {
        _: defaultdict(int) for _ in skill_history[0]
    }

    # Sum the data generated for each skill and subskill.
    for experience in skill_history:
        for skill, experience in experience.items():
            for (
                subskill,
                training_data_ulids,
            ) in experience.state.skill_tree.training_data.items():
                data_generated[skill][subskill] += len(training_data_ulids)

    return data_generated


def sum_data_generated_over_tree_history(
    skill_history: Sequence[SkillExperience],
) -> dict[Subskill, int]:
    data_generated: dict[Subskill, int] = defaultdict(int)
    for experience in skill_history:
        for (
            subskill,
            training_data_ulids,
        ) in experience.state.skill_tree.training_data.items():
            data_generated[subskill] += len(training_data_ulids)
    return data_generated


TrainingDataCache = MutableMapping[ULID, TrainingDatum]


class HasUlid(BaseModel):
    ulid: ULID


HasUlidType = TypeVar("HasUlidType", bound=HasUlid)


class InMemoryKeyValueCache(MutableMapping[ULID, HasUlidType]):
    def __init__(self):
        self.data: dict[ULID, HasUlidType] = {}

    def __setitem__(self, key: ULID, value: HasUlidType) -> None:
        self.data[key] = value

    def __getitem__(self, key: ULID) -> HasUlidType:
        return self.data[key]

    def __iter__(self) -> Iterator[ULID]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __delitem__(self, key: ULID) -> None:
        del self.data[key]

    def keys(self) -> Iterator[ULID]:
        return iter(self.data)

    def values(self) -> Iterator[HasUlidType]:
        return iter(self.data.values())

    def items(self) -> Iterator[tuple[ULID, HasUlidType]]:
        return iter(self.data.items())


class SubskillDataGenerationEngineInterface(Protocol[TrainingDatumCovariant]):
    def __call__(
        self,
        subskill: Subskill,
        data_budget: int,
    ) -> Sequence[TrainingDatumCovariant]: ...


class ActionPolicy(Protocol):
    def __call__(self, history: Sequence[SkillExperience]) -> Action: ...


class StubActionPolicy:
    def __init__(self, fixed_action: Action):
        self.fixed_action = fixed_action

    def __call__(self, history: Sequence[SkillExperience]) -> Action:
        return self.fixed_action


class SubskillProposalPolicy(Protocol):
    def __call__(
        self, skill_tree: SkillTree | PartialSkillTree, num_new_subskills: int
    ) -> Sequence[Subskill]: ...


class StubSubskillProposalPolicy:
    def __init__(self):
        # Keep a counter so we never generate the same subskill twice.
        self.counter = 0

    def __call__(
        self, skill_tree: SkillTree | PartialSkillTree, num_new_subskills: int
    ) -> Sequence[Subskill]:
        subskills = [
            Subskill(f"{skill_tree.skill}_{self.counter + i}")
            for i in range(num_new_subskills)
        ]
        self.counter += num_new_subskills
        return subskills


class ExperienceCheckpointerProtocol(Protocol):
    def save_checkpoint(
        self,
        past_experiences: Sequence[Mapping[Skill, SkillExperience]],
        current_experiences: Optional[Mapping[Skill, PartialSkillExperience]] = None,
    ) -> None: ...

    def load_checkpoint(
        self,
    ) -> tuple[
        Sequence[Mapping[Skill, SkillExperience]],
        Optional[Mapping[Skill, PartialSkillExperience]],
    ]: ...


class BaseDataGenerationAgent(
    Generic[
        TaskInstance,
        TrainingDatumCovariant,
        TrainingDataQualityCheckCovariant,
        CompletedTaskInstance,
    ]
):
    def __init__(
        self,
        skill_discovery_module: SkillDiscoveryInterface,
        training_data_quality_checker: QualityCheckerInterface[
            TrainingDatumCovariant,
            TrainingDataQualityCheckCovariant,
            TaskInstance,
        ],
        subskill_data_generation_engine: SubskillDataGenerationEngineInterface[
            TrainingDatumCovariant
        ],
        training_data_cache: MutableMapping[
            ULID, TrainingDatumCovariant
        ] = InMemoryKeyValueCache(),
        action_policy: ActionPolicy = StubActionPolicy(
            Explore(num_new_skills=1, data_allocation_for_new_skills=1)
        ),
        subskill_proposal_policy: SubskillProposalPolicy = StubSubskillProposalPolicy(),
        initial_explore_action: Explore = Explore(
            num_new_skills=10, data_allocation_for_new_skills=10
        ),
        experience_checkpointer: Optional[ExperienceCheckpointerProtocol] = None,
        quality_check_cache: MutableMapping[
            ULID, TrainingDataQualityCheckCovariant
        ] = InMemoryKeyValueCache(),
    ):
        self.skill_discovery_module = skill_discovery_module
        self.generation_index = 0
        self.current_experiences: Optional[dict[Skill, PartialSkillExperience]] = None
        self.past_experiences: list[dict[Skill, SkillExperience]] = []
        self.subskill_data_generator = subskill_data_generation_engine
        self.training_data_cache = training_data_cache
        self.action_policy = action_policy
        self.subskill_proposal_policy = subskill_proposal_policy
        self.initial_explore_action = initial_explore_action
        self.experience_checkpointer = experience_checkpointer
        self.training_data_quality_checker = training_data_quality_checker
        self.quality_check_cache = quality_check_cache

    def save_checkpoint(self) -> None:
        if self.experience_checkpointer:
            self.experience_checkpointer.save_checkpoint(
                past_experiences=self.past_experiences,
                current_experiences=self.current_experiences,
            )
            logger.info(
                "Saving checkpoint. num_past_experiences={} checkpointer={}",
                len(self.past_experiences),
                self.experience_checkpointer,
            )
        else:
            logger.info("No experience checkpointer found. Skipping checkpoint.")

    def get_current_skill_forest(self) -> Mapping[Skill, SkillTree]:
        assert self.current_experiences is not None
        return {
            skill: experience.state.skill_tree
            for skill, experience in self.current_experiences.items()
        }

    def create_initial_skill_forest(
        self, completed_task_instances: Collection[CompletedTaskInstance]
    ) -> dict[Skill, PartialSkillTree]:
        required_skills = self.determine_required_skills(completed_task_instances)
        skill_trees: dict[Skill, PartialSkillTree] = {}
        for skill, completed_task_instances in required_skills.items():
            skill_trees[skill] = PartialSkillTree(
                subskills=[],
                data_allocation={},
                skill=skill,
            )
        logger.info(
            "Created initial skill forest. num_required_skills={}",
            len(required_skills),
        )
        return skill_trees

    def get_training_data_by_ulid(self, ulid: ULID) -> TrainingDatumCovariant:
        return self.training_data_cache[ulid]

    def get_quality_check_by_ulid(
        self, ulid: ULID
    ) -> TrainingDataQualityCheckCovariant:
        return self.quality_check_cache[ulid]

    def pull_training_data_for_skill_forest(
        self, skill_forest: Mapping[Skill, SkillTree]
    ) -> Sequence[TrainingDatumCovariant]:
        logger.info(
            "Pulling training data for skill forest. expected_training_data={}",
            sum(len(skill_tree.training_data) for skill_tree in skill_forest.values()),
        )
        training_data: list[TrainingDatumCovariant] = []
        for skill_tree in skill_forest.values():
            for subskill in skill_tree.subskills:
                training_data.extend(
                    self.get_training_data_by_ulid(ulid)
                    for ulid in skill_tree.training_data[subskill]
                )

        logger.info("Pulled training data. num_pulled_data={}", len(training_data))
        return training_data

    def generate_data_for_subskill(
        self,
        subskill: Subskill,
        data_budget: int,
    ) -> Sequence[TrainingDatumCovariant]:
        logger.info(
            "Generating data for subskill. subskill={} data_budget={}",
            subskill,
            data_budget,
        )
        training_data = self.subskill_data_generator(subskill, data_budget)
        logger.info(
            "Generated data for subskill. subskill={} num_training_data={}",
            subskill,
            len(training_data),
        )
        return training_data

    def generate_data_for_skill_tree(
        self, skill_tree: PartialSkillTree, predictor: PredictorInterface[TaskInstance]
    ) -> PartialSkillTree:
        logger.info(
            "Generating data for skill tree. subskills={}, data_allocation={}",
            len(skill_tree.subskills),
            skill_tree.data_allocation,
        )

        for subskill in skill_tree.subskills:
            if skill_tree.training_data is None:
                skill_tree.training_data = {}

            assert skill_tree.training_data is not None

            training_data = self.generate_data_for_subskill(
                subskill, skill_tree.data_allocation[subskill]
            )

            for training_datum in training_data:
                self.training_data_cache[training_datum.ulid] = training_datum

            skill_tree.training_data[subskill] = [
                training_datum.ulid for training_datum in training_data
            ]

            if skill_tree.quality_checks is None:
                skill_tree.quality_checks = {}

            quality_checks = self.training_data_quality_checker(
                training_data, predictor
            )

            skill_tree.quality_checks[subskill] = [
                quality_check.ulid for quality_check in quality_checks
            ]

            for quality_check in quality_checks:
                self.quality_check_cache[quality_check.ulid] = quality_check

            if skill_tree.perf_on_training_data is None:
                skill_tree.perf_on_training_data = {}

            skill_tree.perf_on_training_data[subskill] = float(
                np.mean(
                    [
                        _.student_accuracy
                        for _ in quality_checks
                        if _.student_accuracy is not None
                    ]
                )
            )

        return skill_tree

    def exploit_skill_tree(
        self, skill_tree: SkillTree | PartialSkillTree, exploit: Exploit
    ) -> None:
        # Update the skill tree with the new data allocation
        logger.info(
            "Exploiting skill tree. skill={} data_allocation={}",
            skill_tree.skill,
            skill_tree.data_allocation,
        )
        skill_tree.data_allocation = {
            subskill: skill_tree.data_allocation[subskill]
            + exploit.data_allocation_delta[subskill]
            for subskill in skill_tree.subskills
        }
        logger.info(
            "Exploited skill tree. skill={} data_allocation={}",
            skill_tree.skill,
            skill_tree.data_allocation,
        )

    def explore_skill_tree(
        self, skill_tree: SkillTree | PartialSkillTree, explore: Explore
    ) -> None:
        # Propose new subskills
        new_subskills = self.subskill_proposal_policy(
            skill_tree, explore.num_new_skills
        )
        logger.info(
            "Exploring skill tree. skill={} new_subskills={}",
            skill_tree.skill,
            new_subskills,
        )
        # Update the skill tree with the new subskills
        skill_tree.subskills.extend(new_subskills)
        # Update the data allocation for the new subskills
        for subskill in new_subskills:
            skill_tree.data_allocation[subskill] = (
                explore.data_allocation_for_new_skills
            )
        logger.info(
            "Explored skill tree and allocated data. skill={} data_allocation={}",
            skill_tree.skill,
            skill_tree.data_allocation,
        )

    def apply_action_to_skill_tree(
        self, action: Action, skill_tree: SkillTree | PartialSkillTree
    ) -> None:
        if action.action_type == "exploit":
            self.exploit_skill_tree(skill_tree, action)
        elif action.action_type == "explore":
            self.explore_skill_tree(skill_tree, action)
        else:
            raise ValueError(f"Invalid action type: {action.action_type}")

    def determine_performance_per_skill(
        self,
        skills: Sequence[Skill],
        completed_task_instances: Collection[CompletedTaskInstance],
    ) -> dict[Skill, float]:
        performance_per_skill: dict[Skill, float] = defaultdict(float)
        completed_task_instances_by_skill: dict[Skill, list[CompletedTaskInstance]] = (
            defaultdict(list)
        )
        for completed_task_instance in completed_task_instances:
            skill_category = Skill(
                self.skill_discovery_module.get_skill_category_name_for_task_instance(
                    completed_task_instance.task_instance
                )
            )
            completed_task_instances_by_skill[skill_category].append(
                completed_task_instance
            )

        for (
            skill,
            completed_task_instances,
        ) in completed_task_instances_by_skill.items():
            performance_per_skill[skill] = sum(
                completed_task_instance.was_correct
                for completed_task_instance in completed_task_instances
            ) / len(completed_task_instances)

        return performance_per_skill

    def determine_required_skills(
        self, completed_task_instances: Collection[CompletedTaskInstance]
    ) -> dict[Skill, list[CompletedTaskInstance]]:
        skills_to_completed_task_instances: dict[Skill, list[CompletedTaskInstance]] = (
            defaultdict(list)
        )
        for completed_task_instance in completed_task_instances:
            skill_category = Skill(
                self.skill_discovery_module.get_skill_category_name_for_task_instance(
                    completed_task_instance.task_instance
                )
            )
            skills_to_completed_task_instances[skill_category].append(
                completed_task_instance
            )

        return skills_to_completed_task_instances

    def choose_initial_explore_actions(
        self, required_skills: Sequence[Skill]
    ) -> dict[Skill, Action]:
        logger.info(
            "Choosing initial explore actions. required_skills={}, initial_explore_action={}",
            required_skills,
            self.initial_explore_action,
        )
        return {skill: self.initial_explore_action for skill in required_skills}

    def apply_actions_to_skill_forest(
        self,
        actions: dict[Skill, Action],
        skill_forest: Mapping[Skill, PartialSkillTree | SkillTree],
    ) -> None:
        for skill in skill_forest:
            self.apply_action_to_skill_tree(actions[skill], skill_forest[skill])

    def generate_data_for_partial_skill_forest(
        self,
        partial_skill_forest: Mapping[Skill, PartialSkillTree],
        predictor: PredictorInterface[TaskInstance],
    ) -> Mapping[Skill, SkillTree]:
        finalized_skill_forest: dict[Skill, SkillTree] = {}
        required_skills = list(partial_skill_forest.keys())
        for skill in required_skills:
            partial_skill_tree = partial_skill_forest[skill]
            skill_tree = self.generate_data_for_skill_tree(
                partial_skill_tree, predictor
            )
            finalized_skill_forest[skill] = skill_tree.finalize()

        return finalized_skill_forest

    def construct_partial_experience(
        self,
        skill_forest: Mapping[Skill, SkillTree],
        past_performance_per_skill: Mapping[Skill, float],
        actions: Mapping[Skill, Action],
    ) -> dict[Skill, PartialSkillExperience]:
        partial_experiences: dict[Skill, PartialSkillExperience] = {}
        for skill in skill_forest:
            partial_experience = PartialSkillExperience(
                state=SkillState(
                    skill_tree=skill_forest[skill],
                    past_performance=past_performance_per_skill[skill],
                ),
                skill=skill,
                action=actions[skill],
            )
            partial_experiences[skill] = partial_experience
        return partial_experiences

    def init_state(
        self,
        completed_task_instances: Collection[CompletedTaskInstance],
        predictor: PredictorInterface[TaskInstance],
    ) -> Sequence[TrainingDatumCovariant]:

        # Determine the skills that are required to be learned.
        required_skills = self.determine_required_skills(completed_task_instances)
        logger.info(
            "Determined required skills. required_skills={}", required_skills.keys()
        )

        # Determine performance per skill on the completed task instances.
        performance_per_skill = self.determine_performance_per_skill(
            list(required_skills.keys()), completed_task_instances
        )
        logger.info(
            "Determined performance per skill. performance_per_skill={}",
            performance_per_skill,
        )

        # Create an initial skill forest with no subskills and no training data.
        initial_skill_forest = self.create_initial_skill_forest(
            completed_task_instances
        )

        # Choose an explore action for each skill to expand the skill forest with new subskills.
        actions = self.choose_initial_explore_actions(list(required_skills.keys()))

        # Expand the skill forest with the subskills discovered by the explore action.
        self.apply_actions_to_skill_forest(actions, initial_skill_forest)

        # Generate data for each skill and subskill.
        finalized_skill_forest = self.generate_data_for_partial_skill_forest(
            initial_skill_forest, predictor
        )
        logger.info(
            "Generated data for skill forest. expected_training_data={}",
            self.calculate_expected_training_data_size(finalized_skill_forest),
        )

        # Construct a partial experience for each skill.
        partial_experiences = self.construct_partial_experience(
            skill_forest=finalized_skill_forest,
            past_performance_per_skill=performance_per_skill,
            actions=actions,
        )

        # Set the current experiences to the partial experiences.
        self.current_experiences = partial_experiences
        return self.pull_training_data_for_skill_forest(finalized_skill_forest)

    def update_state(
        self, completed_task_instances: Collection[CompletedTaskInstance]
    ) -> None:
        logger.info(
            "Updating state. num_completed_task_instances={}",
            len(completed_task_instances),
        )

        # Update state can only be called if we have started and have a current experience.
        assert self.current_experiences is not None

        # We start by calculating the performance per skill for the completed task instances.
        skills = list(self.current_experiences.keys())
        performance_per_skill = self.determine_performance_per_skill(
            skills, completed_task_instances
        )
        logger.info(
            "Determined performance per skill. performance_per_skill={}",
            performance_per_skill,
        )

        # Calculate the reward for each skill and update the current experience.
        for skill in skills:
            # The reward is the change in performance.
            performance_change = (
                performance_per_skill[skill]
                - self.current_experiences[skill].state.past_performance
            )
            # Now we add the reward to the partial experience.
            self.current_experiences[skill].reward = performance_change
        logger.info(
            "Calculated rewards. rewards={}",
            {skill: self.current_experiences[skill].reward for skill in skills},
        )

        # Now we finalize the partial experiences and add them to the past experiences.
        last_experience: dict[Skill, SkillExperience] = {}
        for skill in skills:
            last_experience[skill] = self.current_experiences[skill].finalize()

        self.past_experiences.append(last_experience)

    def determine_action_for_skill(self, history: Sequence[SkillExperience]) -> Action:
        return self.action_policy(history)

    def choose_next_actions(
        self,
        required_skills: Sequence[Skill],
        past_experiences: Sequence[Mapping[Skill, SkillExperience]],
    ) -> dict[Skill, Action]:
        next_actions: dict[Skill, Action] = {}
        for skill in required_skills:
            history_for_skill = [experience[skill] for experience in past_experiences]
            next_actions[skill] = self.determine_action_for_skill(history_for_skill)
        return next_actions

    def copy_partial_skill_forest_from_experience(
        self,
        experience: Mapping[Skill, SkillExperience],
    ) -> Mapping[Skill, PartialSkillTree]:
        skill_forest: dict[Skill, PartialSkillTree] = {}
        for skill in experience:
            skill_tree = experience[skill].state.skill_tree
            skill_forest[skill] = PartialSkillTree.copy_from_skill_tree(skill_tree)
        return skill_forest

    def construct_partial_experiences_given_past_experiences(
        self,
        past_experiences: Mapping[Skill, SkillExperience],
        current_skill_forest: Mapping[Skill, SkillTree],
        chosen_actions: Mapping[Skill, Action],
    ) -> dict[Skill, PartialSkillExperience]:
        partial_experiences: dict[Skill, PartialSkillExperience] = {}
        for skill, past_skill_experience in past_experiences.items():
            current_skill_tree = current_skill_forest[skill]
            past_performance = past_skill_experience.get_performance()
            partial_experience = PartialSkillExperience(
                state=SkillState(
                    skill_tree=current_skill_tree,
                    past_performance=past_performance,
                ),
                skill=skill,
                action=chosen_actions[skill],
            )
            partial_experiences[skill] = partial_experience
        return partial_experiences

    def transition_to_next_state(
        self,
        predictor: PredictorInterface[TaskInstance],
    ) -> Sequence[TrainingDatumCovariant]:
        assert len(self.past_experiences) > 0

        last_experience = self.past_experiences[-1]
        required_skills = list(last_experience.keys())
        logger.info("Transitioning to next state. required_skills={}", required_skills)

        # Determine the action for each skill.
        actions = self.choose_next_actions(required_skills, self.past_experiences)

        # Copy the last experience to a new partial skill forest.
        new_partial_skill_forest = self.copy_partial_skill_forest_from_experience(
            last_experience
        )

        # Apply the actions to the new partial skill forest.
        self.apply_actions_to_skill_forest(actions, new_partial_skill_forest)

        # Now we generate data for each skill and subskill.
        finalized_skill_forest = self.generate_data_for_partial_skill_forest(
            new_partial_skill_forest, predictor
        )

        # Construct a partial experience for each skill.
        partial_experiences = self.construct_partial_experiences_given_past_experiences(
            past_experiences=last_experience,
            current_skill_forest=finalized_skill_forest,
            chosen_actions=actions,
        )

        # Set the current experiences to the partial experiences.
        self.current_experiences = partial_experiences

        return self.pull_training_data_for_skill_forest(finalized_skill_forest)

    def calculate_expected_training_data_size(
        self, skill_forest: Mapping[Skill, SkillTree]
    ) -> int:
        expected_training_data_size = 0
        for skill_tree in skill_forest.values():
            for subskill in skill_tree.subskills:
                expected_training_data_size += skill_tree.data_allocation[subskill]
        return expected_training_data_size

    def __call__(
        self,
        completed_task_instances: Collection[CompletedTaskInstance],
        predictor: PredictorInterface[TaskInstance],
    ) -> Sequence[TrainingDatumCovariant]:
        """
        This is the entry point for generating training data.

        The expected usage is to call this method once, gather the training data, and then
        train a model on the training data, evaluate it, and then call this method again
        with the evaluation results (completed task instances) to generate more training data.
        """
        if self.current_experiences is None:
            training_data = self.init_state(completed_task_instances, predictor)
        else:
            self.update_state(completed_task_instances=completed_task_instances)
            training_data = self.transition_to_next_state(predictor)

        self.save_checkpoint()
        return training_data

    def step(self) -> None:
        self.generation_index += 1


VqaDataGenerationAgent = BaseDataGenerationAgent[
    OpenEndedVqaTaskInstance,
    VqaTrainingDatum,
    VqaTrainingDataQualityCheck,
    CompletedVqaTaskInstance,
]

MathDataGenerationAgent = BaseDataGenerationAgent[
    MathTaskInstance,
    MathTrainingDatum,
    MathTrainingDatumQualityCheck,
    CompletedMathTaskInstance,
]

CodeGenerationDataGenerationAgent = BaseDataGenerationAgent[
    CodeGenerationTaskInstance,
    CodeGenerationTrainingDatum,
    CodeGenerationTrainingDataQualityCheck,
    CodeGenerationCompletedTaskInstance,
]
