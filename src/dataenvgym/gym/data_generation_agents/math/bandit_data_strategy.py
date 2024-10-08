import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import (
    Collection,
    Iterator,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
)

import numpy as np
from loguru import logger
from pydantic import BaseModel
from ulid import ULID

from dataenvgym.gym.data_generation_agents.math.baselines.skill_list import (
    MathTrainingDatumWithSkillCategory,
)
from dataenvgym.gym.data_generation_agents.skill_tree import (
    Action,
    Exploit,
    Explore,
    PartialSkillExperience,
    PartialSkillTree,
    Skill,
    SkillExperience,
    SkillState,
    SkillTree,
    Subskill,
)
from dataenvgym.gym.domain_models import (
    CompletedMathTaskInstance,
    MathDataGenerationAgent,
    MathPredictorInterface,
    MathSkillDiscoveryInterface,
    MathTrainingDataQualityCheckerInterface,
    MathTrainingDatum,
    MathTrainingDatumQualityCheck,
    implements,
)


TrainingDataCache = MutableMapping[ULID, MathTrainingDatumWithSkillCategory]


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


class SubskillDataGeneratorInterface(Protocol):
    def __call__(
        self,
        subskill: Subskill,
        data_budget: int,
    ) -> Sequence[MathTrainingDatumWithSkillCategory]: ...


class StubSubskillDataGenerator:
    def __call__(
        self,
        subskill: Subskill,
        data_budget: int,
    ) -> Sequence[MathTrainingDatumWithSkillCategory]:
        training_data = []
        for i in range(data_budget):
            training_datum = MathTrainingDatumWithSkillCategory(
                ulid=ULID(),
                instruction=f"This is the {i+1}th training datum for subskill: {subskill}. Solve the following problem.",
                response=f"This is the response for the {i+1}th training datum of subskill: {subskill}. The solution is...",
                skill_category=str(subskill),
            )
            training_data.append(training_datum)
        return training_data


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


class JsonExperienceCheckpointer:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.current_experiences_path = output_path / "current_experiences.jsonl"
        self.past_experiences_path = output_path / "past_experiences.jsonl"

    @staticmethod
    def save_experience_as_json(
        experiences: Sequence[Mapping[Skill, SkillExperience | PartialSkillExperience]],
        save_path: Path,
    ) -> None:
        blob: list[dict[str, dict]] = []
        for experience in experiences:
            experience_blob: dict[str, dict] = {}
            for skill, skill_experience in experience.items():
                experience_blob[str(skill)] = json.loads(
                    skill_experience.model_dump_json()
                )
            blob.append(experience_blob)

        with tempfile.NamedTemporaryFile("w", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(blob, temp_file)

        try:
            # We have to use shutil because temp_path is on a different filesystem
            # than the current one.
            shutil.move(temp_path, save_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def load_past_experiences(self) -> Sequence[Mapping[Skill, SkillExperience]]:
        try:
            blob = json.loads(self.past_experiences_path.read_text())
        except FileNotFoundError:
            return []

        experiences: list[Mapping[Skill, SkillExperience]] = []
        for experience_blob in blob:
            experience: dict[Skill, SkillExperience] = {}
            for skill, raw_experience in experience_blob.items():
                experience[Skill(skill)] = SkillExperience.model_validate(
                    raw_experience
                )
            experiences.append(experience)
        return experiences

    def load_current_experiences(
        self,
    ) -> Optional[Mapping[Skill, PartialSkillExperience]]:
        try:
            blob = json.loads(self.current_experiences_path.read_text())
        except FileNotFoundError:
            return None

        if not blob:
            return None

        # Choose the most recent line to load as the current experiences,
        # in the case there are multiple.
        experience_blob = blob[-1]
        experience: dict[Skill, PartialSkillExperience] = {}
        for skill, raw_experience in experience_blob.items():
            experience[Skill(skill)] = PartialSkillExperience.model_validate(
                raw_experience
            )
        return experience

    def save_checkpoint(
        self,
        past_experiences: Sequence[Mapping[Skill, SkillExperience]],
        current_experiences: Optional[Mapping[Skill, PartialSkillExperience]] = None,
    ) -> None:
        # Make the parent directory if it doesn't exist.
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.save_experience_as_json(past_experiences, self.past_experiences_path)
        if current_experiences:
            self.save_experience_as_json(
                [current_experiences], self.current_experiences_path
            )

    def load_checkpoint(
        self,
    ) -> tuple[
        Sequence[Mapping[Skill, SkillExperience]],
        Optional[Mapping[Skill, PartialSkillExperience]],
    ]:
        past_experiences = self.load_past_experiences()
        current_experiences = self.load_current_experiences()
        return past_experiences, current_experiences

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(output_path={self.output_path})"


class StubMathTrainingDataQualityChecker:
    def check_training_datum(
        self, training_datum: MathTrainingDatum
    ) -> MathTrainingDatumQualityCheck:
        return MathTrainingDatumQualityCheck(
            ulid=ULID(),
            training_datum_ulid=training_datum.ulid,
            annotated_answer_passes_scoring_code=True,
            student_already_knows_answer="not applicable: was not checked",
            qa_passed=True,
        )

    def __call__(
        self,
        training_data: Sequence[MathTrainingDatum],
        predictor: MathPredictorInterface,
    ) -> Sequence[MathTrainingDatumQualityCheck]:
        return [
            self.check_training_datum(training_datum)
            for training_datum in training_data
        ]


implements(MathTrainingDataQualityCheckerInterface)(StubMathTrainingDataQualityChecker)


class MathBanditDataStrategy:
    def __init__(
        self,
        skill_discovery_module: MathSkillDiscoveryInterface,
        logging_folder: Optional[Path] = None,
        subskill_data_generator: SubskillDataGeneratorInterface = StubSubskillDataGenerator(),
        training_data_cache: MutableMapping[
            ULID, MathTrainingDatumWithSkillCategory
        ] = InMemoryKeyValueCache(),
        action_policy: ActionPolicy = StubActionPolicy(
            Explore(num_new_skills=1, data_allocation_for_new_skills=1)
        ),
        subskill_proposal_policy: SubskillProposalPolicy = StubSubskillProposalPolicy(),
        initial_explore_action: Explore = Explore(
            num_new_skills=10, data_allocation_for_new_skills=10
        ),
        experience_checkpointer: Optional[ExperienceCheckpointerProtocol] = None,
        training_data_quality_checker: MathTrainingDataQualityCheckerInterface = StubMathTrainingDataQualityChecker(),
        quality_check_cache: MutableMapping[
            ULID, MathTrainingDatumQualityCheck
        ] = InMemoryKeyValueCache(),
    ):
        self.skill_discovery_module = skill_discovery_module
        self.generation_index = 0
        self.current_experiences: Optional[dict[Skill, PartialSkillExperience]] = None
        self.past_experiences: list[dict[Skill, SkillExperience]] = []
        self.subskill_data_generator = subskill_data_generator
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
        self, completed_task_instances: Collection[CompletedMathTaskInstance]
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

    def get_training_data_by_ulid(
        self, ulid: ULID
    ) -> MathTrainingDatumWithSkillCategory:
        return self.training_data_cache[ulid]

    def get_quality_check_by_ulid(self, ulid: ULID) -> MathTrainingDatumQualityCheck:
        return self.quality_check_cache[ulid]

    def pull_training_data_for_skill_forest(
        self, skill_forest: Mapping[Skill, SkillTree]
    ) -> Sequence[MathTrainingDatumWithSkillCategory]:
        logger.info(
            "Pulling training data for skill forest. expected_training_data={}",
            sum(len(skill_tree.training_data) for skill_tree in skill_forest.values()),
        )
        training_data: list[MathTrainingDatumWithSkillCategory] = []
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
    ) -> Sequence[MathTrainingDatumWithSkillCategory]:
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
        self, skill_tree: PartialSkillTree, predictor: MathPredictorInterface
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
        completed_task_instances: Collection[CompletedMathTaskInstance],
    ) -> dict[Skill, float]:
        performance_per_skill: dict[Skill, float] = defaultdict(float)
        completed_task_instances_by_skill: dict[
            Skill, list[CompletedMathTaskInstance]
        ] = defaultdict(list)
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
        self, completed_task_instances: Collection[CompletedMathTaskInstance]
    ) -> dict[Skill, list[CompletedMathTaskInstance]]:
        skills_to_completed_task_instances: dict[
            Skill, list[CompletedMathTaskInstance]
        ] = defaultdict(list)
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
        predictor: MathPredictorInterface,
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
        completed_task_instances: Collection[CompletedMathTaskInstance],
        predictor: MathPredictorInterface,
    ) -> Sequence[MathTrainingDatumWithSkillCategory]:

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
        self, completed_task_instances: Collection[CompletedMathTaskInstance]
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
        predictor: MathPredictorInterface,
    ) -> Sequence[MathTrainingDatumWithSkillCategory]:
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
        completed_task_instances: Collection[CompletedMathTaskInstance],
        predictor: MathPredictorInterface,
    ) -> Sequence[MathTrainingDatum]:
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


implements(MathDataGenerationAgent)(MathBanditDataStrategy)
