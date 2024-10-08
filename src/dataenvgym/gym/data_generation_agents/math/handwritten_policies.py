import random
from typing import Mapping, Optional, Sequence

from loguru import logger

from dataenvgym.gym.data_generation_agents.math.bandit_data_strategy import (
    Action,
    ActionPolicy,
    Exploit,
    Explore,
    SkillExperience,
    SkillTree,
    Subskill,
)
from dataenvgym.gym.data_generation_agents.math.bandit_utilities import (
    sum_data_generated_over_tree_history,
)
from dataenvgym.gym.domain_models import implements


class AlternatingExploreExploitPolicy:
    """
    A policy that alternates between Explore and Exploit actions.

    Parameters
    ----------
    explore_action : Explore
        The Explore action to be used.
    """

    def __init__(
        self, explore_action: Explore, max_data_for_skill: Optional[int] = None
    ):
        self.explore_action = explore_action
        self.max_data_for_skill = max_data_for_skill

    def __call__(self, history: Sequence[SkillExperience]) -> Action:
        """
        Determine the next action based on the history of actions.

        Parameters
        ----------
        history : Sequence[SkillExperience]
            The history of skill experiences.

        Returns
        -------
        Action
            The next action to be taken.
        """

        if not history:
            # If there's no history, start with the initial explore action.
            self.last_action = self.explore_action
            return self.explore_action

        last_experience = history[-1]
        last_action = last_experience.action

        if (
            self.max_data_for_skill is not None
            and (
                total_data_generated := sum(
                    _.state.skill_tree.total_data_allocated for _ in history
                )
            )
            >= self.max_data_for_skill
        ):
            skill_name = last_experience.state.skill_tree.skill
            log_message = f"Skill {skill_name} exceeds the configured limit, no-oping. limit={self.max_data_for_skill}, available_data={total_data_generated}"
            logger.info(log_message)
            # Emit an exploit action that sets all allocations to 0.
            exploit_action = Exploit(
                data_allocation_delta={
                    subskill: -allocation
                    for subskill, allocation in last_experience.state.skill_tree.data_allocation.items()
                }
            )
            self.last_action = exploit_action
            return exploit_action

        if last_action.action_type == "explore":
            # If the last action was explore, create an exploit action to zero out data allocations.
            data_allocation_delta = {
                subskill: -allocation
                for subskill, allocation in last_experience.state.skill_tree.data_allocation.items()
            }
            exploit_action = Exploit(data_allocation_delta=data_allocation_delta)
            return exploit_action

        elif last_action.action_type == "exploit":
            # If the last action was exploit, reuse the explore action.
            return self.explore_action

        else:
            raise ValueError(f"Unexpected action type: {last_action.action_type}")


implements(ActionPolicy)(AlternatingExploreExploitPolicy)


class FillBalancedSkillTreePolicy:
    """
    Grows a skill tree to a fixed number of skills, then fills each skill to a fixed amount of data.

    Starts by behaving like AlternatingExploreExploitPolicy, but once the skill tree has
    reached the maximum number of subskills, it switches to Exploit actions to fill each skill to a fixed amount of data.

    Parameters
    ----------
    explore_action : Explore
        The Explore action to be used.
    max_subskills : int
        The maximum number of subskills allowed.
    max_data_per_subskill : int
        The maximum amount of data per subskill.
    max_allocation_per_subskill : int
        The maximum allocation of data allowed per subskill.
    """

    def __init__(
        self,
        explore_action: Explore,
        max_subskills: int,
        max_data_per_subskill: int,
        max_allocation_per_subskill: int,
    ):
        self.explore_action = explore_action
        self.max_subskills = max_subskills
        self.max_data_per_subskill = max_data_per_subskill
        self.max_allocation_per_subskill = max_allocation_per_subskill

    def __call__(self, history: Sequence[SkillExperience]) -> Action:
        """
        Determine the next action based on the history of actions.

        Parameters
        ----------
        history : Sequence[SkillExperience]
            The history of skill experiences.

        Returns
        -------
        Action
            The next action to be taken.
        """
        if not history:
            # If there's no history, start with the initial explore action.
            return self.explore_action

        last_experience = history[-1]
        last_action = last_experience.action
        skill_tree = last_experience.state.skill_tree

        if len(skill_tree.subskills) < self.max_subskills:
            logger.info(
                f"Number of subskills: {len(skill_tree.subskills)} < {self.max_subskills}, alternating between explore and exploit."
            )
            # If the number of subskills is less than max_subskills, alternate between explore and exploit.
            return self._alternating_explore_exploit(
                last_action, skill_tree, self.explore_action
            )
        else:
            # If max_subskills is exceeded, calculate and return the next exploit action.
            logger.info(
                f"Number of subskills: {len(skill_tree.subskills)} >= {self.max_subskills}, filling each skill to {self.max_data_per_subskill} data."
            )
            data_already_generated = sum_data_generated_over_tree_history(history)
            return self._calculate_next_exploit_action(
                self.max_data_per_subskill,
                self.max_allocation_per_subskill,
                data_already_generated,
                skill_tree.data_allocation,
            )

    @staticmethod
    def _alternating_explore_exploit(
        last_action: Action, skill_tree: SkillTree, explore_action: Explore
    ) -> Action:
        if last_action.action_type == "explore":
            data_allocation_delta = {
                subskill: -allocation
                for subskill, allocation in skill_tree.data_allocation.items()
            }
            exploit_action = Exploit(data_allocation_delta=data_allocation_delta)
            return exploit_action
        elif last_action.action_type == "exploit":
            return explore_action
        else:
            raise ValueError(f"Unexpected action type: {last_action.action_type}")

    @staticmethod
    def _calculate_next_exploit_action(
        max_data_per_subskill: int,
        max_allocation_per_subskill: int,
        data_already_generated: Mapping[Subskill, int],
        current_data_allocation: Mapping[Subskill, int],
    ) -> Exploit:
        """
        Calculate the next exploit action to reach max_data_per_subskill for each subskill.

        Parameters
        ----------
        data_already_generated : Mapping[Subskill, int]
            The amount of data already generated for each subskill.

        Returns
        -------
        Exploit
            The next exploit action to be taken.
        """
        logger.info(f"data_already_generated: {data_already_generated}")
        data_allocation_delta = {}
        for subskill, subskill_data_count in data_already_generated.items():
            data_needed = max_data_per_subskill - subskill_data_count
            if data_needed > 0:
                forecasted_subskill_data_count = (
                    subskill_data_count + current_data_allocation[subskill]
                )
                deficit = max_data_per_subskill - forecasted_subskill_data_count
                maximum_delta = (
                    max_allocation_per_subskill - current_data_allocation[subskill]
                )
                data_allocation_delta[subskill] = min(deficit, maximum_delta)
            else:
                # We don't need to generate any more data for this subskill.
                # Zero out the allocation for this subskill.
                data_allocation_delta[subskill] = -current_data_allocation[subskill]

        if all(delta == 0 for delta in data_allocation_delta.values()):
            logger.info(
                f"All subskills have reached {max_data_per_subskill} data, emitting no-op exploit action."
            )
        return Exploit(data_allocation_delta=data_allocation_delta)


implements(ActionPolicy)(FillBalancedSkillTreePolicy)


class RandomExploreExploitPolicy:
    """
    A policy that randomly chooses between Explore and Exploit actions with equal probability,
    while respecting a configured limit on data allocation.

    Parameters
    ----------
    max_data_per_subskill : int
        The maximum amount of data allowed per subskill.
    max_new_skills : int
        The maximum number of new skills to add during an Explore action.
    max_data_for_new_skills : int
        The maximum amount of data to allocate for new skills during an Explore action.
    """

    def __init__(
        self,
        max_data_per_subskill: int,
        max_new_skills: int = 3,
        max_data_for_new_skills: int = 10,
        max_active_subskills: int = 10,
    ):
        self.max_data_per_subskill = max_data_per_subskill
        self.max_new_skills = max_new_skills
        self.max_data_for_new_skills = max_data_for_new_skills
        self.max_active_subskills = max_active_subskills

    def __call__(self, history: Sequence[SkillExperience]) -> Action:
        """
        Determine the next action based on the history of actions.

        Parameters
        ----------
        history : Sequence[SkillExperience]
            The history of skill experiences.

        Returns
        -------
        Action
            The next action to be taken.
        """
        if not history:
            return self._generate_explore_action()

        last_experience = history[-1]
        skill_tree = last_experience.state.skill_tree

        if random.choice([True, False]):
            return self._generate_explore_action()
        else:
            return self._generate_exploit_action(skill_tree)

    def _generate_explore_action(self) -> Explore:
        num_new_skills = random.randint(1, self.max_new_skills)
        data_allocation = random.randint(1, self.max_data_for_new_skills)
        return Explore(
            num_new_skills=num_new_skills,
            data_allocation_for_new_skills=data_allocation,
        )

    def _generate_exploit_action(self, skill_tree: SkillTree) -> Exploit:
        data_allocation_delta = {}
        active_subskills = [
            s for s in skill_tree.subskills if skill_tree.data_allocation.get(s, 0) > 0
        ]
        inactive_subskills = [
            s for s in skill_tree.subskills if skill_tree.data_allocation.get(s, 0) == 0
        ]

        # Deactivate subskills if we're over the limit
        if len(active_subskills) > self.max_active_subskills:
            subskills_to_deactivate = random.sample(
                active_subskills, len(active_subskills) - self.max_active_subskills
            )
            for subskill in subskills_to_deactivate:
                data_allocation_delta[subskill] = -skill_tree.data_allocation[subskill]
            active_subskills = [
                s for s in active_subskills if s not in subskills_to_deactivate
            ]
            inactive_subskills.extend(subskills_to_deactivate)

        # Activate subskills if we're under the limit
        if len(active_subskills) < self.max_active_subskills and inactive_subskills:
            num_to_activate = min(
                self.max_active_subskills - len(active_subskills),
                len(inactive_subskills),
            )
            subskills_to_activate = random.sample(inactive_subskills, num_to_activate)
            active_subskills.extend(subskills_to_activate)

        # Adjust allocations for active subskills
        for subskill in active_subskills:
            current_allocation = skill_tree.data_allocation.get(subskill, 0)
            max_increase = self.max_data_per_subskill - current_allocation
            max_decrease = -current_allocation
            delta = random.randint(max_decrease, max_increase)
            data_allocation_delta[subskill] = delta

        # zero out the allocations for all other subskills
        for subskill in skill_tree.subskills:
            if subskill not in data_allocation_delta:
                data_allocation_delta[subskill] = -skill_tree.data_allocation.get(
                    subskill, 0
                )

        return Exploit(data_allocation_delta=data_allocation_delta)


implements(ActionPolicy)(RandomExploreExploitPolicy)
