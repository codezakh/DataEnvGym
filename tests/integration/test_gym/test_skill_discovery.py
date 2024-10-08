import pytest
from dataenvgym.gym.skill_discovery.llm_metacognitive import (
    VqaSkillDiscoverer,
    MathSkillDiscoverer,
)
from dataenvgym.gym.domain_models import OpenEndedVqaTaskInstance, MathTaskInstance
from PIL import Image
from pathlib import Path


@pytest.fixture(scope="module")
def vqa_discovery_setup() -> tuple[
    VqaSkillDiscoverer,
    list[OpenEndedVqaTaskInstance],
    list[OpenEndedVqaTaskInstance],
]:
    example_image = Image.new("RGB", (100, 100))

    requires_color_identification = [
        OpenEndedVqaTaskInstance(
            task_name="gqa",
            instance_id="123",
            instruction="What is the color of the object?",
            ground_truth_label="blue",
            image=example_image,
        ),
        OpenEndedVqaTaskInstance(
            task_name="gqa",
            instance_id="124",
            instruction="What is the color of the object?",
            ground_truth_label="green",
            image=example_image,
        ),
        OpenEndedVqaTaskInstance(
            task_name="gqa",
            instance_id="125",
            instruction="What is the color of the object?",
            ground_truth_label="red",
            image=example_image,
        ),
    ]
    requires_shape_identification = [
        OpenEndedVqaTaskInstance(
            task_name="gqa",
            instance_id="126",
            instruction="What shape is the object?",
            ground_truth_label="pentagon",
            image=example_image,
        ),
        OpenEndedVqaTaskInstance(
            task_name="gqa",
            instance_id="127",
            instruction="What shape is the object?",
            ground_truth_label="rhombus",
            image=example_image,
        ),
        OpenEndedVqaTaskInstance(
            task_name="gqa",
            instance_id="128",
            instruction="What shape is the object?",
            ground_truth_label="ellipse",
            image=example_image,
        ),
    ]

    discoverer = VqaSkillDiscoverer()
    discoverer.discover_skills(
        requires_color_identification + requires_shape_identification
    )

    return discoverer, requires_color_identification, requires_shape_identification


def test_skills_discovered(vqa_discovery_setup):
    discoverer, _, _ = vqa_discovery_setup
    assert discoverer.discovery_result is not None
    assert len(discoverer.discovery_result.skill_categories) >= 2


def test_skill_category_for_color_identification(vqa_discovery_setup):
    discoverer, requires_color_identification, _ = vqa_discovery_setup
    category_for_color_identification = (
        discoverer.get_skill_category_name_for_task_instance(
            requires_color_identification[0]
        )
    )
    assert category_for_color_identification is not None


def test_skill_category_for_shape_identification(vqa_discovery_setup):
    discoverer, _, requires_shape_identification = vqa_discovery_setup
    category_for_shape_identification = (
        discoverer.get_skill_category_name_for_task_instance(
            requires_shape_identification[0]
        )
    )
    assert category_for_shape_identification is not None


def test_new_shape_identification_task(vqa_discovery_setup):
    discoverer, _, requires_shape_identification = vqa_discovery_setup
    example_image = Image.new("RGB", (100, 100))
    new_shape_identification_task = OpenEndedVqaTaskInstance(
        task_name="gqa",
        instance_id="129",
        instruction="What shape is the object?",
        ground_truth_label="circle",
        image=example_image,
    )

    category_for_shape_identification = (
        discoverer.get_skill_category_name_for_task_instance(
            requires_shape_identification[0]
        )
    )

    assert (
        discoverer.get_skill_category_name_for_task_instance(
            new_shape_identification_task
        )
        == category_for_shape_identification
    )


def test_loading_and_saving_results(tmp_path: Path, vqa_discovery_setup) -> None:
    discoverer, _, _ = vqa_discovery_setup
    save_path = tmp_path / "discovery_result.json"
    discoverer.save_precomputed_skills(save_path)

    new_discoverer = VqaSkillDiscoverer()
    assert new_discoverer.discovery_result is None
    new_discoverer.set_precomputed_skills(save_path)

    assert new_discoverer.discovery_result is not None

    example_image = Image.new("RGB", (100, 100))
    new_shape_identification_task = OpenEndedVqaTaskInstance(
        task_name="gqa",
        instance_id="129",
        instruction="What shape is the object?",
        ground_truth_label="circle",
        image=example_image,
    )

    # Check that the new and old discoverer give the same category for the same task.
    assert discoverer.get_skill_category_name_for_task_instance(
        new_shape_identification_task
    ) == new_discoverer.get_skill_category_name_for_task_instance(
        new_shape_identification_task
    )


class TestMathSkillDiscoverer:
    @pytest.fixture(scope="module")
    def math_discovery_setup(self) -> tuple[
        MathSkillDiscoverer,
        list[MathTaskInstance],
        list[MathTaskInstance],
    ]:
        requires_number_theory_skill = [
            MathTaskInstance(
                task_name="math",
                instance_id="201",
                instruction="Find the greatest common divisor of 24 and 36.",
                ground_truth_label=(
                    "We can find the GCD by finding the prime factorizations of the numbers. "
                    "24 = 2^3 * 3 and 36 = 2^2 * 3^2. The GCD is the product of the lowest powers "
                    "of all primes that appear in both factorizations, which gives us 2^2 * 3 = 12.\n"
                    "Final Answer: The final answer is \\boxed{12}."
                ),
            ),
            MathTaskInstance(
                task_name="math",
                instance_id="202",
                instruction="What is the least common multiple of 8 and 12?",
                ground_truth_label=(
                    "The LCM can be found by taking the highest powers of all primes in the factorizations. "
                    "8 = 2^3 and 12 = 2^2 * 3. The LCM is 2^3 * 3 = 24.\n"
                    "Final Answer: The final answer is \\boxed{24}."
                ),
            ),
        ]
        requires_probability_skill = [
            MathTaskInstance(
                task_name="math",
                instance_id="203",
                instruction="What is the probability of rolling a sum of 7 on two six-sided dice?",
                ground_truth_label=(
                    "The possible outcomes for two dice are (1,6), (2,5), (3,4), (4,3), (5,2), and (6,1). "
                    "There are 6 favorable outcomes and 36 total outcomes, so the probability is 6/36 = 1/6.\n"
                    "Final Answer: The final answer is \\boxed{\\frac{1}{6}}."
                ),
            ),
            MathTaskInstance(
                task_name="math",
                instance_id="204",
                instruction="A box contains 3 red balls and 2 blue balls. If two balls are drawn at random without replacement, "
                "what is the probability that both are red?",
                ground_truth_label=(
                    "The probability of drawing the first red ball is 3/5, and the probability of drawing a second red ball "
                    "after drawing the first is 2/4. Therefore, the probability of drawing two red balls is (3/5) * (2/4) = 6/20 = 3/10.\n"
                    "Final Answer: The final answer is \\boxed{\\frac{3}{10}}."
                ),
            ),
        ]

        discoverer = MathSkillDiscoverer()
        discoverer.discover_skills(
            requires_number_theory_skill + requires_probability_skill
        )

        return discoverer, requires_number_theory_skill, requires_probability_skill

    def test_skills_discovered(self, math_discovery_setup):
        discoverer, _, _ = math_discovery_setup
        assert discoverer.discovery_result is not None
        assert len(discoverer.discovery_result.skill_categories) >= 2

    def test_skill_category_for_number_theory(self, math_discovery_setup):
        discoverer, requires_number_theory_skill, _ = math_discovery_setup
        category_for_number_theory_skill = (
            discoverer.get_skill_category_name_for_task_instance(
                requires_number_theory_skill[0]
            )
        )
        assert category_for_number_theory_skill is not None

    def test_skill_category_for_probability(self, math_discovery_setup):
        discoverer, _, requires_probability_skill = math_discovery_setup
        category_for_probability_skill = (
            discoverer.get_skill_category_name_for_task_instance(
                requires_probability_skill[0]
            )
        )
        assert category_for_probability_skill is not None

    def test_new_probability_task(self, math_discovery_setup):
        discoverer, _, requires_probability_skill = math_discovery_setup
        new_probability_task = MathTaskInstance(
            task_name="math",
            instance_id="205",
            instruction="What is the probability of drawing a king from a standard deck of cards?",
            ground_truth_label=(
                "There are 4 kings in a standard deck of 52 cards. The probability of drawing a king is 4/52 = 1/13.\n"
                "Final Answer: The final answer is \\boxed{\\frac{1}{13}}."
            ),
        )

        category_for_probability_skill = (
            discoverer.get_skill_category_name_for_task_instance(
                requires_probability_skill[0]
            )
        )

        assert (
            discoverer.get_skill_category_name_for_task_instance(new_probability_task)
            == category_for_probability_skill
        )

    def test_loading_and_saving_results(
        self, tmp_path: Path, math_discovery_setup
    ) -> None:
        discoverer, _, _ = math_discovery_setup
        save_path = tmp_path / "discovery_result.json"
        discoverer.save_precomputed_skills(save_path)

        new_discoverer = MathSkillDiscoverer()
        assert new_discoverer.discovery_result is None
        new_discoverer.set_precomputed_skills(save_path)

        assert new_discoverer.discovery_result is not None

        new_probability_task = MathTaskInstance(
            task_name="math",
            instance_id="205",
            instruction="What is the probability of drawing a king from a standard deck of cards?",
            ground_truth_label=(
                "There are  4 kings in a standard deck of 52 cards. The probability of drawing a king is 4/52 = 1/13.\n"
                "Final Answer: The final answer is \\boxed{\\frac{1}{13}}."
            ),
        )

        assert discoverer.get_skill_category_name_for_task_instance(
            new_probability_task
        ) == new_discoverer.get_skill_category_name_for_task_instance(
            new_probability_task
        )
