from dataenvgym.gym.tasks.math.MATH.task import MATHTask, MATHRecordWithIDAndAnswer
from dataenvgym.gym.domain_models import (
    MathPredictorInterface,
    MathTaskInstance,
    implements,
)
from typing import Collection
from dataenvgym.gym.tasks.math.MATH.scoring import (
    render_solution_for_scoring,
    score_candidate_answer,
    get_unnormalized_answer,
    normalize_final_answer,
)
from dataenvgym.gym.tasks.math.MATH.task import load_math_dataset, load_split


class AlwaysRightMathPredictor:
    def __init__(
        self,
        math_id_to_math_record: dict[str, MATHRecordWithIDAndAnswer],
        # answer_prefix: str = EXPECTED_ANSWER_PREFIX,
    ):
        # self.answer_prefix = answer_prefix
        # self.template = jinja2.Template(
        #     r"{{ answer_prefix }} $\boxed{ {{ answer }} }$. I hope it is correct."
        # )
        self.math_id_to_math_record = math_id_to_math_record

    def answer_task_instance(self, task_instance: MathTaskInstance) -> str:
        # Get the processed answer _only_ rather than the full solution.
        # We want to make sure that the scoring code is looking _only_ at
        # the final answer, not the full solution.
        answer_for_instance = self.math_id_to_math_record[task_instance.instance_id][
            "answer"
        ]
        chain_of_thought = self.math_id_to_math_record[task_instance.instance_id][
            "solution"
        ]
        return render_solution_for_scoring(
            chain_of_thought=chain_of_thought, final_answer=answer_for_instance
        )
        # return self.template.render(
        #     answer_prefix=self.answer_prefix,
        #     answer=answer_for_instance,
        #     trim_blocks=True,
        #     lstrip_blocks=True,
        # )

    def predict(self, task_instances: Collection[MathTaskInstance]) -> list[str]:
        return [
            self.answer_task_instance(task_instance) for task_instance in task_instances
        ]


class AlwaysWrongMathPredictor:
    def predict(self, task_instances: Collection[MathTaskInstance]) -> list[str]:
        return ["wrong"] * len(task_instances)


implements(MathPredictorInterface)(AlwaysRightMathPredictor)


def test_always_predicts_right_answer():
    math_task = MATHTask("val_balanced_subset_10")
    predictor = AlwaysRightMathPredictor(math_task.math_id_to_math_record)
    completed_task_instances = math_task.evaluate(predictor)
    performance_report = math_task.generate_performance_report(completed_task_instances)
    print("Overall performance:", performance_report.overall_performance)
    assert performance_report.overall_performance >= 0.99


def test_always_predicts_wrong_answer():
    math_task = MATHTask("val_balanced_subset_10")
    predictor = AlwaysWrongMathPredictor()
    completed_task_instances = math_task.evaluate(predictor)
    performance_report = math_task.generate_performance_report(completed_task_instances)
    print("Overall performance:", performance_report.overall_performance)
    assert performance_report.overall_performance == 0.0


def test_solution_template_is_correct():
    math_records = load_split("debug", load_math_dataset())
    record = math_records[0]
    response = render_solution_for_scoring(
        chain_of_thought=record["solution"], final_answer=record["answer"]
    )

    # Get some debugging information...
    extracted_answer = get_unnormalized_answer(response)
    print(f"Extracted answer: {extracted_answer}")
    normalized_answer = normalize_final_answer(extracted_answer)
    print(f"Normalized answer: {normalized_answer}")

    score = score_candidate_answer(
        ground_truth_answer=record["answer"], candidate=response
    )

    assert score == 1


def test_performance_report_stratified_by_type():
    math_task = MATHTask("val_balanced_subset_10")
    predictor = AlwaysRightMathPredictor(math_task.math_id_to_math_record)
    completed_task_instances = math_task.evaluate(predictor)
    performance_report = math_task.generate_performance_report(completed_task_instances)

    print("Overall performance:", performance_report.overall_performance)
    assert performance_report.overall_performance >= 0.99

    # Check that the performance report includes slices for each question type
    question_types = {
        record["type"] for record in math_task.math_id_to_math_record.values()
    }
    slice_names = {slice.slice_name for slice in performance_report.slices}

    assert question_types == slice_names

    for slice in performance_report.slices:
        print(f"Performance for type {slice.slice_name}: {slice.metric_value}")
        assert slice.metric_value >= 0.99
        assert slice.count > 0


def test_scoring_problem_case():
    problem_case = {
        "ulid": "01J8E5KD9X4PEE04ACFGC9NX10",
        "task_instance": {
            "task_name": "MATH",
            "instance_id": "MATH_test_4157",
            "instruction": "Two-thirds of the students at Baker Middle School take music. There are 834 students who take music. How many students are there at Baker Middle School?",
            "ground_truth_label": "If there are $S$ students at Baker Middle School, then $\\frac{2}{3}S$ students take music.  Setting $\\frac{2}{3}S$ equal to 834 and multiplying both sides by $\\frac{3}{2}$, we find that there are $\\frac{3}{2}\\times 834=\\boxed{1251}$ students at the school.\nFinal Answer: The final answer is $\\boxed{ 1251 }$.",
        },
        "predictor_response": "<|start_header_id|>assistant<|end_header_id|>\n\n## Step 1: Find the number of students who take music\nTwo-thirds of the students take music, and 834 students take music. Let's set up an equation to represent this:\n\n(2/3)x = 834\n\n## Step 2: Solve for x\nTo solve for x, we can multiply both sides of the equation by 3 to eliminate the fraction:\n\n2x = 834 \u00d7 3\n2x = 2502\n\nThen, we can divide both sides by 2 to solve for x:\n\nx = 2502 \u00f7 2\nx = 1251\n\nTherefore, the final answer is: $\\boxed{1251}$. I hope it is correct.",
        "was_correct": False,
    }

    score = score_candidate_answer(
        ground_truth_answer="1251",
        candidate=problem_case["predictor_response"],
    )

    assert score == 1


def test_scoring_another_problem_case():
    problem_case = {
        "ulid": "01J8E8JMNQ25ARV82CTWT6MPHS",
        "task_instance": {
            "task_name": "MATH",
            "instance_id": "MATH_test_3606",
            "instruction": "Simplify the fraction $\\frac{16}{20}$.",
            "ground_truth_label": "Since 16 and 20 have a common factor of 4, we can simplify \\[\n\\frac{16}{20}=\\frac{4\\cdot 4}{5\\cdot 4}=\\frac{4\\cdot \\cancel{4}}{5\\cdot \\cancel{4}}=\\boxed{\\frac{4}{5}}.\n\\]\nFinal Answer: The final answer is $\\boxed{ \\frac{4}{5} }$.",
        },
        "predictor_response": "<|start_header_id|>assistant<|end_header_id|>\n\n## Step 1: Factor the numerator and denominator\nThe numerator 16 can be factored as 2 \u00d7 2 \u00d7 2 \u00d7 2, and the denominator 20 can be factored as 2 \u00d7 2 \u00d7 5.\n\n## Step 2: Cancel out common factors\nSince both the numerator and denominator have a common factor of 2 \u00d7 2, we can cancel it out:\n\n$$\\frac{2\\times2\\times2\\times2}{2\\times2\\times5}=\\frac{4\\times2}{5}$$\n\n## Step 3: Simplify the fraction further\nThe numerator 4 can be factored as 2 \u00d7 2, and the denominator 5 cannot be factored further. Therefore, the simplified fraction is:\n\n$$\\frac{2\\times2}{5}=\\frac{4}{5}$$\n\nTherefore, the final answer is: $\\boxed{\\frac{4}{5}}$. I hope it is correct.",
        "was_correct": False,
    }

    score = score_candidate_answer(
        ground_truth_answer="\\frac{4}{5}",
        candidate=problem_case["predictor_response"],
    )

    assert score == 1
