from dataenvgym.gym.data_generation_agents.code.baselines.open_ended import (
    DataGenerationAgent,
)
from dataenvgym.gym.domain_models import (
    CodeGenerationCompletedTaskInstance,
    CodeGenerationTaskInstance,
    CodeGenerationPredictorInterface,
)
from ulid import ULID
from typing import Sequence
import rich


class StubCodeGenerationPredictor:
    def predict(
        self, task_instances: Sequence[CodeGenerationTaskInstance]
    ) -> list[str]:
        # Emit a constant answer for any input
        return ["# This is a stub response"] * len(task_instances)


def test_error_conditioned_baseline_code_data_strategy():
    # Define two CodeGenerationCompletedTaskInstance objects
    correct_task_instance = CodeGenerationCompletedTaskInstance(
        ulid=ULID(),
        task_instance=CodeGenerationTaskInstance(
            task_name="Penalty Kicks",
            instance_id="1",
            instruction="### Question:\nTakahashi will have N penalty kicks in a soccer match.\nFor the i-th penalty kick, he will fail if i is a multiple of 3, and succeed otherwise.\nPrint the results of his penalty kicks.\n\nInput\n\nThe input is given from Standard Input in the following format:\nN\n\nOutput\n\nPrint a string of length N representing the results of Takahashi's penalty kicks. The i-th character (1 \\leq i \\leq N) should be o if Takahashi succeeds in the i-th penalty kick, and x if he fails.\n\nConstraints\n\n\n- 1 \\leq N \\leq 100\n- All inputs are integers.\n\nSample Input 1\n\n7\n\nSample Output 1\n\nooxooxo\r\n\nTakahashi fails the third and sixth penalty kicks, so the third and sixth characters will be x.\n\nSample Input 2\n\n9\n\nSample Output 2\n\nooxooxoox\n\n### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.\n```python\n# YOUR CODE HERE\n```\n\n### Answer: (use the provided format with backticks)\n\n",
            solution="```python\n# YOUR CODE HERE\nn = int(input())\nans = ''\nfor i in range(1, n + 1):\n    if i % 3 == 0:\n        ans += 'x'\n    else:\n        ans += 'o'\nprint(ans)\n```",
            starter_code=None,
        ),
        predictor_response="```python\n# YOUR CODE HERE\nn = int(input())\nans = ''\nfor i in range(1, n + 1):\n    if i % 3 == 0:\n        ans += 'x'\n    else:\n        ans += 'o'\nprint(ans)\n```",
        was_correct=True,
    )

    incorrect_task_instance = CodeGenerationCompletedTaskInstance(
        ulid=ULID(),
        task_instance=CodeGenerationTaskInstance(
            task_name="Longest Monotonic Subarray",
            instance_id="2",
            instruction="### Question:\nYou are given an array of integers nums. Return the length of the longest subarray of nums which is either strictly increasing or strictly decreasing.\n \nExample 1:\n\nInput: nums = [1,4,3,3,2]\nOutput: 2\nExplanation:\nThe strictly increasing subarrays of nums are [1], [2], [3], [3], [4], and [1,4].\nThe strictly decreasing subarrays of nums are [1], [2], [3], [3], [4], [3,2], and [4,3].\nHence, we return 2.\n\nExample 2:\n\nInput: nums = [3,3,3,3]\nOutput: 1\nExplanation:\nThe strictly increasing subarrays of nums are [3], [3], [3], and [3].\nThe strictly decreasing subarrays of nums are [3], [3], [3], and [3].\nHence, we return 1.\n\nExample 3:\n\nInput: nums = [3,2,1]\nOutput: 3\nExplanation:\nThe strictly increasing subarrays of nums are [3], [2], and [1].\nThe strictly decreasing subarrays of nums are [3], [2], [1], [3,2], [2,1], and [3,2,1].\nHence, we return 3.\n\n \nConstraints:\n\n1 <= nums.length <= 50\n1 <= nums[i] <= 50\n\n### Format: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n```python\nclass Solution:\n    def longestMonotonicSubarray(self, nums: List[int]) -> int:\n        \n```\n\n### Answer: (use the provided format with backticks)\n\n",
            solution="```python\nclass Solution:\n    def longestMonotonicSubarray(self, nums: List[int]) -> int:\n        if len(nums) == 1:\n            return 1\n        increasing = 1\n        decreasing = 1\n        max_len = 1\n        for i in range(1, len(nums)):\n            if nums[i] > nums[i-1]:\n                increasing += 1\n                decreasing = 1\n            elif nums[i] < nums[i-1]:\n                decreasing += 1\n                increasing = 1\n            else:\n                increasing = 1\n                decreasing = 1\n            max_len = max(max_len, increasing, decreasing)\n        return max_len\n```",
            starter_code=None,
        ),
        predictor_response="```python\nclass Solution:\n    def longestMonotonicSubarray(self, nums: List[int]) -> int:\n        if len(nums) == 1:\n            return 1\n        increasing = 1\n        decreasing = 1\n        max_len = 1\n        for i in range(1, len(nums)):\n            if nums[i] > nums[i-1]:\n                increasing += 1\n                decreasing = 1\n            elif nums[i] < nums[i-1]:\n                decreasing += 1\n                increasing = 1\n            else:\n                increasing = 1\n                decreasing = 1\n            max_len = max(max_len, increasing, decreasing)\n        return max_len\n```",
        was_correct=False,
    )

    # Create an instance of the strategy
    strategy = DataGenerationAgent()

    # Generate training data
    completed_task_instances = [correct_task_instance, incorrect_task_instance]
    training_data = strategy(completed_task_instances, StubCodeGenerationPredictor())

    rich.print(training_data)

    # Assert that the expected amount of data was produced
    assert len(training_data) == 3  # Since datum_to_generate_per_error is 3 by default
