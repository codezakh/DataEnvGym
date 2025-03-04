from dataenvgym.gym.tasks.code.livecodebench_task import LiveCodeBenchEvaluationManager
import sys


if __name__ == "__main__":
    print(sys.argv)  # debugging
    LiveCodeBenchEvaluationManager.cli_entrypoint()
