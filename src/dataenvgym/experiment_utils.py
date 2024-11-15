from pathlib import Path
from git import Repo

# NOTE: There is some bad naming here. We use run in some places and experiment in others.
# This should be cleaned up. For now, know that "run" and "experiment" are synonyms.
# This file contains code for generating paths to the output directory for experiments
# based on the script name.


def get_experiment_path_from_script_name(script_name: str) -> Path:
    # Each experiments file is named something like
    # 001_skill_conditioned_gqa.py
    # the output should go in workspace/experiments__001_skill_conditioned_gqa
    output_dir_name = f"experiments__{Path(script_name).stem}"
    output_dir = Path("workspace") / output_dir_name
    return output_dir


def make_output_dir_for_run(script_name: str) -> Path:
    # Each experiments file is named something like
    # 001_skill_conditioned_gqa.py
    # the output should go in workspace/experiments__001_skill_conditioned_gqa

    # Build the name of the output directory
    output_dir_name = f"experiments__{Path(script_name).stem}"

    # Create the output directory
    output_dir = Path("workspace") / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def write_current_commit_hash_to_file(output_dir: Path, repo_path: str = ".") -> None:
    repo = Repo(repo_path)
    commit_hash = repo.head.commit.hexsha
    with open(output_dir / "commit_hash.txt", "w") as f:
        f.write(commit_hash)


def get_output_dir_for_prev_experiment(experiment_identifier: str | int | Path) -> Path:
    """
    If a Path is given, return it directly.
    If an int is given, look in `experiments/` and return the path to the experiment starting with that number.
    Throw an error if there is not exactly one match.
    If an experiment name is given, return the path to the experiment with that name.
    """
    experiments_dir = Path("experiments")

    if isinstance(experiment_identifier, Path):
        return experiment_identifier

    if isinstance(experiment_identifier, int):
        matches = []
        for path in experiments_dir.iterdir():
            try:
                # Split at first underscore and try to parse the number
                experiment_num = int(path.name.split("_")[0])
                if experiment_num == experiment_identifier:
                    matches.append(path)
            except ValueError:
                # Skip files that don't start with a parseable number
                continue

        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one match for experiment number {experiment_identifier}, found {len(matches)}: {matches}"
            )
        script_name = matches[0].name
        experiment_path = get_experiment_path_from_script_name(script_name)
        if not experiment_path.exists():
            raise ValueError(
                f"No experiment found with number {experiment_identifier}."
            )
        return experiment_path

    if isinstance(experiment_identifier, str):
        experiment_path = experiments_dir / experiment_identifier
        if not experiment_path.exists():
            raise ValueError(f"No experiment found with name {experiment_identifier}.")
        return get_experiment_path_from_script_name(experiment_path.name)

    raise TypeError("experiment_identifier must be a str, int, or Path.")
