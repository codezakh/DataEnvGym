from pathlib import Path
from git import Repo


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
