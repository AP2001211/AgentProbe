from pathlib import Path

from agentprobe.models.schemas import EvaluationRun


class JSONRunStore:
    def __init__(self, directory: str = "runs"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, run: EvaluationRun) -> Path:
        path = self.directory / f"{run.run_id}.json"
        path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
        return path

    def load(self, run_id: str) -> EvaluationRun:
        path = self.directory / f"{run_id}.json"

        if not path.exists():
            raise FileNotFoundError(f"Evaluation run not found: {run_id}")

        return EvaluationRun.model_validate_json(path.read_text(encoding="utf-8"))
