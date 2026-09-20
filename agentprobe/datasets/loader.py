import json
from pathlib import Path

from agentprobe.models.schemas import Task


def load_dataset(path: str) -> list[Task]:
    dataset_path = Path(path)
    tasks = []

    with dataset_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)
                task = Task.model_validate(data)
                tasks.append(task)
            except Exception as e:
                raise ValueError(
                    f"Invalid task at {dataset_path}:{line_number}: {e}"
                ) from e

    return tasks
