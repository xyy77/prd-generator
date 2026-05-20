import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalSample:
    product_idea: str
    category: str
    gold_checklist: list[str] = field(default_factory=list)
    gold_prd: dict | None = None
    expected_modules: list[str] = field(default_factory=list)


DEFAULT_DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "evaluation" / "dataset.json"


def load_dataset(dataset_path: str | None = None) -> list[EvalSample]:
    path = Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        samples.append(EvalSample(
            product_idea=item["product_idea"],
            category=item.get("category", "general"),
            gold_checklist=item.get("gold_checklist", []),
            gold_prd=item.get("gold_prd"),
            expected_modules=item.get("expected_modules", []),
        ))
    return samples
