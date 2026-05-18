from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def resolve_data_path(relative_path: str) -> Path:
    p = Path(relative_path)
    if not p.is_absolute():
        p = get_project_root() / p
    return p


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_safe_filename(name: str, max_len: int = 60) -> str:
    safe = "".join(c for c in name if c.isalnum() or c in "._- ")
    safe = safe.strip().replace(" ", "_")
    return safe[:max_len]
