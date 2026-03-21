import json
import re
from datetime import datetime, timezone
from pathlib import Path
from saroku.core.schema import BehavioralScores

RESULTS_DIR = Path(".saroku_results")


def save_run(scores: BehavioralScores) -> Path:
    slug = re.sub(r"[^a-zA-Z0-9._-]", "_", scores.model)
    run_dir = RESULTS_DIR / slug
    run_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.fromtimestamp(scores.timestamp, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = run_dir / f"{ts}_{scores.run_id[:8]}.json"
    path.write_text(json.dumps(scores.model_dump(), indent=2))
    return path


def list_runs(model: str | None = None) -> list[Path]:
    RESULTS_DIR.mkdir(exist_ok=True)
    if model:
        slug = re.sub(r"[^a-zA-Z0-9._-]", "_", model)
        return sorted((RESULTS_DIR / slug).glob("*.json"))
    return sorted(RESULTS_DIR.rglob("*.json"))
