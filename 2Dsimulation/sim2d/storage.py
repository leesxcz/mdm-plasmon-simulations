"""Package paths and scientific data serialization."""

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, complex):
        return {"re": value.real, "im": value.imag}
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def canonical(value):
    def encode(item):
        if isinstance(item, complex):
            return [item.real, item.imag]
        return item.tolist()

    return json.dumps(value, sort_keys=True, default=encode, separators=(",", ":"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), indent=2, allow_nan=False) + "\n", encoding="utf-8")
