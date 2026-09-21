"""Content fingerprints that survive Git checkouts on different operating systems."""
import hashlib
import json
from pathlib import Path


def json_fingerprint(path: Path) -> str | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
