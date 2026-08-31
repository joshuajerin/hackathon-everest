from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "docs/media/manifest.json"
README = ROOT / "README.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_readme_media_bundle_is_complete_and_hash_matched() -> None:
    document = json.loads(MANIFEST.read_text())
    readme = README.read_text()
    artifacts = document["artifacts"]

    assert document["schema_version"] == "1.0.0"
    assert len(artifacts) == 4
    assert "simulator-only" in document["general_claim_boundary"]

    for artifact in artifacts:
        video = ROOT / artifact["published_path"]
        poster = ROOT / artifact["poster_path"]
        assert video.is_file()
        assert poster.is_file()
        assert video.stat().st_size < 25 * 1024 * 1024
        assert sha256(video) == artifact["published_sha256"]
        assert sha256(poster) == artifact["poster_sha256"]
        assert artifact["published_path"] in readme
        assert artifact["poster_path"] in readme
        assert artifact["decode_check"] == "pass"
        assert artifact["claim_boundary"]
