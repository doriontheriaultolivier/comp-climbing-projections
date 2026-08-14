"""Build a static, hash-bound archive of a published Streamlit release.

The browser-capture step is deliberately separate: this builder refuses a
partial capture set, then creates an offline HTML index, a combined PDF, a
state manifest, and a source bundle pinned to one Git commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Iterable

from PIL import Image


VISION_PERSONAS = (
    "joe-bigbiceps",
    "maya-transfergap",
    "sam-pressureproof",
    "alex-latebloomer",
)
VISION_VIEWS = ("history", "development")
EVIDENCE_VIEWS = (
    "canadian-pool",
    "ifsc-pool",
    "wr-pool",
    "global-progression",
    "towards-olympics",
)


def required_state_ids() -> tuple[str, ...]:
    vision = tuple(
        f"vision--{persona}--{view}"
        for persona in VISION_PERSONAS
        for view in VISION_VIEWS
    )
    evidence = tuple(f"evidence--{view}" for view in EVIDENCE_VIEWS)
    return vision + evidence


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_captures(capture_dir: Path) -> list[Path]:
    expected = [capture_dir / f"{state_id}.png" for state_id in required_state_ids()]
    missing = [path.name for path in expected if not path.is_file()]
    if missing:
        raise ValueError("Missing required release captures: " + ", ".join(missing))
    return expected


def _write_pdf(images: Iterable[Path], output: Path) -> None:
    pages = [Image.open(path).convert("RGB") for path in images]
    try:
        pages[0].save(output, save_all=True, append_images=pages[1:], resolution=120)
    finally:
        for page in pages:
            page.close()


def _write_html(release_id: str, states: list[dict[str, object]], output: Path) -> None:
    cards = "\n".join(
        f"<article><h2>{state['state_id']}</h2>"
        f"<img src=\"captures/{state['filename']}\" alt=\"{state['state_id']}\"></article>"
        for state in states
    )
    output.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{release_id} - release museum</title>"
        "<style>body{font:16px system-ui;max-width:1400px;margin:auto;padding:24px}"
        "article{margin:30px 0 60px}img{width:100%;border:1px solid #ccd3d1}"
        "h1,h2{color:#102f2b}</style></head><body>"
        f"<h1>{release_id}</h1><p>Static release museum: every primary state captured at publication.</p>"
        f"{cards}</body></html>",
        encoding="utf-8",
    )


def build_release_museum(
    *,
    repo: Path,
    capture_dir: Path,
    output_root: Path,
    release_id: str,
    deploy_url: str,
    commit: str,
) -> Path:
    captures = validate_captures(capture_dir)
    target = output_root / f"{release_id}-{commit[:12]}"
    if target.exists():
        raise FileExistsError(f"Release museum already exists: {target}")
    capture_target = target / "captures"
    capture_target.mkdir(parents=True)

    states: list[dict[str, object]] = []
    for source in captures:
        destination = capture_target / source.name
        shutil.copy2(source, destination)
        states.append(
            {
                "state_id": source.stem,
                "filename": source.name,
                "sha256": sha256(destination),
                "bytes": destination.stat().st_size,
            }
        )

    _write_html(release_id, states, target / "index.html")
    _write_pdf((capture_target / item["filename"] for item in states), target / "all-states.pdf")
    bundle = target / "source.bundle"
    subprocess.run(
        ["git", "bundle", "create", str(bundle), commit],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = {
        "schema": "streamlit-release-museum-v1",
        "release_id": release_id,
        "deploy_url": deploy_url,
        "git_commit": commit,
        "states": states,
        "state_count": len(states),
        "index_sha256": sha256(target / "index.html"),
        "pdf_sha256": sha256(target / "all-states.pdf"),
        "source_bundle_sha256": sha256(bundle),
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--deploy-url", required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    output = build_release_museum(**vars(args))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
