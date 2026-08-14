"""Build the complete, full-scroll Streamlit release museum (V2).

V1 proved the publication state could be archived, but its browser captures
were viewport-sized.  V2 is deliberately additive: it preserves every
persona/history combination, the expanded model-choice explanation, and all
five real-evidence overview states, and it rejects viewport-only or duplicate
captures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from PIL import Image

try:
    from scripts.build_streamlit_release_museum import (
        EVIDENCE_VIEWS,
        VISION_PERSONAS,
        VISION_VIEWS,
        _write_pdf,
        sha256,
    )
except ModuleNotFoundError:  # Support direct ``python scripts/...`` execution.
    from build_streamlit_release_museum import (  # type: ignore[no-redef]
        EVIDENCE_VIEWS,
        VISION_PERSONAS,
        VISION_VIEWS,
        _write_pdf,
        sha256,
    )


MIN_FULL_PAGE_HEIGHT = 1_000


def required_state_ids_v2() -> tuple[str, ...]:
    vision = tuple(
        f"vision--{persona}--{view}"
        for persona in VISION_PERSONAS
        for view in VISION_VIEWS
    )
    evidence = tuple(f"evidence--{view}" for view in EVIDENCE_VIEWS)
    return vision + ("vision--model-choice-expanded",) + evidence


def validate_full_page_captures(capture_dir: Path) -> list[Path]:
    expected = [capture_dir / f"{state_id}.png" for state_id in required_state_ids_v2()]
    missing = [path.name for path in expected if not path.is_file()]
    if missing:
        raise ValueError("Missing required V2 release captures: " + ", ".join(missing))

    hashes: dict[str, str] = {}
    for path in expected:
        try:
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
        except Exception as exc:  # Pillow exposes several format-specific errors.
            raise ValueError(f"Invalid release capture: {path.name}") from exc
        if width < 1_000 or height < MIN_FULL_PAGE_HEIGHT:
            raise ValueError(
                f"Capture is not full-page: {path.name} is {width}x{height}"
            )
        digest = sha256(path)
        if digest in hashes:
            raise ValueError(
                f"Duplicate release captures: {path.name} and {hashes[digest]}"
            )
        hashes[digest] = path.name
    return expected


def _write_html_v2(
    release_id: str, states: list[dict[str, object]], output: Path
) -> None:
    navigation = "".join(
        f'<a href="#{state["state_id"]}">{state["state_id"]}</a><br>'
        for state in states
    )
    cards = "\n".join(
        f"<article id=\"{state['state_id']}\"><h2>{state['state_id']}</h2>"
        f"<p>{state['width']} × {state['height']} px · full-scroll capture</p>"
        f"<img loading=\"lazy\" src=\"captures/{state['filename']}\" "
        f"alt=\"{state['state_id']}\"></article>"
        for state in states
    )
    output.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{release_id} - complete release museum</title>"
        "<style>body{font:16px system-ui;max-width:1400px;margin:auto;padding:24px}"
        "nav{columns:2}article{margin:40px 0 80px}img{width:100%;border:1px solid #ccd3d1}"
        "h1,h2{color:#102f2b}</style></head><body>"
        f"<h1>{release_id}</h1><p>Complete full-scroll release museum. "
        "Synthetic Vision states and real Evidence states remain explicitly separate.</p>"
        f"<nav>{navigation}</nav>"
        f"{cards}</body></html>",
        encoding="utf-8",
    )


def build_release_museum_v2(
    *,
    repo: Path,
    capture_dir: Path,
    output_root: Path,
    release_id: str,
    deploy_url: str,
    commit: str,
) -> Path:
    captures = validate_full_page_captures(capture_dir)
    target = output_root / f"{release_id}-{commit[:12]}"
    if target.exists():
        raise FileExistsError(f"Release museum V2 already exists: {target}")
    capture_target = target / "captures"
    capture_target.mkdir(parents=True)

    states: list[dict[str, object]] = []
    for source in captures:
        destination = capture_target / source.name
        shutil.copy2(source, destination)
        with Image.open(destination) as image:
            width, height = image.size
        states.append(
            {
                "state_id": source.stem,
                "filename": source.name,
                "sha256": sha256(destination),
                "bytes": destination.stat().st_size,
                "width": width,
                "height": height,
                "capture_mode": "stitched_full_scroll",
            }
        )

    _write_html_v2(release_id, states, target / "index.html")
    _write_pdf(
        (capture_target / str(item["filename"]) for item in states),
        target / "all-states.pdf",
    )
    source_archive = target / "source.zip"
    subprocess.run(
        [
            "git",
            "archive",
            "--format=zip",
            f"--output={source_archive.resolve()}",
            commit,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = {
        "schema": "streamlit-release-museum-v2",
        "capture_contract": "complete_full_scroll_primary_states_v1",
        "release_id": release_id,
        "deploy_url": deploy_url,
        "git_commit": commit,
        "states": states,
        "state_count": len(states),
        "index_sha256": sha256(target / "index.html"),
        "pdf_sha256": sha256(target / "all-states.pdf"),
        "source_archive_sha256": sha256(source_archive),
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
    print(build_release_museum_v2(**vars(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
