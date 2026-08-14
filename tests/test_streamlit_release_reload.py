from __future__ import annotations

import ast
from pathlib import Path

import comp_climbing_app


ROOT = Path(__file__).resolve().parents[1]


def _entry_release() -> str:
    tree = ast.parse((ROOT / "streamlit_app.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "APP_RELEASE" for target in node.targets):
                return ast.literal_eval(node.value)
    raise AssertionError("APP_RELEASE is not defined")


def test_entrypoint_and_app_module_share_release_identifier() -> None:
    assert _entry_release() == comp_climbing_app.APP_CODE_RELEASE


def test_entrypoint_has_release_mismatch_reload_guard() -> None:
    source = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    assert 'getattr(comp_climbing_app, "APP_CODE_RELEASE", None) != APP_RELEASE' in source
    assert "import future_vision_demo" in source
    assert "importlib.reload(future_vision_demo)" in source
    assert "importlib.reload(comp_climbing_app)" in source
    assert source.index("importlib.reload(future_vision_demo)") < source.index(
        "importlib.reload(comp_climbing_app)"
    )
