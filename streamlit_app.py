"""Streamlit Community Cloud entry point."""

import importlib

import comp_climbing_app
import future_vision_demo


# Changing the entry point forces Community Cloud to restart imported modules
# when a release replaces the projection engine rather than only page content.
APP_RELEASE = "synthetic-future-vision-pathway-taxonomy-2026-08-14"


if getattr(comp_climbing_app, "APP_CODE_RELEASE", None) != APP_RELEASE:
    future_vision_demo = importlib.reload(future_vision_demo)
    comp_climbing_app = importlib.reload(comp_climbing_app)

comp_climbing_app.main()
