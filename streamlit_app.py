"""Streamlit Community Cloud entry point."""

import importlib

import comp_climbing_app


# Changing the entry point forces Community Cloud to restart imported modules
# when a release replaces the projection engine rather than only page content.
APP_RELEASE = "synthetic-future-vision-grant-demo-2026-08-13"


if getattr(comp_climbing_app, "APP_CODE_RELEASE", None) != APP_RELEASE:
    comp_climbing_app = importlib.reload(comp_climbing_app)

comp_climbing_app.main()
