"""Streamlit Community Cloud entry point."""

from comp_climbing_app import main


# Changing the entry point forces Community Cloud to restart imported modules
# when a release replaces the projection engine rather than only page content.
APP_RELEASE = "physical-board-governed-release-2026-08-13"


main()
