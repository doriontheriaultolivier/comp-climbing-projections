"""Streamlit Community Cloud entry point."""

from comp_climbing_app import main


# Changing the entry point forces Community Cloud to restart imported modules
# when a release replaces the projection engine rather than only page content.
APP_RELEASE = "canadian-current-wc-projection-v3-youth-world-complete"


main()
