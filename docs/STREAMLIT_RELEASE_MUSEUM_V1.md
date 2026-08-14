# Streamlit release museum

Every published version must leave a static, offline record of what the user could see.
The archive is deliberately stricter than a single homepage screenshot.

Each release contains:

- eight Future Vision captures: four fictional personas across History and Development;
- five Evidence Lab captures: one for each primary overview;
- an offline HTML index and one combined PDF;
- a JSON manifest with every state and file hash;
- a Git source bundle pinned to the deployed commit.

The builder fails when any required state is missing. Captures happen before publication;
the archive is then built from the same commit and deployment candidate. This makes it
possible to revisit older interface choices even if the live app or its data later changes.
