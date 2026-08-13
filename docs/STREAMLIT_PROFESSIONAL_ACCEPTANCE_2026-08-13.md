# Streamlit professional acceptance — 2026-08-13

## Outcome

The current remote `main` candidate at commit
`d6396c16e75dd517a304f64bdc4006616e782967` passes local professional
acceptance in a clean worktree. This is the release app repository, not a
research or rebuilt-data worktree.

The application is technically deployable under its declared dependency
contract. This result does not promote any research-only rating or probability
artifact, change Streamlit visibility, or claim that the public URL is
anonymous.

## Environment

Acceptance used a newly created virtual environment outside the repository:

| Dependency | Version |
|---|---:|
| Python | 3.12.13 |
| Streamlit | 1.61.1 |
| pandas | 2.3.3 |
| pyarrow | 22.0.0 |
| NumPy | 2.5.2 |
| Plotly | 6.9.0 |
| pytest | 9.1.1 |

These versions satisfy `requirements.txt`, including the release constraint
`pyarrow>=14,<23`. The existing research environment had pyarrow 24 and was
therefore not used as release evidence.

## Checks

| Check | Result |
|---|---|
| All tracked tests | 115 passed, 22 subtests passed |
| Test runtime | 132.15 seconds |
| Release entry-point compilation | Passed |
| Main Streamlit AppTest | Passed without exceptions |
| Overview sections | Global overview, Global progression, IFSC Pool, WR Pool and Towards Olympics covered |
| Athlete identity and missing-evidence behavior | Passed |
| Projected WC-readiness display and model-status behavior | Passed |
| Probability-spectrum and target-event safeguards | Passed |
| Physical-to-board coaching slice | Passed |
| Standalone Boulder tagging application | Passed |
| Local `/_stcore/health` | HTTP 200, body `ok` |
| Local root application shell | HTTP 200, 10,626 bytes |

The app server used loopback address `127.0.0.1`, port `8765`, and was stopped
after the health check. No background acceptance server remains.

## CI correction

The GitHub release workflow previously named an older subset of test files.
Newer merged tests for athlete-profile integrity, probability-spectrum
visibility, current WC model status and the physical coaching slice could have
regressed without CI noticing. The workflow now runs `python -m pytest -q`,
which discovers every tracked release test.

This is preferable to maintaining another manual list: adding a tracked test
automatically expands the release gate.

## Remaining operator checks

Before changing public visibility or presenting the app as a production
service:

1. confirm the deployed Community Cloud app is built from this `main` commit;
2. sign in and verify **Data health** reports Ready;
3. inspect all five sections at desktop and phone width;
4. confirm the standalone tagging app opens and saves only to its intended
   review backend or local artifact;
5. keep research-only YW-IFSC, REG-IFSC and WC+ readiness challengers out of
   the release until their own gates pass; and
6. deliberately choose private or public visibility rather than treating an
   authentication redirect as a health check.

The application remains on the supported `.streamlit.app` hostname. A
`.streamlit.net` subdomain is not a Community Cloud deployment target.
