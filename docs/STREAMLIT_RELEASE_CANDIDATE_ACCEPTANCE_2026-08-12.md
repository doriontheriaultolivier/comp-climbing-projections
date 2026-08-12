# Streamlit release-candidate acceptance — 2026-08-12

## Scope

This is a local acceptance record for the clean deployment candidate based on
`comp-v2/main` commit `8e32e30f013a797f60cbec4ac2e9dad5d9aeba6e`.
It deliberately excludes the separate research worktree, which is both
divergent and actively dirty.

## Checks

| Check | Result |
| --- | --- |
| Focused app smoke and style-tagger tests | 14 passed, 13 subtests passed |
| Full consolidated release-candidate test suite | 25 passed, 13 subtests passed |
| `py_compile` (`streamlit_app.py`, `comp_climbing_app.py`, `style_tagging_app.py`) | Passed |
| Local Streamlit `/_stcore/health` | HTTP 200, `ok` |
| Olympics default view | Deterministic across two fresh AppTest sessions; all four athlete-set caption modes verified semantically |

The test update in this candidate removes a fragile, serialized-protobuf hash
that changed under compatible dependency versions allowed by `requirements.txt`.
It replaces it with a same-environment fresh-render determinism assertion while
retaining the visible-content checks for the comparison, EEQ, youth and
Canadian-proxy views.

The candidate also makes `style_tagging_app.py` genuinely standalone: it owns
its governed local inventory path and no longer imports the projection app just
to find `data/`.  Its Streamlit entrypoint is smoke-tested with the inventory,
so an unrelated projection-app import error cannot prevent the tagger from
starting.

## Deliberate non-actions

- No Streamlit Cloud deploy or visibility setting was changed.
- The existing public URL redirects to Streamlit authentication, so it could
  not be audited as an anonymous public page.
- `comp-climbing-projections.streamlit.net` is not a valid Community Cloud
  custom hostname; the supported Community Cloud endpoint is the
  `.streamlit.app` domain.

## Next operator gate

Review this small candidate independently, then deploy from this clean branch
only.  Confirm **Data health** says Ready, inspect all five sections at desktop
and phone width, and make public visibility a deliberate product decision.
