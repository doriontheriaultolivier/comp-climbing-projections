# Comp Climbing Projections

> **Boulder-first release:** strength, depth and progression of Canadian
> climbers, from local competitions to the Olympics.

This is the focused successor to
[`ifsc-performance-projections`](https://github.com/doriontheriaultolivier/ifsc-performance-projections).
The [former live application](https://ifsc-performance-projections.streamlit.app/)
stays available as a frozen comparison and rollback release.

**Live Boulder product:**
[comp-climbing-projections.streamlit.app](https://comp-climbing-projections.streamlit.app/)

## What the Overview answers

- **Canadian Pool:** where every matched CNR athlete sits on Global, IFSC and
  World Ranking evidence, with recent movement and data recency.
- **IFSC Pool:** Canadians beside every 2025–2026 IFSC Boulder finalist.
- **WR Pool:** the current top 40 and every current Canadian participant,
  including country comparisons.
- **Global progression:** same-age pathway comparisons, empirical reference
  lines and a clearly labelled bounded-trend hypothesis.
- **Towards Olympics:** a compact view of readiness, World Ranking access and
  evidence gaps; it does not claim qualification odds before the LA28 pathway
  model is governed.

The top ribbon supports three individual athletes, EEQ, Canadian 2026 Youth
Worlds participants, or a clearly labelled CNR top-15 national-team proxy.
The official national-team roster will replace that proxy when supplied.

## Rating contract

`Global-ELO` uses every de-duplicated local, national, international, youth
and senior Boulder round on one Open World-Cup-readiness scale. Confirmed
Onsight, Scramble and Flash rounds have specialist Global ratings. `IFSC-ELO`
uses non-para IFSC results; `WR-ELO` uses events in the current IFSC World
Ranking window. Both provide Qualies, Semies and Finals specialists.
`Performance-ELO` is one round's isolated level, never a stable athlete rating.

All displayed families share an intuitive anchor: **2000 is the fitted 50%
semifinal-advancement level at a randomly sampled 2025 IFSC Open World Cup**.
It is estimated from pre-competition ratings and actual advancement, separately
for the men's and women's pools. The translation does not change athlete order,
spacing or historical updates.

Specialists require at least two eligible contests and shrink toward
`Global-ELO` while evidence is limited. Correlations with `WR-ELO` are
descriptive and cannot isolate setting, training environment, attendance,
travel or selection effects. See
[the full contract](docs/BOULDER_RATING_FAMILY_CONTRACT.md).

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\streamlit run streamlit_app.py
```

The app starts safely when an artifact is missing and names the missing file
under **Data health**.

## Streamlit Community Cloud

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and choose
   **Create app**.
2. Select `doriontheriaultolivier/comp-climbing-projections`, branch `main`,
   and main file `streamlit_app.py`.
3. In **Advanced settings**, select Python 3.12. The public Overview needs no
   secrets.
4. Choose `comp-climbing-projections` as the subdomain when available.
5. Deploy, confirm **Data health** says Ready, then test all five sections at
   desktop and phone width.

The free Community Cloud name ends in `.streamlit.app`. Streamlit does not
offer a user-selected `.streamlit.net` subdomain. That address would require a
separately owned domain and forwarding or proxy configuration.

## Hugging Face Spaces

1. Create a new **Docker** Space.
2. Push this repository to the Space.
3. Keep the supplied `Dockerfile`; it installs the same free dependencies,
   exposes port `7860`, and checks Streamlit health.
4. Confirm the Space reports Running and **Data health** says Ready.

## Data and licensing

The interface reads compact, prebuilt artifacts so it remains within free
hosting memory. The source warehouse and rebuild pipeline remain in the
research repository. See [LICENSE.md](LICENSE.md) and
[DATA_USE_TERMS.md](DATA_USE_TERMS.md). Viewing the product does not grant a
licence to reuse its code, data, model or approach.
