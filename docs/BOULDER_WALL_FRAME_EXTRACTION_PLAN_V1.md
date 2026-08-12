# Boulder wall-frame extraction plan v1

The anchor workflow now has a practical successor: a small planner consumes
*executed, supported* verification records and produces one deterministic frame
candidate per supported M1–M4 anchor window.  It deliberately selects the
midpoint of the largest interval not reported to contain a visible athlete,
with a two-second margin.

That is useful triage for the requested coaching reference library, but it is
not proof that a frame is an empty wall, that all holds are visible, or that it
is the best camera view.  Every planned record therefore has
`REQUIRES_VISUAL_EMPTY_WALL_REVIEW` and `empty_wall_verified: false`.

`scripts/plan_boulder_wall_frames.py` is local/no-download.  It binds the
complete verification JSONL hash and writes a review-only JSON plan.  A later
media worker may download only the official World Climbing video identified in
that plan.  `scripts/extract_boulder_wall_frames.py` is the corresponding
bounded extractor: given only pre-staged `<video_id>.mp4` files, it generates
at most 24 JPEGs using one `ffmpeg` frame command per plan row and records the
media/frame hashes.  It never discovers or downloads a URL.  It must not promote a
candidate directly into a coaching tag, rating feature, or public asset.

This keeps the next cloud job short and useful: first run the existing bounded
Gemini verification job; then plan/extract only its supported anchors rather
than downloading every broadcast blindly.  The packet currently contains
discovery evidence only, so the planner intentionally produces no production
artifact until that verification checkpoint exists.
