# MASK-1 sample set (synthetic only)

- `reference/<name>.png` -- fc-mobile's committed synthetic stand-ins for its seven dev
  fixtures (`src/dev-fixtures/synthetic`, drawn by its `scripts/synthetic-fixtures.mjs`:
  flat geometric figures, no photos). RGBA; the alpha is the reference matte, playing the
  part of the July BiRefNet cut-outs.
- `originals/<entry>.png` -- each reference composited over a studio backdrop by
  `_generate.py`: what the mask lane receives. `madoka-black-dress` reproduces the July
  failure case (black dress on a black marble floor); `spike-collector-photo` carries a
  collector-photo origin and must get no mask.
- `manifest.json` -- entries, backdrops and synthetic origin URLs (nothing is fetched).
- `client_golden.json` -- the grounding fc-mobile's own client code measures on each
  reference alpha; the lane must report the same numbers.

No store or MFC image is committed here, and none may be: real figures are checked
locally against the July artifacts, never in the repo.
