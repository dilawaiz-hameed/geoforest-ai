# Contributing

Four people on one repository for two semesters. These rules exist so that
`main` is always the thing that is deployed and always works.

## Branches

```
main                 protected. deployed on every merge. never pushed to directly.
└── develop          integration branch. everyone's work meets here first.
    ├── feature/...  new capability
    ├── fix/...      something is wrong
    ├── docs/...     report, README, diagrams
    └── chore/...    dependencies, CI, tooling
```

Branch names say what the work is, not who is doing it:
`feature/deepforest-adapter`, `fix/tile-edge-clamp`, `docs/chapter-6`.
Not `nashwa-branch`, not `test2`, not `final-final`.

### The flow

```bash
git switch develop && git pull
git switch -c feature/deepforest-adapter
# ... work, commit in small pieces ...
git push -u origin feature/deepforest-adapter
# open a PR into develop
```

When `develop` is green and a milestone is done, open a PR from `develop`
into `main`. Merging it deploys. That is the only way anything reaches the
live interface.

## Pull requests

Required into both `develop` and `main`:

- CI green — lint, tests on three Python versions, Docker image builds and
  answers a real request
- one approving review from someone who did not write the code
- a description that says what changed and how you know it works

A PR that only adds code is suspicious. New behaviour comes with a test that
fails without it.

## Commits

One logical change per commit, present tense, under ~70 characters:

```
Add cross-tile de-duplication to the pipeline
Clamp the final tile origin so tiles stay full size
Fix NMS threshold convention to match torchvision
```

Not `update`, not `fixes`, not `asdasd`.

## Tests

```bash
export PYTHONPATH=src
pytest                      # everything
pytest tests/test_pipeline.py -v
ruff check . && ruff format .
```

Rules that have earned their place:

- **No network, no weights, no fixture images.** Scenes are generated with a
  seeded RNG. The suite runs in seconds, so nobody is tempted to skip it.
- **A test must be able to fail.** `test_tree_in_overlap_is_counted_once`
  asserts `raw_count > 1` before asserting `count == 1`; without that first
  line the test would pass even if the tree were never detected twice and
  de-duplication were doing nothing.
- **Coverage of `src/geoforest` must stay at or above 90%.** CI fails below
  it. This is the counting logic; an uncovered branch there is an uncounted
  tree.
- **If a test and the code disagree, work out which is wrong.** When
  `test_threshold_boundary_is_inclusive` failed, the implementation was
  right and the test was wrong: torchvision suppresses only on
  `IoU > threshold`. The test was corrected and the convention written into
  the `nms()` docstring, so swapping in a GPU implementation later will not
  silently change counts.

## Adding a detector

Satisfy the protocol in `src/geoforest/detector.py` — `name`,
`is_real_model`, `predict(tile) -> (N, 5)` of `[x0, y0, x1, y1, score]` in
tile-local pixels — and register it in `get_detector()`. Nothing downstream
changes: tiling, merging, the API, the interface and their tests all work
against the protocol, not against a model.

Set `is_real_model` honestly. The interface shows a warning when it is
`False`, and that warning is what keeps a placeholder count from being
read as a result.

## Secrets

Never commit tokens, keys, service-account JSON or Earth Engine
credentials. `HF_TOKEN` lives in GitHub repository secrets. Imagery and
model weights are not committed; `data/` and `outputs/` are ignored.
