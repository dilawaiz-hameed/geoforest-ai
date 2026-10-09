# One-time setup

Everything in this file is done by hand, once. After it, merging to `main`
deploys and nobody touches a server again.

## 1. Push the repository

The history and both branches already exist in this archive.

```bash
git remote add origin https://github.com/<org-or-user>/geoforest-ai.git
git push -u origin main
git push -u origin develop
```

## 2. Protect `main`

GitHub → Settings → Branches → Add branch ruleset, targeting `main`:

- Require a pull request before merging, with **1 approval**
- Dismiss stale approvals when new commits are pushed
- Require status checks to pass, selecting `Lint`, `Tests (Python 3.11)`
  and `Image builds and serves`
- Require branches to be up to date before merging
- Block force pushes

Add the same ruleset for `develop` without the approval requirement, so
work-in-progress still gets CI but does not need a second person at 2am.

This is the part the evaluation actually looks at. A repository where
everyone pushes to `main` has no CI/CD pipeline, only CI/CD files.

## 3. Create the Hugging Face Space

huggingface.co → New Space:

| Field | Value |
| --- | --- |
| Name | `geoforest-ai` |
| SDK | **Docker** |
| Hardware | CPU basic (free) |
| Visibility | Public |

The Space needs no files of its own. CD pushes this repository into it, and
the YAML block at the top of `README.md` is what configures it — `sdk:
docker` and `app_port: 7860`.

## 4. Give CD its credentials

Hugging Face → Settings → Access Tokens → New token, **write** scope.

GitHub → Settings → Secrets and variables → Actions:

| Kind | Name | Value |
| --- | --- | --- |
| Secret | `HF_TOKEN` | the write token |
| Variable | `HF_USERNAME` | your Hugging Face username |
| Variable | `HF_SPACE` | `geoforest-ai` |

Then GitHub → Settings → Environments → New environment → `production`.
Add the reviewers you want if you would rather approve each deployment;
leave it empty and every merge to `main` goes straight out.

## 5. Prove the path works before the model exists

```bash
git switch develop
git switch -c chore/verify-deployment
# change one line of README copy
git commit -am "Verify the deployment path end to end"
git push -u origin chore/verify-deployment
```

Open the PR into `develop`, watch CI, merge. Open a PR from `develop` into
`main`, merge. Watch `CD` run and the Space come up. Open the Space URL,
upload any image, and check that the count, the raw count and the merged
duplicates all appear.

Do this on day one, with the stub detector. If the deployment path is going
to break, it should break now and not the night before the demo.

## 6. Switch the model on

Once `DeepForestDetector` is validated on real tiles:

1. In the `Dockerfile`, change `ARG INSTALL_ML=false` to `true`
2. In the Space settings, add a variable `GEOFOREST_DETECTOR=deepforest`
3. Open it as a PR like anything else

The interface stops showing the stub warning by itself, because it reads
`is_real_model` from the API rather than being told.

## Local checks before you open a PR

```bash
export PYTHONPATH=src
ruff check . && ruff format --check .
pytest
python scripts/dedup_table.py
docker build -t geoforest . && docker run --rm -p 7860:7860 geoforest
```
