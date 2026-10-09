# One-time setup

Everything in this file is done by hand, once. After it, merging to `main`
deploys and nobody touches a server again.

## 1. Push the repository

```bash
git remote add origin https://github.com/<org-or-user>/geoforest-ai.git
git push -u origin main
git push -u origin develop
```

## 2. Create the Render service

Render runs the same Dockerfile that CI builds and tests, so what deploys is
what was verified.

render.com → **New** → **Web Service** → connect this GitHub repository.

| Field | Value |
| --- | --- |
| Language | **Docker** |
| Branch | `main` |
| Instance type | **Free** |
| Health check path | `/_stcore/health` |
| Auto-Deploy | **Off** |

Auto-Deploy is off on purpose. With it on, Render deploys whenever `main`
changes and GitHub Actions is left out of the loop. With it off, the CD
workflow triggers the deploy and then verifies it, which is the pipeline the
evaluation is asking for.

The free instance sleeps after about 15 minutes idle and takes roughly a
minute to wake. Open the URL a few minutes before any demo.

## 3. Give CD its credentials

In the Render service: **Settings** → **Deploy Hook** → copy the URL. It
contains a key, so treat it as a password.

GitHub → Settings → Secrets and variables → Actions:

| Kind | Name | Value |
| --- | --- | --- |
| Secret | `RENDER_DEPLOY_HOOK` | the deploy hook URL |
| Variable | `SERVICE_URL` | your service URL, e.g. `https://geoforest-ai.onrender.com`, no trailing slash |

Then GitHub → Settings → Environments → New environment → `production`.
Leave it empty, or add reviewers if you want to approve each deployment.

## 4. Protect `main`

GitHub → Settings → Rules → Rulesets → New branch ruleset, targeting `main`:

- Require a pull request before merging
- Require status checks to pass: `Lint`, `Tests (Python 3.11)`,
  `Image builds and serves`
- Require branches to be up to date before merging
- Block force pushes

Add the same ruleset for `develop` without the pull-request requirement, so
work in progress still gets CI but does not need a second person at 2am.

**Approvals need a second person.** GitHub will not let you approve your own
pull request, so if you turn on "Require approvals" while you are the only
collaborator, nothing can ever merge. Add your teammates first
(Settings → Collaborators), then set approvals to 1.

This section is the part the evaluation actually looks at. A repository where
everyone pushes to `main` has no CI/CD pipeline, only CI/CD files.

## 5. Prove the path works before the model exists

```bash
git switch develop && git pull
git switch -c chore/verify-deployment
# change one line of README copy
git commit -am "Verify the deployment path end to end"
git push -u origin chore/verify-deployment
```

Open the PR into `develop`, watch CI, merge. Open a PR from `develop` into
`main`, merge. Watch `CD` trigger the deploy and wait for the service. Open
the URL, upload any image, and check that the count, the raw count and the
merged duplicates all appear.

Do this on day one, with the stub detector. If the deployment path is going
to break, it should break now and not the night before the demo.

## 6. Switch the model on

Once `DeepForestDetector` is validated on real tiles:

1. In the `Dockerfile`, change `ARG INSTALL_ML=false` to `true`
2. In Render → Environment, add `GEOFOREST_DETECTOR=deepforest`
3. Open it as a PR like anything else

The interface stops showing the stub warning by itself, because it reads
`is_real_model` from the API rather than being told.

Watch memory. The free instance has 512 MB, which is comfortable for the stub
and tight once torch is installed. If the service dies on boot after the
switch, that is why — either move to a paid instance for the demo, or keep
the deployed build on the stub and run DeepForest locally.

## Local checks before you open a PR

```bash
export PYTHONPATH=src          # Windows cmd: set PYTHONPATH=src
ruff check . && ruff format --check .
pytest
python scripts/dedup_table.py
docker build -t geoforest . && docker run --rm -p 7860:7860 geoforest
```
