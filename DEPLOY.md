# Landslide app deployment

Last verified: 22 September 2026 (Asia/Dhaka).

The Rangamati Hill Tracts (a district within Chattogram division, Bangladesh)
landslide app is a **Python Flask web service on Render**. It is separate from the
Cloudflare portfolio. **Pushing an app change to the public repository's `main`
branch automatically rebuilds and deploys this service.**

## Current setup

| Item | Value |
| --- | --- |
| Public application | <https://landslide-rangamati.onrender.com/> |
| Deployment repository | [Ricko-Shaha/Landslide-Prediction](https://github.com/Ricko-Shaha/Landslide-Prediction) — public |
| Branch | `main` |
| Render service | `landslide-rangamati` |
| Dashboard | [Open the service](https://dashboard.render.com/web/srv-daoot76gekts73b3ktb0) |
| Configuration | Blueprint-managed using [render.yaml](render.yaml) |
| Region / instance | Singapore / Free |
| Python | 3.13, selected by [.python-version](.python-version) |
| Root directory | Repository root; leave the Render setting blank |
| Build command | `pip install -r requirements-production.txt && python -m unittest discover -s tests` |
| Start command | `python web/serve.py` |
| Health check | `/healthz` |
| Production server | Waitress, one process with four threads |
| Database / persistent disk | None |

`web/serve.py` loads the saved model before accepting traffic and listens on
Render's assigned `PORT`, bound to `0.0.0.0`. Locally its default port is `10000`.
Do not use Flask's development server as the Render start command.

## How this hosting was created

1. The replacement app was pushed to the existing
   `Ricko-Shaha/Landslide-Prediction` repository, preserving its Git history.
2. Render was signed into using GitHub. GitHub access was then granted to this
   specific repository; signing into Render alone did not select the repository.
3. A Render Blueprint was created from the repository's `render.yaml`.
4. The Blueprint created one Free Python service named `landslide-rangamati`
   in Singapore, with the build, start, and health-check settings above.
5. The first build installed the pinned runtime, ran the tests, and started Waitress.
6. The portfolio's app link was updated to the new HTTPS `onrender.com` address.

There is also a private `Ricko-Shaha/rangamati-app` copy used during setup. It is
**not the repository connected to the live service**. The earlier temporary
Render service named `rangamati` is suspended; use `landslide-rangamati` for updates.

To recreate the service in another account, connect the same repository through
**New → Blueprint**, review `render.yaml`, and keep the instance type **Free**.
The requested name may receive a suffix if unavailable. See Render's
[Blueprint specification](https://render.com/docs/blueprint-spec).

## Deploy a normal update

From a clone of `Ricko-Shaha/Landslide-Prediction`, on `main`:

```sh
python -m pip install -r requirements-production.txt
python -m unittest discover -s tests -v
git status
```

Review the change, stage the intended files, commit, and run:

```sh
git push origin main
```

Watch the service's **Deploys** page. Confirm the displayed commit matches the
one pushed and the deployment becomes **Live**. Render reruns the configured
build tests. Open the public site and verify the changed behavior as well;
successful deployment status alone does not prove the expected content is served.

For Markdown-only changes, a commit message such as
`Document hosting [skip render]` saves an unnecessary rebuild. Do not use the skip
phrase for app changes that need publishing. See
[Render deployment controls](https://render.com/docs/deploys#skipping-an-auto-deploy).

## Files shipped with every deployment

The app uses the committed, evaluated artifacts directly. Deploying does not
retrain models or rebuild the full terrain grid.

| File | Purpose |
| --- | --- |
| `models/landslide_model.joblib` | Selected fitted model and feature metadata |
| `reports/metrics.json` | Model evaluation results |
| `data/landslide_inventory.csv` | Inventory used by the surveyed-site demo |
| `data/study_area.json` | Study-area boundary |
| `data/rainfall_grid.json` | Prepared rainfall values |
| `data/susceptibility_grid.json` | Prepared district susceptibility map |
| `web/static/walk_data.js` | Packed terrain data for the walk simulator |
| `notebooks/01_analysis.html` | Readable analysis served at `/notebook` |

Keep model, metrics, grid, and derived terrain data consistent when changing the
analysis. The runtime versions in [requirements-production.txt](requirements-production.txt)
are pinned to load the shipped model. Do not casually replace those pins or
regenerate one artifact without checking its dependants.

The notebook embeds its stylesheet and favicon. After notebook, notebook-theme,
or notebook-branding changes, regenerate its HTML before committing:

```sh
python src/nb_to_html.py
```

## Environment and portfolio connection

The Blueprint sets these non-secret variables:

| Variable | Value / purpose |
| --- | --- |
| `PYTHONUNBUFFERED` | `1`, for immediate Python log output |
| `OMP_NUM_THREADS` | `1`, to limit numerical-library threads |
| `OPENBLAS_NUM_THREADS` | `1`, to limit numerical-library threads |
| `PORT` | Assigned by Render; read by `web/serve.py` |
| `PORTFOLIO_TOS_URL` | Optional override for the portfolio research link |

The static portfolio link in [web/static/config.js](web/static/config.js) currently
points to `https://ricko-shaha.pages.dev/research#tos`. If needed, set
`PORTFOLIO_TOS_URL` in Render's Environment settings and deploy the change; it
overrides that static value through `/deployment-config.js`.

In the portfolio repository, `portfolio/assets/js/config.js` sets
`landslideAppUrl` to `https://landslide-rangamati.onrender.com/`. Changing that
destination also requires a separate Cloudflare Pages upload.

The default demo does not need Google Earth Engine credentials. Live coordinate
lookup depends on the external OpenTopoData elevation service; a failure there
does not imply the saved-model demo or precomputed district map is unavailable.

## Local production check

Use Python 3.13 and a virtual environment. From this app's root, on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-production.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe web/serve.py
```

Open **http://localhost:10000/**. The development command `python web/app.py`
still serves **http://127.0.0.1:5000/**. The portfolio's localhost app link expects
port `5000`; for testing that link with Waitress, set `$env:PORT = '5000'` before
starting `web/serve.py` in that PowerShell session.

## Verify a release

1. Open [the app](https://landslide-rangamati.onrender.com/) and confirm the changed UI.
2. Check [/healthz](https://landslide-rangamati.onrender.com/healthz) returns `{"status":"ok"}`.
3. Select a surveyed site, assess it, and confirm a score is displayed.
4. Open the district map and the analysis notebook.
5. For simulator changes, test touch-and-slide movement on a mobile viewport,
   lifting to stop, and desktop arrow keys/WASD.
6. Check the **About TOS** link returns to the portfolio research page.

## Troubleshooting and rollback

| Symptom | Check or action |
| --- | --- |
| First visit takes a while | A Free service may be waking from sleep; give it about a minute. |
| No deploy after a Git push | Verify the remote is `Ricko-Shaha/Landslide-Prediction`, branch is `main`, auto-deploy is enabled, and the commit has no skip phrase. |
| Render lists another person's repository | Check the GitHub owner and Render GitHub installation's repository access. Do not connect an unrelated repository. |
| Build fails | Read the first error in the build logs; verify Python 3.13, the pinned requirements, and all committed artifacts. |
| Health check/startup fails | Confirm `python web/serve.py`, the assigned `PORT`, and successful saved-model loading. |
| The map reports stale artifacts | Rebuild and commit matching model/grid artifacts together; do not hide the mismatch warning. |
| Live coordinate lookup fails | Check the external elevation response and study-area boundary; try a surveyed site to distinguish external lookup failure from a model failure. |
| Deploy is Live, but old files still appear | Confirm the deployed commit, reload with fresh asset versions, then use **Manual Deploy → Clear build cache & deploy** if the old files persist. |

On 22 September 2026, a deployment reported success while the old app text was
still served. Clearing the build cache and redeploying the same commit resolved
it; the live desktop and mobile pages were then verified. Use this recovery when
needed, not for every release. Render documents it under
[manual deployments](https://render.com/docs/deploys#manual-deploys).

For an ordinary regression, revert the bad source commit and push the revert to
`main`. Render will deploy the corrected history. This keeps GitHub and the live
version aligned.

## Free-plan behavior and costs

Keep this service on **Free**, with no paid database, disk, or purchased domain.
Render provides its HTTPS `onrender.com` address.

- Idle Free services sleep after 15 minutes; waking takes about a minute.
- A workspace shares 750 Free instance hours per month. Exhaustion suspends its
  Free web services until the next month.
- Local filesystem changes disappear on restart, redeploy, or sleep. The model
  and map survive because they are committed and shipped with each build.
- Bandwidth and build minutes have separate allowances. With a payment method,
  excess usage can incur charges. Without one, Render suspends affected services
  or disables new builds according to the exhausted allowance. Check Billing
  before changing payment or plan settings.

These provider limits can change; see [Render's Free service documentation](https://render.com/docs/free).

## This computer's source and GitHub copies

The editable app is `D:\Personal\Study\Prof Hunting\landslide-prediction`.
From its parent workspace, `python hosting/prepare.py` copies it to:

| Local checkout | Purpose |
| --- | --- |
| `.hosting/landslide-public` | Public repository connected to Render; push here to deploy |
| `.hosting/rangamati-app` | Private setup copy; pushing here does not deploy the live service |

The sync command does not commit, push, or deploy. A fresh clone of the public
repository can be edited and pushed directly without this mirror workflow.

The portfolio ships copies of `web/static/walk.js` and `web/static/walk_data.js`.
When changing the shared simulator, sync those copies and publish both apps.
