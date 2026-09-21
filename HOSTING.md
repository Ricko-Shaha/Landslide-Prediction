# Host the Rangamati app on Render

This is a separate Python web service. The portfolio runs on Cloudflare.

## Deploy

1. Create a free Render account and connect your GitHub account.
2. Choose **New → Blueprint**, then select the `Ricko-Shaha/Landslide-Prediction` repository.
3. Review `render.yaml`: one Python web service on the **Free** plan, with no database or disk.
4. Deploy. The build installs the pinned runtime and checks the shipped model, metrics,
   district map, notebook and API. The service starts using Waitress and Render's assigned port.
5. Copy the actual HTTPS `onrender.com` URL from the service dashboard. Names may receive a
   suffix if the requested address is already taken.

If using **New → Web Service** instead of a Blueprint, use:

| Setting | Value |
| --- | --- |
| Runtime | Python 3 |
| Root directory | Leave blank |
| Build command | `pip install -r requirements-production.txt && python -m unittest discover -s tests` |
| Start command | `python web/serve.py` |
| Instance type | Free |
| Health check path | `/healthz` |

`.python-version` selects Python 3.13. The prebuilt model and matching artifacts are committed;
deploying does not retrain the model or fetch thousands of terrain samples.

## Connect the portfolio

In Render's **Environment** settings, add `PORTFOLIO_TOS_URL` with the portfolio's full
HTTPS research-section address, ending in `/research.html#tos`, then redeploy. This overrides
`web/static/config.js`. If the environment variable is empty, the existing static configuration
and localhost defaults still work.

Set the portfolio's `landslideAppUrl` in `portfolio/assets/js/config.js` to this app's actual
HTTPS URL, then redeploy the portfolio. Both sites keep their own homepages and domains.

## Free-plan behavior

Render sleeps a free web service after 15 minutes without incoming traffic. The first subsequent
visit can take about a minute to start it. The free instance has 512 MB RAM; the app uses one
process with four threads and limits numerical-library threads. Its runtime does not require a
database or persistent disk. The saved model, district map and notebook ship with every build.

The HTTPS service supports **Use my location** in browsers. Live elevation lookup still depends
on the external OpenTopoData service; surveyed sites, factor classes and the district map can
work without a live elevation response.

Stay on the Free instance type. Review Render's bandwidth/build limits and billing settings
before adding a payment method. A paid domain is optional; Render supplies a free HTTPS address.

## Local production check

```sh
python -m pip install -r requirements-production.txt
python -m unittest discover -s tests -v
python web/serve.py
```

Open `http://localhost:10000/`. `PORT` changes the listening port; `HOST` changes the bind
address. The ordinary `python web/app.py` development preview remains available on port 5000.

References checked 21 September 2026:

- [Render free services](https://render.com/docs/free)
- [Render Blueprint configuration](https://render.com/docs/blueprint-spec)
- [Render Python versions](https://render.com/docs/python-version)
- [Flask's production deployment with Waitress](https://flask.palletsprojects.com/en/stable/deploying/waitress/)
