# Landslide susceptibility, Rangamati Hill Tracts

A restructured version of my undergraduate thesis project at Chittagong University of Engineering
& Technology: a model that ranks hillslope sites by landslide susceptibility from eight terrain
and vegetation factors, with a web application that makes the model, its evidence and its limits
inspectable in one page.

```
python -m pip install -r requirements.txt
python src/train.py                  # fits, selects by TOS, writes models/ and reports/
python src/susceptibility_map.py     # scores every cell of the district (a few minutes, once)
python web/app.py                    # http://127.0.0.1:5000
```

Nothing here needs credentials. The original application would not start without an authenticated
Google Earth Engine account; this one runs immediately, and live coordinate lookup works out of
the box because elevation comes from SRTM through a public endpoint that needs no account.

Five ways to ask the model a question:

| Tab | What it does |
|---|---|
| Surveyed site | scores a real row from the inventory |
| Factor classes | steps each factor through the classes it actually takes |
| Live coordinate | use your location, click the map, or enter coordinates to measure the terrain there |
| District map | every cell of Rangamati scored and painted on one scale |
| Walk it | the same grid as a solid, at eye level, with a figure you steer across it |

## Interface and separate hosting

For the live Render setup, deployment commands, and troubleshooting, see [DEPLOY.md](DEPLOY.md). `render.yaml` selects
the Free plan, `requirements-production.txt` pins the model runtime, and `web/serve.py` runs a
production server. The shipped model and map are used directly, with no training during deploy.

Rangamati has its own terrain mark, compact workspace, forest-green controls, Fraunces headings,
and theme preference. Warm neutrals, restrained borders, and readable body text give it a related
style to the portfolio. The app and notebook have their own navigation and favicon. Mobile mode
tabs scroll horizontally only. Map colors retain the original five susceptibility bands.

With the portfolio running at `http://127.0.0.1:4173/` and this app at
`http://127.0.0.1:5000/`, **Open the landslide app** in the portfolio's selected work and project
notes opens the separate app directly. Use the same hostname for both local previews.

For public hosting, set `landslideAppUrl` in `../portfolio/assets/js/config.js` to the deployed
app's HTTPS address. There is no intermediate portfolio page or shared routing. Until a URL is
set, the public portfolio omits the app link while retaining its project notes and source links.
The Flask application runs independently and is deployed separately from the portfolio.

For the **About TOS** link in Model evidence, set `tosReferenceUrl` in `web/static/config.js`
to the portfolio's full research-section URL, ending in `/research.html#tos`. Local previews
link to port 4173 automatically. On public hosting the link stays hidden until configured.
On Render, setting the `PORTFOLIO_TOS_URL` environment variable overrides this static setting.

**Use my location** in Live coordinate asks for browser location permission only when clicked.
It fills the coordinates, moves the map marker, and assesses the point. Locations outside
Rangamati keep the existing outside-study-area response and receive no score. If access is
denied or unavailable, manual coordinates and map selection remain available. Browser location
access requires HTTPS on public hosting; localhost previews also support it.

Interface files are `web/static/theme.css`, `app.css`, and `theme.js`. The standalone notebook
embeds `theme.css`, `notebook.css`, the theme script, and the favicon when regenerated with
`python src/nb_to_html.py`. Its analysis and model outputs are unchanged.

---

## What the model is

| | |
|---|---|
| Task | Binary classification: is this site landslide-prone? |
| Inventory | 392 sites, balanced 196 / 196. **Not collected here**: reused under MIT, see [DATA_SOURCES.md](DATA_SOURCES.md) |
| Predictors | ELEVATION, SLOPE, ASPECT, TWI, SPI, NDVI, RAINFALL, LANDUSE |
| Selected model | Gradient boosting |
| Selection rule | TOS, a relative score over the candidate pool (see finding 2) |
| Out-of-fold ROC AUC | 0.965 |
| Out-of-fold accuracy | 0.898 |
| Out-of-fold recall | 0.888 |

Evaluation is 5-fold stratified cross-validation repeated 6 times, 30 fits per model.

---

## Three findings from the restructure

These are the reasons the project needed more than a tidy-up. Each is reproducible from the code
in this repository.

### 1. The deployed application was scoring inputs the model had never seen

This is the serious one.

The training columns are **not physical measurements**. They are reclassified factor weights: each
terrain factor was binned into classes and each class assigned a score. Across the whole inventory
`ELEVATION` takes **4** distinct integers between 11 and 50, `ASPECT` takes **8** between 2 and 17,
and `NDVI` ranges **0 to 36** when a true NDVI lies between −1 and 1.

The original `app.py` sampled Google Earth Engine and fed the results straight in: elevation in
metres (213), slope in degrees (0.94), NDVI as a ratio (0.52). scikit-learn raises nothing when
asked to score out-of-range inputs, so the page returned a confident percentage computed from
numbers the model could not interpret.

The conversion table from raw measurement to class weight was never recorded. Rather than leave
the project's headline feature broken, `src/reclassify.py` reconstructs it, and says so wherever
it surfaces. Three things turned out to be recoverable from the data itself:

- **How many classes each factor has.** Every factor's distinct ratings sum to exactly 100.
  `SLOPE` is 6 + 13 + 22 + 28 + 31; `ELEVATION` is 11 + 13 + 26 + 50. That is a normalisation,
  which confirms these are per-factor class ratings rather than measurements. ASPECT's observed
  ratings sum to 89, so one class of weight 11 simply never occurs among the 392 sites, which is
  consistent with the usual flat-plus-eight-octants scheme.
- **Their order.** Within every factor the rating rises with the observed landslide rate:
  SLOPE runs 0.12, 0.62, 1.00, 1.00, 1.00 and TWI runs 0.00, 0.12, 0.20, 0.46, 0.96. Higher
  rating means more susceptible class.
- **Where the study area is.** The undocumented `coordinates_3` file, 11,774 bare `lon,lat`
  pairs, is a single closed ring enclosing 5,782 km2. That is the Rangamati district boundary,
  not a list of sample sites, so there is no hidden coordinate-to-label table in the repository.

What is *not* recoverable is where one class stops and the next begins, so the cut points are the
standard physical breaks for slope and elevation. That is the assumption to attack first.

The live tab is now usable rather than labelled unusable, and it states its own limits: three of
the eight predictors (elevation, slope, aspect, carrying **0.819** of the model's feature
importance) are measured per click, and the other five are held at fixed classes and tagged as
held, so any difference between two points is attributable to terrain. Clicking outside the
district returns no score at all, only a refusal: the model has no evidence about ground it has
never seen, and a number with a caveat under it gets quoted without the caveat.

### 2. Accuracy could not choose the model, so the choice was made a different way

The thesis compared nine models on one 75/25 split with the default seed and reported Random
Forest best at 0.88. With 392 rows, the test set is 98 samples, so one percentage point is one
sample.

Repeating the split 30 times, every candidate above the baseline sits within roughly one standard
deviation of the others, and the top three are separated by three thousandths:

| Model | Accuracy | Missed (FNR) | TOS | ROC AUC |
|---|---|---|---|---|
| Gradient boosting | 0.893 ± 0.026 | 0.114 | **+0.877** | 0.966 |
| Logistic regression | 0.890 ± 0.035 | 0.134 | +0.597 | 0.952 |
| Random forest | 0.889 ± 0.029 | 0.133 | +0.561 | 0.964 |
| Support vector machine | 0.887 ± 0.030 | 0.159 | −0.158 | 0.949 |
| AdaBoost | 0.878 ± 0.034 | 0.140 | −0.208 | 0.951 |
| k-nearest neighbours | 0.882 ± 0.031 | 0.170 | −0.542 | 0.931 |
| Decision tree | 0.869 ± 0.036 | 0.163 | −0.855 | 0.931 |
| Baseline (most frequent) | 0.497 ± 0.003 | — | not a candidate | 0.500 |

Two things are wrong with reading that accuracy column and taking the largest number. The
differences are inside the noise, so the pick is close to a coin toss. And accuracy prices a
missed landslide and a false alarm identically, which in susceptibility mapping is simply false.

So the model is selected by **TOS**, the relative metric from my RAAICON 2021 paper:

```
TOS_i = ½ ( tanh(z_acc,i) − tanh(z_err,i) )
```

with the z-scores taken across the candidate pool. It forces both axes to be named. The accuracy
axis is cross-validated accuracy; the error axis is the **false negative rate**, because the
expensive mistake here is the landslide that was not predicted. Gradient boosting wins, and ROC
AUC would have chosen it too, but TOS opens a real margin (+0.877 against +0.597) between
candidates that accuracy left three thousandths apart, and it does so for a stated reason.

It also has a limitation, and this project happens to demonstrate it. TOS is computed over the
pool, so it is **not independent of irrelevant alternatives**. Put the baseline back into the
pool, a model nobody would deploy, and the winner holds but Support vector machine and AdaBoost
swap ranks 4 and 5. `reports/metrics.json` records this under `selection.iia_demonstration`, and
`selection.pool_sensitivity` confirms the winner survives dropping the weakest candidates. A
relative score is only meaningful next to the pool it was computed over, so the pool is always
reported with the ranking.

Two smaller results fall out of the same table. **Plain logistic regression matches the
ensembles**, which says the signal in these eight factors is close to linearly separable and an
interpretable model costs nothing here. And a baseline row was added because the original had
none; it is the only way to see that 0.88 is a real result rather than a property of the data.

### 3. The probability is not a probability of failure

The inventory is balanced 196 / 196 by construction. Real terrain is nowhere near 50 percent
landslide, so the number the model returns is calibrated to a sampling design, not to the world.
It ranks sites against each other; it overstates absolute risk. Reading it as a chance of failure
would be a serious error in the one setting where this work could matter.

The application states this next to every number it prints, and `reports/metrics.json` records it
under `prevalence_note`.

---

## Smaller repairs

- **`except:` that fabricated data.** TWI was computed as `log(fac * 1000 / tan(slope))` inside a
  bare `except:` that substituted the constant `10` on failure. On flat ground `tan(slope)`
  approaches zero, so that branch fired routinely and fed the model an invented value that looked
  like a real one. Slope is now floored at 0.1°, the standard treatment, and nothing is
  substituted silently.
- **Positional feature vectors.** The model was pickled bare and the app rebuilt the input as a
  positional list, so reordering the training columns would have produced silent nonsense. The
  saved artefact now carries its feature names and inference assembles the row **by name**,
  raising on a mismatch.
- **Hardcoded rainfall.** `rainfall = 28` was a constant in the request handler. It is now an
  explicit input with a stated default.
- **Unloadable model.** The committed `model.pkl` cannot be opened by any current scikit-learn
  (`No module named 'sklearn.ensemble.forest'`). The model is now rebuilt from source by
  `src/train.py`, so the artefact is never the only copy of the work.
- **A virtualenv in version control.** The original tracked `project/Lib/site-packages`, roughly
  7,900 files of third-party code. Dependencies belong in `requirements.txt`.
- **Untitled notebooks.** `Untitled.ipynb`, `Untitled1.ipynb`, `Test.ipynb` and `Test 2.ipynb` gave
  no clue which one produced the shipped model. It was `Test.ipynb`.

---

## The district map

A single coordinate answers "is this spot dangerous". The map answers "where in Rangamati is the
danger", which is the question anyone planning a road, a settlement or an evacuation route is
actually asking, and it is the artefact the thesis was always aiming at.

`src/susceptibility_map.py` lays a grid over the district, measures elevation, slope and aspect at
every cell from SRTM, converts each to class ratings and scores it. It is built once and cached,
because each cell needs its own elevation window from a shared public endpoint.

### Walking it

On phones, touch anywhere on the terrain and slide to steer. Keep holding to continue moving,
slide back to the starting point or lift to stop. A small floating indicator follows the gesture;
there are no on-screen arrow buttons. Longer drags move faster, and movement follows the direction
on screen. Desktop arrow keys and WASD still work; mouse dragging is also supported. Interrupted
touches, changing tabs, and leaving the view clear the controls so movement cannot get stuck.
The portfolio uses the same simulator and touch behavior.

The **Walk it** tab draws the same grid as a solid: every cell inside the boundary at the
elevation SRTM reports for it, coloured by the probability the selected model assigns it, with a
figure you steer across it. The rule it enforces is the model's own classification and not
something invented for the demonstration. Ground scored 0.60 to 0.80 (High) shakes underfoot;
ground scored 0.80 and over (Very high) gives way, and each collapse leaves a scar in the mesh at
the cell that failed. Roughly a quarter of the district scores that badly, which is the point: the
safe routes are the valleys the model already likes, and you find them by walking.

Vertical scale is exaggerated 7x. The district is 210 km from end to end and 895 m at its highest,
so at true scale there is nothing to walk over. 7x was chosen by measuring the grid rather than by
eye: it puts the 99th percentile step between neighbouring cells at about 62 degrees. Distances
across the ground, and every number in the readout, are real.

The other three tabs drive it. Click a point on the live coordinate map and the surveyor walks to
it. Choose a surveyed site and he goes to ground in the same elevation and slope classes, because
inventory rows carry factor ratings and no coordinates -- there is no single place any of them
came from, and the panel says so rather than implying otherwise. A point outside the district is
refused, in the same terms the live lookup refuses it.

`python src/walk_export.py` regenerates `web/static/walk_data.js` from the susceptibility grid.
Run it whenever that grid is rebuilt; the 3D view reads nothing else.

Three details that took a second pass to get right:

- **Nothing is painted outside the boundary.** A cell is kept when its centre falls inside the
  district, so a square drawn around that centre hangs half a cell over the border. The surface is
  now clipped to the boundary polygon instead.
- **It is a field, not a mosaic.** Drawn as one image with bilinear smoothing between measured
  cell centres rather than as thousands of squares. The smoothing is presentation, not resolution;
  the true cell spacing is printed beside the map.
- **The edge needs data beyond the edge.** Cells one step outside the boundary are measured too,
  purely so the interpolation has values to work with right up to the border and does not fade
  into a pale fringe. They are flagged, and the summary statistics count only interior cells.

## Two things that only appeared once the pieces were connected

**A proxy that correlates is not a measurement.** TWI and SPI are defined through flow
accumulation, a catchment property no point query can supply. An early version derived them from
slope, reasoning that water collects on gentle ground. The reasoning is fine and the result was
wrong: high TWI marks convergent *hollows* on a hillslope, and slope alone cannot tell a hollow
from a flat plateau, so every flat cell was handed the inventory's most landslide-prone TWI class
and lake shores came back at 0.91. TWI and SPI carry 0.048 of the model's importance between them,
so the invention bought nothing. They are held fixed and flagged instead.

**Quantile matching needs both distributions to describe the same population.** The first attempt
at the class cut points placed them at quantiles of the district's own terrain, so each class
covered an equal share of the ground. But the 392-row inventory is balanced 196/196 by design and
is therefore not a random sample of the district; its slope class 22 has a 1.00 observed landslide
rate. Mapping the district's median hillside into that class declared half of Rangamati a
certainty, and every coordinate returned above 0.96. The physical breaks fixed it: a lake shore
now reads 0.06, the valley town 0.12, low hills 0.74 to 0.80, and the Sajek ridge at 291 m and
16 degrees reads 0.995.

---

## Layout

```
landslide-prediction/
├── data/landslide_inventory.csv     the 392-site inventory (upstream Dis2.csv, MIT)
├── data/study_area.json             the district boundary, decoded from `coordinates_3`
├── src/
│   ├── train.py                     fit, cross-validate, select by TOS, write metrics
│   ├── tos.py                       the relative selection metric, with its IIA check
│   ├── terrain.py                   slope and aspect from SRTM, no credentials needed
│   ├── study_area.py                the boundary polygon and the terrain inside it
│   ├── reclassify.py                raw measurement to class rating, reconstructed
│   ├── susceptibility_map.py        scores every cell of the district
│   └── predict.py                   inference by feature name, with domain checking
├── models/landslide_model.joblib    estimator + feature names + provenance
├── reports/metrics.json             every number the web page displays
├── src/walk_export.py               packs the district grid for the browser
├── web/
│   ├── app.py                       Flask API and page
│   ├── static/walk.js               the district as a solid, in three.js
│   ├── static/walk_data.js          that grid, quantised to a byte a field (generated)
│   └── templates/index.html         the interface
└── requirements.txt
```

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/schema` | the observed domain of each predictor, read from the training data |
| `GET /api/sites` | 60 real inventory rows, outcome included |
| `GET /api/metrics` | the full evaluation report |
| `POST /api/predict` | `{"features": {...}}` → probability, band, and out-of-domain warnings |
| `POST /api/sample` | `{"lat":…, "lon":…}` → measured terrain, class ratings, provenance, score. `422` and no score outside the district |
| `GET /api/boundary` | the district ring, its bounding box and area |
| `GET /api/susceptibility` | the precomputed district map, `404` until it is built |
| `POST /api/sample_ee` | the original Earth Engine path, kept so the defect can be reproduced rather than only described |

## Earth Engine

Live mode does not need it. `POST /api/sample_ee` is kept only so that the original defect can be
reproduced rather than just described: it returns raw physical readings in physical units, the
exact thing the old application fed to a model trained on class ratings, with every out-of-domain
value listed beside them.

```
python -m pip install earthengine-api
earthengine authenticate
```

## Data sources

Elevation SRTM (`USGS/SRTMGL1_003`) · land cover MODIS (`MODIS/006/MCD12Q1`) · NDVI Landsat 8
(`LANDSAT/LC08/C01/T1_TOA`) · flow accumulation MERIT Hydro (`MERIT/Hydro/v1_0_1`).

## Attribution

**The landslide inventory is not mine, and it did not originate with the repository I took it
from.** Three parties deserve credit, in order:

1. **The fieldwork.** Rabby, Y. W. and Li, Y. (2020). *Landslide Inventory (2001-2017) of
   Chittagong Hilly Areas, Bangladesh.* Data 5(1), 4. CC BY.
   [10.3390/data5010004](https://doi.org/10.3390/data5010004) — landslides mapped from Google
   Earth imagery, field mapping and a literature search, for an area that had no official
   inventory.
2. **The susceptibility study for Rangamati.** Rabby, Y. W., Hossain, M. B. and Abedin, J.
   (2021). *Landslide susceptibility mapping in three Upazilas of Rangamati hill district
   Bangladesh.* Geocarto International.
   [10.1080/10106049.2020.1864026](https://doi.org/10.1080/10106049.2020.1864026)
3. **The CSV I actually read.** Inan, M. S. K. and Rahman, I. (2023). *Explainable AI Integrated
   Feature Selection for Landslide Susceptibility Mapping Using TreeSHAP.* SN Computer Science.
   [10.1007/s42979-023-01960-5](https://doi.org/10.1007/s42979-023-01960-5) — published as
   `Dataset/Dis2.csv` in
   [scumechanics/Landslide-Susceptibility-Prediction-Using-Machine-Learning-Algorithms](https://github.com/scumechanics/Landslide-Susceptibility-Prediction-Using-Machine-Learning-Algorithms)
   under MIT, which the copy here matches byte for byte. Licence notice in
   [`LICENSE-DATASET`](LICENSE-DATASET).

The study area is **Rangamati District, Bangladesh**, stated by the Geocarto paper and
corroborated independently by the boundary polygon in this repository, which encloses 5,782 km²
against Rangamati's ~6,116 km² of record. What is not documented anywhere upstream is the
row-level derivation: no published source says which inventory records became which of the 392
rows, and the rows carry no coordinates.

Elevation is SRTM 30 m via [OpenTopoData](https://www.opentopodata.org/datasets/srtm/); rainfall is
[NASA POWER](https://power.larc.nasa.gov/) climatology; map imagery is Esri World Imagery with
CARTO labels. Full links, licences and required attributions are in
[DATA_SOURCES.md](DATA_SOURCES.md).

## First version of this project

<https://github.com/Ricko-Shaha/Landslide-Prediction>
