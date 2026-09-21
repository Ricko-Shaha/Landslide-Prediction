# Data sources and attribution

Everything this project reads, where it came from, and what its licence asks for.

---

## 1. The landslide inventory (`data/landslide_inventory.csv`)

**This dataset is not mine, and it did not originate with the repository I took it from.** The
lineage runs through three hands, and all three are credited below.

### The origin: the landslide inventory

> **Rabby, Y. W. and Li, Y. (2020).** Landslide Inventory (2001-2017) of Chittagong Hilly Areas,
> Bangladesh. *Data*, 5(1), 4.
> DOI [10.3390/data5010004](https://doi.org/10.3390/data5010004) · **CC BY**, gold open access

Landslides mapped from Google Earth imagery, field mapping and a literature search for the
Chittagong Hilly Areas, which had no official landslide inventory. This is the geographic
fieldwork the whole chain rests on.

The susceptibility study that shaped the factor set for Rangamati specifically:

> **Rabby, Y. W., Hossain, M. B. and Abedin, J. (2021).** Landslide susceptibility mapping in
> three Upazilas of Rangamati hill district Bangladesh: application and comparison of GIS-based
> machine learning methods. *Geocarto International*.
> DOI [10.1080/10106049.2020.1864026](https://doi.org/10.1080/10106049.2020.1864026)

Yasin Wahid Rabby was at the University of Tennessee, Knoxville; Md Belal Hossain at Oklahoma
State University; Joynal Abedin at the University of Dhaka; Yingkui Li at the University of
Tennessee, Knoxville.

### The intermediate: the CSV I actually read

> **Inan, M. S. K. and Rahman, I. (2023).** Explainable AI Integrated Feature Selection for
> Landslide Susceptibility Mapping Using TreeSHAP. *SN Computer Science*.
> DOI [10.1007/s42979-023-01960-5](https://doi.org/10.1007/s42979-023-01960-5)
>
> Code and data: <https://github.com/scumechanics/Landslide-Susceptibility-Prediction-Using-Machine-Learning-Algorithms>
> File `Dataset/Dis2.csv` · Copyright (c) 2021 **Muhammad Sakib Khan Inan** · Licence: **MIT**

That paper draws its Rangamati landslide data from Rabby et al. above. The file in this
repository is renamed from `Dis2.csv` to `landslide_inventory.csv` and is otherwise byte-for-byte
identical to theirs:

```
sha256(header + 392 rows) = 8d74ec433342191af8d8...   upstream Dis2.csv
sha256(header + 392 rows) = 8d74ec433342191af8d8...   data/landslide_inventory.csv
```

392 rows, 16 columns, balanced 196 landslide / 196 non-landslide. Every column is a reclassified
class rating rather than a physical measurement; `src/reclassify.py` explains how much of that
scheme is recoverable and how much is reconstructed.

### The study area, and how it is known

The study area is **Rangamati District, Bangladesh**. That is not an assumption: the Geocarto
paper states its own area as "part of areas in Rangamati District, Bangladesh", and the boundary
polygon carried by the first version of this project independently encloses 5,782 km2 against
Rangamati's ~6,116 km2 of record.

What is *not* stated anywhere upstream is the row-level derivation: neither the GitHub repository
nor its paper documents which inventory records became which rows of `Dis2.csv`, and the rows
carry no coordinates. So the district is known; the exact sampling that produced these 392 rows
is not reconstructable from the published material.

### MIT licence notice

The upstream project is MIT licensed, which permits reuse with attribution and requires the notice
to travel with the work. Its licence is reproduced in [`LICENSE-DATASET`](LICENSE-DATASET).

---

## 2. Elevation, slope and aspect

| | |
|---|---|
| Dataset | SRTM 1 Arc-Second Global (`USGS/SRTMGL1_003`), 30 m |
| Source | USGS EROS (2017), DOI [10.5066/F7PR7TFT](https://doi.org/10.5066/F7PR7TFT) |
| Served by | OpenTopoData, <https://www.opentopodata.org/datasets/srtm/> |
| API endpoint | `https://api.opentopodata.org/v1/srtm30m?locations=LAT,LON` (returns 400 without parameters) |
| Licence | SRTM is public domain (U.S. Government work). OpenTopoData's public API is free, rate limited to 1 call/second and 1000 calls/day |

Slope and aspect are not downloaded; they are computed from a 3×3 elevation window by Horn's
method in `src/terrain.py`, which is the same algorithm GDAL and Earth Engine use.

The public endpoint is rate limited, which is why the district map is built once and cached rather
than scored on demand.

---

## 3. Rainfall

| | |
|---|---|
| Dataset | NASA POWER climatology, parameter `PRECTOTCORR` (corrected total precipitation) |
| Community | Agroclimatology (`AG`) |
| Source | <https://power.larc.nasa.gov/> |
| API endpoint | `https://power.larc.nasa.gov/api/temporal/climatology/point?parameters=PRECTOTCORR&community=AG&longitude=LON&latitude=LAT&format=JSON` |
| Docs | <https://power.larc.nasa.gov/docs/services/api/temporal/climatology/> |
| Licence | Freely available; NASA asks that the POWER Project be acknowledged |

> These data were obtained from the NASA Langley Research Center POWER Project, funded through the
> NASA Earth Science Directorate Applied Science Program.

Cached to `data/rainfall_grid.json` on a ~11 km grid: annual precipitation is a smooth regional
field, so it does not need a request per map cell. Rebuild with `python src/rainfall.py --rebuild`.

**Not used, but kept as a documented fallback:** Open-Meteo's ERA5 archive,
<https://open-meteo.com/en/docs/historical-weather-api>. It needs one request per year per point
and was returning HTTP 500s and timeouts when this was built.

---

## 4. Map imagery

The web page draws two tile layers over Leaflet.

**Satellite imagery — Esri World Imagery**

| | |
|---|---|
| Service | <https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer> |
| Tile template | `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}` |
| Item page | <https://www.arcgis.com/home/item.html?id=10df2279f9684e4a9f6a7f08febac2a9> |
| Terms | <https://www.esri.com/en-us/legal/terms/full-master-agreement> |

Required attribution, taken from the service's own `copyrightText` field rather than transcribed:

> Source: Esri, Vantor, Earthstar Geographics, and the GIS User Community

**Place labels — CARTO Positron (labels only)**

| | |
|---|---|
| Tile template | `https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}.png` |
| Attribution | © [CARTO](https://carto.com/attributions), data © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors |
| Licence | CARTO basemaps are free for non-commercial use; OpenStreetMap data is ODbL |

**Mapping library — Leaflet 1.9.4**, <https://leafletjs.com/>, BSD-2-Clause.

---

## 5. The study-area boundary (`data/study_area.json`)

Decoded from `coordinates_3`, an undocumented file of 11,774 `lon,lat` pairs carried by the first
version of this project. It forms a single closed ring enclosing 5,782 km², which identifies it as
the Rangamati district boundary. Reduced to 1,401 vertices for use as a mask and map outline; see
`src/study_area.py`.

Its own upstream provenance — which map it was traced from — is not recorded.

---

## 6. Model selection metric

> Shaha, R., Talukder, D., Iqbal, M. A. and Haque, M. M. (2021). TOS: A Relative Metric Approach
> for Model Selection in Machine Learning Solutions. *2021 IEEE International Conference on
> Robotics, Automation, Artificial-Intelligence and Internet-of-Things (RAAICON)*, 26–31.
> DOI [10.1109/raaicon54709.2021.9929722](https://doi.org/10.1109/raaicon54709.2021.9929722)

---

## Summary of what is original to this project

| Original | Reused |
|---|---|
| The restructure, the analysis and every finding in the README | The landslide inventory (MIT, see §1) |
| TOS-based model selection and the IIA demonstration | SRTM elevation, NASA POWER rainfall |
| The reclassification reconstruction (`src/reclassify.py`) | Esri and CARTO map tiles |
| Live coordinate lookup without credentials | Leaflet |
| The district susceptibility map | |
| The web application and the notebook | |
