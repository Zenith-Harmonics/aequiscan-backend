

Cross-planetary analog comparison dashboard API and autonomous frontier discovery engine — comparing **Earth**, **Moon**, and **Mars** surface properties for mission planning and planetary science discovery.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Running with Docker](#running-with-docker)
- [Project Structure](#project-structure)
- [REST API Endpoints](#rest-api-endpoints)
- [Frontier Analog Discovery Engine (`discovery/`)](#frontier-analog-discovery-engine-discovery)
- [Running Tests](#running-tests)
- [Architecture & Extending](#architecture--extending)

---

## Quick Start

```bash
# 1. Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -e ".[dev]"
# or: pip install -r requirements.txt

# 3. Configure NASA Earthdata credentials
cp .env.example .env
# Edit .env with your NASA Earthdata Login (EDL) credentials

# 4. Run the FastAPI development server
uvicorn app.main:app --reload --port 8000

# 5. Open interactive Swagger API docs
# http://localhost:8000/docs
```

### Running with Docker

```bash
docker build -t aequiscan-backend .
docker run --env-file .env -p 8000:8000 aequiscan-backend
```

---

## Project Structure

```text
aequiscan-backend/
├── app/                                      # FastAPI REST API service
│   ├── main.py                               # FastAPI app factory
│   ├── api/v1/
│   │   ├── __init__.py                       # v1 router aggregator
│   │   ├── health.py                         # GET  /api/v1/health
│   │   ├── targets.py                        # GET  /api/v1/targets/...
│   │   └── mineralogy.py                     # POST /api/v1/mineralogy/compare
│   ├── core/
│   │   ├── config.py                         # Pydantic Settings (env vars)
│   │   ├── constants.py                      # Mineral taxonomy & preloaded targets
│   │   └── exceptions.py                     # Custom error hierarchy
│   ├── schemas/
│   │   ├── common.py                         # Shared Pydantic models
│   │   └── mineralogy.py                     # Mineralogy request/response models
│   ├── services/
│   │   ├── earth_mineralogy.py               # EMITL2BMIN extraction via earthaccess
│   │   ├── planetary_mineralogy.py           # PDS ODE queries (Mars/Moon)
│   │   └── similarity.py                     # Bray-Curtis ASI computation engine
│   └── models/                               # DB/ORM models
├── discovery/                                # Standalone Frontier Discovery Engine (decoupled from API)
│   ├── discover_novel_americas_analogs.py    # Self-contained 6-sector analog scanner CLI
│   ├── novel_north_america_frontier_jezero_crater_balanced_ranked.csv
│   ├── novel_north_america_frontier_jezero_crater_balanced_ranked.json
│   └── docs/
│       └── novel_analog_discovery.md         # Full technical & scientific discovery report
├── docs/                                     # API & model feature extraction docs
├── scripts/                                  # Backend utility scripts
├── tests/                                    # Unit & integration test suite
│   ├── conftest.py
│   ├── test_health.py
│   ├── test_similarity.py
│   └── test_analogs.py
├── Dockerfile
├── .dockerignore
├── pyproject.toml
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## REST API Endpoints

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Service health check |
| `GET` | `/api/v1/targets/` | All preloaded targets grouped by planetary body |
| `GET` | `/api/v1/targets/{body}` | Targets for `earth`, `moon`, or `mars` |
| `GET` | `/api/v1/targets/moon/artemis-iii` | 9 Artemis III lunar south pole candidate regions |
| `GET` | `/api/v1/mineralogy/mineral-classes` | 6 harmonised planetary mineral classes |
| `POST` | `/api/v1/mineralogy/compare` | Earth vs. planetary mineralogy comparison (preset ID or custom coordinates) |

### Comparison Request Examples

<details>
<summary><strong>Option A: Compare against a preloaded target ID</strong></summary>

```json
{
  "earth_lat": 38.45,
  "earth_lon": 91.42,
  "earth_radius_m": 2000.0,
  "planetary_target_id": "gale_crater"
}
```
</details>

<details>
<summary><strong>Option B: Compare against custom planetary coordinates (e.g., map click)</strong></summary>

```json
{
  "earth_lat": 38.45,
  "earth_lon": 91.42,
  "earth_radius_m": 2000.0,
  "planetary_body": "mars",
  "planetary_lat": 18.65,
  "planetary_lon": 226.20
}
```
</details>

---

## Frontier Analog Discovery Engine (`discovery/`)

Separate from the FastAPI web service, **[`discovery/discover_novel_americas_analogs.py`](discovery/discover_novel_americas_analogs.py)** is a standalone, two-stage coarse-to-fine batch discovery pipeline that systematically sweeps under-explored geological corridors across North and South America while geofence-excluding famous NASA/ESA analog testbeds (Mojave, Atacama Yungay, MDRS Utah, Devon Island, White Sands, etc.).

### Running a Discovery Sweep

```bash
# 1. Export NASA Earthdata credentials from .env
set -a; source .env; set +a

# 2. Sweep unexplored North American corridors against Mars Jezero Crater
python discovery/discover_novel_americas_analogs.py \
  --target-id jezero_crater \
  --corridor north_america_frontier \
  --ensure-weights \
  --output-dir discovery
```

### North America Frontier Discoveries (vs. Jezero Crater, Mars)

In a **229-tile continental sweep** targeting **Jezero Crater (`18.45° N, 77.45° E`)**, **76 validated tiles** (`14` coarse `0.45°` + `62` fine `0.10°` zoom tiles) in **`Vizcaino_Baja_Mesas` (Baja California Sur, Mexico)** passed all aridity (`P/PET <= 0.25`) and confidence thresholds, located **312.9 km to 533.1 km** from the nearest known NASA analog site (`Pinacate_Sonora_Mexico`):

| Rank | Resolution | Center Coordinates | Overall Score | Active-Sensor Parity | Mineralogy (`EMIT`/`S2`) | Subsurface (`S1 SAR`) | Atmospheric (`POWER`) | Distance from Known Site |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | `0.10°` Fine | `27.650° N, -113.950° W` | **31.89%** | **63.78%** | `59.97%` | `99.98%` | `32.66%` | `460.2 km` |
| **#2** | `0.10°` Fine | `27.750° N, -114.050° W` | **30.12%** | **60.24%** | `53.36%` | `97.98%` | `31.67%` | `450.1 km` |
| **#3** | `0.10°` Fine | `28.450° N, -112.850° W` | **29.93%** | **59.86%** | `51.01%` | `99.99%` | `31.54%` | `374.4 km` |
| **#4** | `0.10°` Fine | `28.200° N, -114.050° W` | **29.60%** | **59.20%** | `51.23%` | `98.12%` | `30.93%` | `400.5 km` |
| **#5** | `0.10°` Fine | `27.750° N, -113.950° W` | **29.35%** | **58.70%** | `58.51%` | `85.68%` | `31.95%` | `449.1 km` |

#### Discovery Documentation & Ranked Datasets
- **Full Technical & Scientific Report:** [`discovery/docs/novel_analog_discovery.md`](discovery/docs/novel_analog_discovery.md)
- **Standalone Discovery Script:** [`discovery/discover_novel_americas_analogs.py`](discovery/discover_novel_americas_analogs.py)
- **Ranked Results Dataset:** [`CSV`](discovery/novel_north_america_frontier_jezero_crater_balanced_ranked.csv) | [`JSON`](discovery/novel_north_america_frontier_jezero_crater_balanced_ranked.json)

---

## Running Tests

```bash
# Unit + smoke tests (no credentials needed)
pytest tests/test_health.py tests/test_similarity.py -v

# Integration tests (requires NASA EDL credentials + network)
pytest tests/test_analogs.py -v -m integration
```

---

## Architecture & Extending

### Adding New Analytical Axes to the API

Each analytical dimension follows the same modular pattern:
1. **Service** in `app/services/` — data extraction & processing logic
2. **Schema** in `app/schemas/` — Pydantic request/response models
3. **Router** in `app/api/v1/` — FastAPI endpoint definitions
4. **Registration** — mount the router in `app/api/v1/__init__.py`

### Core Design Principles

- **API-First & Decoupled**: The FastAPI backend (`app/`) is completely independent from both frontends and the offline batch discovery engine (`discovery/`).
- **Versioned**: All REST routes live under `/api/v1/` for backwards compatibility.
- **Async I/O**: NASA PDS ODE queries use `httpx` async; synchronous `earthaccess` granule streaming runs in a threadpool.
- **Testable**: Pure-logic similarity and ASI engines have zero network/IO dependencies.
