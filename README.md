# AequiScan Backend

Cross-planetary analog comparison dashboard API — comparing Earth, Moon, and Mars surface properties for mission planning and science discovery.

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
# Edit .env with your Earthdata Login credentials

# 4. Run the dev server
uvicorn app.main:app --reload --port 8000

# 5. Open API docs
# http://localhost:8000/docs
```

## Running Tests

```bash
# Unit + smoke tests (no credentials needed)
pytest tests/test_health.py tests/test_similarity.py -v

# Integration tests (requires EDL credentials + network)
pytest tests/test_analogs.py -v -m integration
```

## Project Structure

```
aequiscan-backend/
├── app/
│   ├── main.py                 # FastAPI app factory
│   ├── api/
│   │   └── v1/
│   │       ├── __init__.py     # v1 router aggregator
│   │       ├── health.py       # GET  /api/v1/health
│   │       ├── targets.py      # GET  /api/v1/targets/...
│   │       └── mineralogy.py   # POST /api/v1/mineralogy/compare
│   ├── core/
│   │   ├── config.py           # Pydantic Settings (env vars)
│   │   ├── constants.py        # Mineral taxonomy, preloaded targets
│   │   └── exceptions.py       # Custom error hierarchy
│   ├── schemas/
│   │   ├── common.py           # Shared Pydantic models
│   │   └── mineralogy.py       # Mineralogy request/response models
│   ├── services/
│   │   ├── earth_mineralogy.py # EMITL2BMIN extraction via earthaccess
│   │   ├── planetary_mineralogy.py  # PDS ODE queries (Mars/Moon)
│   │   └── similarity.py       # ASI computation engine
│   └── models/                 # Future: DB/ORM models
├── tests/
│   ├── conftest.py             # Shared fixtures (async test client)
│   ├── test_health.py          # Smoke tests (no credentials)
│   ├── test_similarity.py      # Pure-logic ASI unit tests
│   └── test_analogs.py         # Integration tests (needs EDL)
├── pyproject.toml
├── requirements.txt
├── .env.example
└── .gitignore
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/api/v1/targets/` | All preloaded targets by body |
| `GET` | `/api/v1/targets/{body}` | Targets for earth/moon/mars |
| `GET` | `/api/v1/targets/moon/artemis-iii` | 9 Artemis III candidate regions |
| `GET` | `/api/v1/mineralogy/mineral-classes` | 6 harmonised mineral classes |
| `POST` | `/api/v1/mineralogy/compare` | Earth vs planetary mineralogy comparison (preloaded ID or custom planetary lat/lon) |

### Comparison Request Examples

#### Option A: Using a preloaded target ID
```json
{
  "earth_lat": 38.45,
  "earth_lon": 91.42,
  "earth_radius_m": 2000.0,
  "planetary_target_id": "gale_crater"
}
```

#### Option B: Using custom arbitrary planetary coordinates (e.g., map click)
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

## Adding New Analytical Axes

Each of the 7 analytical dimensions follows the same pattern:

1. **Service** in `app/services/` — data extraction + processing logic
2. **Schema** in `app/schemas/` — Pydantic request/response models
3. **Router** in `app/api/v1/` — FastAPI endpoint definitions
4. Register the router in `app/api/v1/__init__.py`

## Architecture Notes

- **API-first**: The backend is fully independent from any frontend
- **Versioned**: All routes live under `/api/v1/` for backwards compatibility
- **Modular**: Each analytical axis is a self-contained service + router
- **Async**: PDS ODE queries use `httpx` async; earthaccess (sync) runs in threadpool
- **Testable**: Pure-logic services have zero I/O dependencies
