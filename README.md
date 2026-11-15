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

- ### 3. `discovery/docs/novel_analog_discovery.md`


# AequiScan Novel Terrestrial Analog Discovery Engine: Technical & Scientific Documentation

## 1. System Overview & Purpose

The **AequiScan Standalone Discovery Engine** (`discovery/discover_novel_americas_analogs.py`) is an automated geospatial and multi-modal deep learning pipeline built to discover **unexplored, non-famous planetary analog sites** on Earth for Mars and Moon mission preparation.

Rather than re-evaluating well-known space agency testbeds (such as the Mojave Desert, Atacama Yungay, Mars Desert Research Station in Utah, or Devon Island), the system applies **spatial exclusion geofencing** around established analog sites and systematically sweeps broad, under-studied geological corridors across North and South America using live orbital telemetry from NASA, ESA, Microsoft Planetary Computer, and ISRIC.

---

## 2. System Architecture & Execution Pipeline

The standalone script executes in **six sequential stages**:

### Stage A: Geofenced Grid Generation & Exclusion Masking
1. **Corridor Tiling:** Divides candidate continental regions (`Owyhee_Alvord_Basin`, `Bolson_de_Mapimi_Mexico`, `Vizcaino_Baja_Mesas`, `Sevier_West_Desert_UT`, `Mackenzie_Peel_Plateau`, and South American frontier corridors) into uniform latitude/longitude grid tiles (`--coarse-deg`, default `0.45°`).
2. **Land Masking:** Uses `global_land_mask` to discard open-ocean cells.
3. **Novelty Geofence Filter (`check_novelty_geofence`):** Calculates great-circle Haversine distances against 12 established NASA/ESA analog sites (exclusion radius `35–85 km`) and discards any tile overlapping known testbeds, logging the exact distance (`distance_to_known_km`) to the nearest legacy testbed.

### Stage B: Tier-0 Rapid Climate & Aridity Pre-Filter
Before downloading heavy satellite rasters, each candidate tile queries the **NASA POWER API** (`T2M`, `T2M_MAX`, `T2M_MIN`, `PRECTOTCORR`, `ALLSKY_SFC_SW_DWN`) to compute the **UNESCO Hargreaves-Samani Aridity Index ($P/\text{PET}$)**:
- Tiles wetter than `--max-aridity` (default `0.25`) are immediately skipped (`[PRE-SKIP]`) in ~1.5 seconds and recorded in the crash-safe `.jsonl` checkpoint log so vegetated or humid terrain never consumes raster bandwidth.

### Stage C: Live Multi-Sensor Orbital Data Ingestion
For tiles that pass the aridity threshold, `UnifiedPlanetaryDataProvider` streams and co-registers multi-sensor observations onto a uniform local **UTM (`EPSG:326xx`/`327xx`) `120 × 120` grid**:

| Source Platform | Sensor / Product | Extracted Scientific Telemetry |
| :--- | :--- | :--- |
| **Microsoft Planetary Computer STAC** | `cop-dem-glo-30` | Copernicus 30m DEM (elevation, slope angles, RMS roughness, hypsometric relief) |
| **Microsoft Planetary Computer STAC** | `sentinel-2-l2a` | 10 multispectral bands (`B02–B12`, `<25%` cloud cover) for true-color RGB & bolometric albedo |
| **Microsoft Planetary Computer STAC** | `sentinel-1-rtc` | C-band SAR (`VV`, `VH`) denoised with a $5\times5$ **Lee speckle filter** for backscatter (dB) & CPR |
| **Microsoft Planetary Computer STAC** | `landsat-c2-l2` (`lwir11`) | Thermal infrared brightness temperature converted to Kelvin |
| **NASA Earthdata (`earthaccess`)** | `EMITL2BMIN` (v001) | 60m ISS hyperspectral mineral detections (`group_1/2_mineral_id`, `band_depth`) across 6 mineral families |
| **NASA POWER API** | Daily Meteorology & Flux | Surface pressure (Pa), 10m wind speed (m/s), UVA/UVB flux ($\text{W/m}^2$), dust AOD, soil wetness ($a_w$) |
| **ISRIC SoilGrids v2.0** | `phh2o` & `soc` (0–5 cm) | Topsoil pH and Soil Organic Carbon (`g/kg`) |
| **NASA PDS ODE REST API** | MRO (`CRISM`, `HiRISE`, `MCS`), ODY (`THEMIS`), LRO (`LROC`, `DLRE`, `Mini-RF`) | Live orbital browse imagery, incidence angles, brightness temperatures, pressure, and radar CPR |

### Stage D: Deep Neural Feature Extraction & Cross-Domain Alignment
- **`RemoteCLIP` (`ViT-L-14`):** Encodes true-color Earth and Mars/Moon orbital imagery into L2-normalized **768-D** visual embeddings.
- **`ConvNeXtV2-Base` (`BigEarthNet v2.0` S1 & S2):** Extracts **1024-D** radar backscatter and **1024-D** 10-band multispectral embeddings for Earth tiles.
- **`LatentSpaceAligner` (`Ridge` / `CCA`):** Optionally projects 1024-D/2048-D Earth Sentinel embeddings into the 768-D RemoteCLIP planetary latent space.

### Stage E: Six-Sector Scientific Evaluation (`PlanetaryAnalogPipeline`)
Every Earth tile is compared against the target planetary bundle across **six scientific domains**, using four mathematical similarity kernels:
1. **Gaussian Similarity** $\exp\left(-\frac{1}{2}\left(\frac{x_E - x_T}{\sigma_{\text{tol}}}\right)^2\right)$ for scalar physical parameters (roughness, relief, albedo, temperature, wind, log-pressure, UV flux, water activity, pH).
2. **1D Wasserstein (Earth-Mover's) Distance** $\exp(-W_1(P, Q)/s)$ for spatial raster distributions (DEM slope angles, radar CPR).
3. **Bray-Curtis Analog Similarity Index (ASI)** $1 - \frac{\sum w_i |e_i - p_i|}{\sum w_i (e_i + p_i)}$ across the 6 harmonized mineral classes (*mafic olivine/pyroxene, plagioclase/anorthosite, phyllosilicates/clays, sulfates/evaporites, carbonates, iron oxides/hydroxides*).
4. **Shifted Cosine Similarity** $\frac{1}{2}(\cos\theta + 1)$ for 768-D neural embeddings.

Sector scores are combined via mission-profile weights (`balanced`: `20%` Geomorphology, `20%` Mineralogy, `15%` Subsurface, `15%` Thermal, `15%` Atmospheric, `15%` Astrobiology).

### Stage F: Two-Stage Coarse-to-Fine Zoom & Incremental Persistence
1. **Stage 1 (`stage1_coarse`):** Evaluates all arid `0.45°` tiles across the selected frontier corridors.
2. **Stage 2 (`stage2_fine`):** Automatically subdivides the top `--zoom-top-k` (default `4`) coarse discoveries into `0.10°` (`~11 km × 11 km`) high-resolution sub-tiles to pinpoint localized mineralogical and radar hotspots.
3. **Dynamic Persistence (`DynamicNovelStore`):** Appends every result live to `*_stream.jsonl` and re-sorts `*_ranked.csv` and `*_ranked.json` after each tile so progress is crash-safe and inspectable in real time.

---

## 3. Summary of Discoveries: North America Frontier vs. Jezero Crater (Mars)

In the completed North American sweep targeting **Jezero Crater (`18.45° N, 77.45° E`)** under the `balanced` mission profile ([`novel_north_america_frontier_jezero_crater_balanced_ranked.csv`](../discovery/novel_north_america_frontier_jezero_crater_balanced_ranked.csv)), **229 candidate frontier tiles** were generated across five North American regions:

- **4 Corridors Filtered Out by Aridity (`P/PET > 0.25`):** `Owyhee_Alvord_Basin`, `Bolson_de_Mapimi_Mexico`, `Sevier_West_Desert_UT`, and `Mackenzie_Peel_Plateau` were automatically skipped at Tier-0 because their 2023–2024 precipitation-to-evapotranspiration ratios exceeded `0.25`.
- **1 Corridor Validated (`Vizcaino_Baja_Mesas`, Baja California Sur, Mexico):** **76 tiles** (`14` Stage 1 coarse tiles + `62` Stage 2 fine zoom tiles) passed all aridity and confidence checks, located **312.9 km to 533.1 km** away from the nearest known NASA analog site (`Pinacate_Sonora_Mexico`).

### Dataset-Wide Statistical Profile (76 Validated Tiles)

| Metric | Mean | Median | Std Dev | Min | Max |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall Score** | `23.97%` | `24.60%` | `3.72%` | `14.65%` | `31.89%` |
| **Subsurface Parity (15% wt)** | `77.46%` | `88.40%` | `24.77%` | `18.70%` | `99.99%` |
| **Mineralogy Parity (20% wt)** | `37.88%` | `36.08%` | `12.27%` | `10.74%` | `68.66%` |
| **Atmospheric Parity (15% wt)** | `31.85%` | `31.77%` | `0.61%` | `28.53%` | `32.66%` |
| **Aridity Index (`P/PET`)** | `0.1995` | `0.1952` | `0.0117` | `0.1836` | `0.2350` |
| **Distance to Nearest Known Site** | `438.0 km` | `433.1 km` | `60.1 km` | `312.9 km` | `533.1 km` |

### Top 10 Discovered Jezero Crater Analog Locations

Every candidate below activated the complete Earth telemetry stack (`DEM+S2+S1_SAR+Thermal+POWER+SoilGrids+EMIT`) plus `RemoteCLIP`:

| Rank | Stage | Center Coordinates | Overall Score | Normalized Active Score* | Mineralogy | Subsurface | Atmospheric | Aridity (`P/PET`) | Distance from Pinacate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | `stage2_fine` | **`27.650° N, -113.950° W`** | **31.89%** | **63.78%** | **59.97%** | **99.98%** | **32.66%** | `0.1887` | `460.2 km` |
| **#2** | `stage2_fine` | **`27.750° N, -114.050° W`** | **30.12%** | **60.24%** | **53.36%** | **97.98%** | **31.67%** | `0.2024` | `450.1 km` |
| **#3** | `stage2_fine` | **`28.450° N, -112.850° W`** | **29.93%** | **59.86%** | **51.01%** | **99.99%** | **31.54%** | `0.1948` | `374.4 km` |
| **#4** | `stage2_fine` | **`28.200° N, -114.050° W`** | **29.60%** | **59.20%** | **51.23%** | **98.12%** | **30.93%** | `0.2217` | `400.5 km` |
| **#5** | `stage2_fine` | **`27.750° N, -113.950° W`** | **29.35%** | **58.70%** | **58.51%** | **85.68%** | **31.95%** | `0.1969` | `449.1 km` |
| **#6** | `stage2_fine` | **`28.450° N, -112.950° W`** | **29.32%** | **58.64%** | **59.30%** | **84.89%** | **31.54%** | `0.1948` | `372.9 km` |
| **#7** | `stage2_fine` | **`28.350° N, -112.850° W`** | **29.12%** | **58.24%** | **47.63%** | **99.06%** | **31.54%** | `0.1948` | `385.4 km` |
| **#8** | `stage2_fine` | **`27.650° N, -114.050° W`** | **29.08%** | **58.16%** | **49.04%** | **96.17%** | **32.32%** | `0.1939` | `461.2 km` |
| **#9** | `stage1_coarse` | **`28.525° N, -112.975° W`** | **28.97%** | **57.94%** | **49.25%** | **95.92%** | **31.54%** | `0.1948` | `364.3 km` |
| **#10** | `stage2_fine` | **`27.550° N, -113.850° W`** | **27.92%** | **55.84%** | **43.79%** | **95.06%** | **32.66%** | `0.1887` | `470.5 km` |

*\*Note on Normalized Active Score: In this run, **Geomorphology (`20%`)**, **Thermal (`15%`)**, and **Astrobiology (`15%`)** evaluated to `0.0` (`overall_confidence = 33.33%`) because `jezero_crater` did not return a Mars DEM raster or astrobiology scalar from PDS ODE, and Earth daytime Landsat thermal did not pair with a nighttime thermal raster. Dividing `overall_score` by the `0.50` active sector weight yields the normalized parity across the three active sectors (`Mineralogy`, `Subsurface`, `Atmospheric`).*

### Key Scientific Insights from the Discoveries

1. **Two Distinct High-Parity Analog Zones Identified in Baja California Sur:**
   - **Western Vizcaíno Desert / Sierra Vizcaíno Zone (`27.65°–28.20° N, -113.95°–-114.05° W`):** Produced the top overall match (**Rank #1 at `27.650° N, -113.950° W`**, `31.89%` raw / `63.78%` active parity), combining hyper-arid conditions (`P/PET = 0.1887`), `99.98%` Sentinel-1 radar backscatter parity, and `59.97%` EMIT hyperspectral mineral parity.
   - **Eastern San Borja / Gulf Escarpment Mesa Zone (`28.35°–28.45° N, -112.85°–-112.95° W`):** Produced **Ranks #3, #6, and #7**, featuring `99.99%` subsurface radar parity and `59.30%` mineralogical parity.
2. **Peak Pure Mineralogy Hotspot (`Rank #17`):** Coarse tile `s1_Vizcaino_Baja_Mesas_r003_c003` (`28.075° N, -113.425° W`) achieved the single highest **Mineralogy parity (`68.66%`)** in the entire dataset.
    **Value of Coarse-to-Fine Zooming:** Stage 2 (`0.10°`) sub-tiles raised the mean overall score from **`21.70%`** (Stage 1 coarse) to **`24.49%`** (Stage 2 fine) and captured **8 of the top 9 rankings**, isolating localized basaltic/evaporitic terrain pockets that were diluted in `0.45°` coarse cells.
