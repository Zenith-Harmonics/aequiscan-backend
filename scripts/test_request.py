#!/usr/bin/env python3
"""
Quick CLI script to test sending a comparison request to the FastAPI app.

Usage:
    python scripts/test_request.py
"""
import asyncio
import json
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from httpx import ASGITransport, AsyncClient

from app.main import app


async def test_comparison():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://localhost:8000") as client:
        print("=" * 60)
        print("  AequiScan API: Mineralogy Comparison Request Test")
        print("=" * 60)

        payload = {
            "earth_lat": 38.4500,
            "earth_lon": 91.4200,
            "earth_radius_m": 2000.0,
            "planetary_target_id": "gale_crater",
            "simulate": True,
            "weights": {
                "mafic_olivine_pyroxene": 1.0,
                "plagioclase_anorthosite": 1.0,
                "phyllosilicates_clays": 1.5,
                "sulfates_evaporites": 2.0,
                "carbonates": 1.0,
                "iron_oxides_hydroxides": 1.0,
            },
        }

        print("\nSending POST /api/v1/mineralogy/compare ...")
        print(f"Payload:\n{json.dumps(payload, indent=2)}\n")

        resp = await client.post("/api/v1/mineralogy/compare", json=payload)
        print(f"HTTP Status: {resp.status_code}")

        if resp.status_code != 200:
            print("Error response:", resp.text)
            return

        data = resp.json()
        print("\n" + "=" * 60)
        print("  COMPARISON RESULTS")
        print("=" * 60)
        print(f"Earth Site:        {data['earth_site_name']}")
        print(f"Planetary Target:  {data['planetary_site_name']} ({data['planetary_body'].upper()})")
        print(f"Overall ASI Score: {data['asi']['asi_score']:.4f} / 1.0000\n")

        print("Axis-by-Axis Breakdown:")
        header = f"{'Mineral Class':<28} | {'Earth':<8} | {'Target':<8} | {'Delta':<8} | {'Weight':<6}"
        print(header)
        print("-" * len(header))
        for ax in data["asi"]["axis_details"]:
            print(
                f"{ax['mineral_class']:<28} | "
                f"{ax['earth_fraction']:8.4f} | "
                f"{ax['planetary_fraction']:8.4f} | "
                f"{ax['abs_delta']:8.4f} | "
                f"{ax['weight']:6.1f}"
            )

        print("\nTop USGS Detections on Earth:")
        for name, count in data["asi"]["earth_profile"]["top_raw_minerals"]:
            print(f"  - {name}: {count} pixels")


if __name__ == "__main__":
    asyncio.run(test_comparison())
