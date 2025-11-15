#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Debug-enhanced Xeno-canto v3 downloader (fixed for API v3 key usage)
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import requests

# --- CONFIG ---
API_BASE = "https://xeno-canto.org/api/3/recordings"
API_KEY = os.getenv("XC_API_KEY")

print("========== DEBUG INFO ==========")
print(f"[DEBUG] API_BASE: {API_BASE}")
print(f"[DEBUG] API_KEY (first 5 chars): {API_KEY[:5] + '...' if API_KEY else 'None'}")
print("================================\n")

# --- COMMON NAME TO SCIENTIFIC NAME MAP ---
COMMON_TO_SCI = {
    "Indian peafowl": ("Pavo", "cristatus"),
    "Grey francolin": ("Ortygornis", "pondicerianus"),
    "Spotted dove": ("Spilopelia", "chinensis"),
    "Eastern barn owl": ("Tyto", "javanica"),
    "Plum headed parakeet": ("Psittacula", "cyanocephala"),
    "Asian green bee eater": ("Merops", "orientalis"),
    "Coppersmith barbet": ("Psilopogon", "haemacephalus"),
    "Indian robin": ("Copsychus", "fulicatus"),
    "Oriental magpie Robin": ("Copsychus", "saularis"),
    "Taiga flycatcher": ("Ficedula", "albicilla"),
    "Red-breasted Flycatcher": ("Ficedula", "parva"),
    "Red-vented Bulbul": ("Pycnonotus", "cafer"),
    "Large Gray Babbler": ("Argya", "malcolmi"),
    "Gray-breasted prinia": ("Prinia", "hodgsonii"),
    "Jungle Prinia": ("Prinia", "sylvatica"),
    "Ashy Prinia": ("Prinia", "socialis"),
    "Hume's Warbler": ("Phylloscopus", "humei"),
    "Common Chiffchaff": ("Phylloscopus", "collybita"),
    "Green Warbler": ("Phylloscopus", "nitidus"),
    "Greenish Warbler": ("Phylloscopus", "trochiloides"),
    "Purple Sunbird": ("Cinnyris", "asiaticus"),
    "Western yellow wagtail": ("Motacilla", "flava"),
    "Red avadavat": ("Amandava", "amandava"),
}

DEFAULT_SPECIES = list(COMMON_TO_SCI.keys())

# --- HELPERS ---
def slugify(s: str) -> str:
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)
    return s


def build_query(gen: str, sp: str, country: str, qualities: List[str], vtypes: List[str]) -> str:
    parts = [f"gen:{gen}", f"sp:{sp}"]
    if country:
        parts.append(f"cnt:{country}")
    if qualities:
        if len(qualities) == 1:
            parts.append(f"q:{qualities[0]}")
        else:
            parts.append("(" + " OR ".join([f"q:{q}" for q in qualities]) + ")")
    if vtypes:
        if len(vtypes) == 1:
            parts.append(f"type:{vtypes[0]}")
        else:
            parts.append("(" + " OR ".join([f"type:{t}" for t in vtypes]) + ")")
    return " ".join(parts)


def fetch_all_pages(query: str, delay: float = 0.5, max_pages: int = 50) -> List[dict]:
    page = 1
    all_recs = []
    while page <= max_pages:
        params = {"query": query, "page": page, "key": API_KEY}
        print(f"[DEBUG] Fetching page {page} with query: {query}")
        try:
            r = requests.get(API_BASE, params=params, timeout=30)
        except Exception as e:
            print(f"[ERROR] Request exception on page {page}: {e}", file=sys.stderr)
            break

        print(f"[DEBUG] Response status: {r.status_code}")
        if r.status_code != 200:
            print(f"[WARN] Non-200 status on page {page}: {r.text[:200]}", file=sys.stderr)
            break
        try:
            data = r.json()
        except Exception as e:
            print(f"[ERROR] JSON parse failed: {e}", file=sys.stderr)
            print(f"[DEBUG] Response text: {r.text[:500]}")
            break

        recs = data.get("recordings", [])
        print(f"[DEBUG] Got {len(recs)} recordings on page {page}")
        all_recs.extend(recs)
        num_pages = int(data.get("numPages", 1) or 1)
        if page >= num_pages:
            break
        page += 1
        time.sleep(delay)
    return all_recs


def safe_download(url: str, out_path: Path, max_retries: int = 3) -> bool:
    """Download a file with retries (no API key needed for audio files)."""
    for i in range(max_retries):
        try:
            print(f"[DEBUG] Downloading {url} -> {out_path}")
            with requests.get(url, stream=True, timeout=60) as r:
                if r.status_code != 200:
                    print(f"[WARN] Download failed (status {r.status_code})", file=sys.stderr)
                    time.sleep(0.75)
                    continue
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(1024 * 64):
                        if chunk:
                            f.write(chunk)
            print("[DEBUG] Download success")
            return True
        except requests.RequestException as e:
            print(f"[ERROR] Download exception {i+1}: {e}", file=sys.stderr)
            time.sleep(1.0 + i * 0.5)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="xc_data")
    ap.add_argument("--per_species", type=int, default=40)
    ap.add_argument("--country", default="India")
    ap.add_argument("--quality", default="A,B")
    ap.add_argument("--type", default="song,call")
    ap.add_argument("--species_file", default="")
    ap.add_argument("--delay", type=float, default=0.5)
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    qualities = [q.strip().upper() for q in args.quality.split(",") if q.strip()]
    vtypes = [t.strip().lower() for t in args.type.split(",") if t.strip()]

    if args.species_file and Path(args.species_file).exists():
        with open(args.species_file, "r", encoding="utf-8") as fh:
            species_list = [line.strip() for line in fh if line.strip()]
    else:
        species_list = DEFAULT_SPECIES

    # --- CSV Setup ---
    tsv_file = out_root / "train_mapping.tsv"
    with open(tsv_file, "w", newline="", encoding="utf-8") as tsv_fh:
        tsv_writer = csv.writer(tsv_fh, delimiter="\t")
        tsv_writer.writerow(["species", "file_name", "url", "quality", "type", "duration", "date"])

        for common in species_list:
            print(f"\n========== STARTING SPECIES: {common} ==========")
            if common not in COMMON_TO_SCI:
                print(f"[WARN] Species not in map: {common}")
                continue
            gen, sp = COMMON_TO_SCI[common]
            query = build_query(gen, sp, args.country, qualities, vtypes)
            print(f"[DEBUG] Query string: {query}")
            recs = fetch_all_pages(query, delay=args.delay)
            print(f"[DEBUG] Total recordings fetched for {common}: {len(recs)}")
            
            for rec in recs:
                # Extract relevant fields
                species = common
                file_name = f"{slugify(species)}_{rec['id']}.mp3"
                url = rec["file"]  # assuming 'file' is the URL for download
                quality = rec.get("q", "N/A")  # Quality
                vtype = rec.get("type", "N/A")  # Type (song/call)
                duration = rec.get("length", "N/A")  # Duration
                date = rec.get("date", "N/A")  # Date of recording
                
                # Write to the TSV file
                tsv_writer.writerow([species, file_name, url, quality, vtype, duration, date])

    print(f"Finished writing data to {tsv_file}")


if __name__ == "__main__":
    main()
