"""
Download CUAD and save a small sample locally as cuad_sample.json.
 
We download directly from the CUAD official release on Zenodo
(the authoritative source). This avoids the HuggingFace `datasets`
library's broken support for script-based loaders in v4.x.
 
The zip from Zenodo contains a file called CUADv1.json which is in
SQuAD format:
    {
      "data": [
        {
          "title": "contract_name",
          "paragraphs": [
            {
              "context": "<full contract text>",
              "qas": [
                {
                  "question": "...'Governing Law'...",
                  "answers": [{"text": "...", "answer_start": ...}],
                  ...
                },
                ...
              ]
            }
          ]
        },
        ...
      ]
    }
 
We reorganise this into our simpler per-contract format.
 
Usage:
    python data/prepare_cuad.py
"""
 
import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
 
import requests
from tqdm import tqdm
 
sys.path.append(str(Path(__file__).parent.parent))
import config
 
 
# Zenodo record for CUAD v1 (DOI 10.5281/zenodo.4595826).
# This is the OFFICIAL release and is a stable URL.
CUAD_ZIP_URL = "https://zenodo.org/records/4595826/files/CUAD_v1.zip"
 
 
# Map our 10-category shortlist to the substring that identifies the
# CUAD question. Each CUAD question contains the category name (the
# exact phrasing varies slightly across versions, so we match by
# substring, case-insensitive).
OUR_CATEGORIES_TO_CUAD_LABEL = {
    "Governing Law":                 "governing law",
    "Termination for Convenience":   "termination for convenience",
    "Liability Cap":                 "cap on liability",
    "Indemnification":               "indemnification",
    "IP Ownership Assignment":       "ip ownership assignment",
    "Change of Control":             "change of control",
    "Audit Rights":                  "audit rights",
    "Warranty Duration":             "warranty duration",
    "Non-Compete":                   "non-compete",
    "Insurance Requirements":        "insurance",
}
 
 
def download_cuad_json() -> dict:
    """Download CUAD_v1.zip from Zenodo and extract the SQuAD JSON."""
    # Cache the zip locally so re-running the script doesn't
    # re-download 500MB.
    cache_dir = config.DATA_DIR / ".cache"
    cache_dir.mkdir(exist_ok=True)
    zip_path = cache_dir / "CUAD_v1.zip"
 
    if zip_path.exists():
        print(f"  using cached zip at {zip_path}")
    else:
        print(f"  downloading from {CUAD_ZIP_URL}")
        # Stream with progress bar.
        with requests.get(CUAD_ZIP_URL, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(zip_path, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True,
                desc="  downloading"
            ) as bar:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    f.write(chunk)
                    bar.update(len(chunk))
 
    # Extract CUADv1.json from the zip.
    print(f"  extracting CUADv1.json from zip...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find the JSON file inside (name might include subfolder).
        json_names = [n for n in zf.namelist() if n.endswith("CUAD_v1.json")]
        if not json_names:
            raise RuntimeError(
                "Could not find CUADv1.json inside the Zenodo zip. "
                "Available files: " + ", ".join(zf.namelist()[:10])
            )
        with zf.open(json_names[0]) as f:
            data = json.load(f)
 
    print(f"  loaded {len(data['data'])} contracts from CUADv1.json")
    return data
 
 
def reorganise_cuad(raw: dict, selected_categories: list[dict]) -> list[dict]:
    """Convert SQuAD-format CUAD into our simpler per-contract format."""
    # Build a lookup of keyword -> our category name.
    our_names = {c["name"] for c in selected_categories}
    keyword_to_our_name = {
        kw: our_name
        for our_name, kw in OUR_CATEGORIES_TO_CUAD_LABEL.items()
        if our_name in our_names
    }
 
    contracts = []
    for item in raw["data"]:
        title = item.get("title", "unknown")
 
        # Merge paragraphs into a single contract text (most CUAD
        # entries have a single paragraph anyway).
        context = ""
        all_qas = []
        for p in item.get("paragraphs", []):
            context = context + "\n\n" + p.get("context", "") if context \
                      else p.get("context", "")
            all_qas.extend(p.get("qas", []))
 
        gold_clauses = defaultdict(list)
        for qa in all_qas:
            q = qa.get("question", "").lower()
            answers = [a["text"] for a in qa.get("answers", []) if a.get("text")]
            if not answers:
                continue
            # Find which category this question is about.
            for kw, our_name in keyword_to_our_name.items():
                if kw in q:
                    gold_clauses[our_name].extend(answers)
                    break
 
        contracts.append({
            "contract_id": title,
            "title": title,
            "text": context.strip(),
            "gold_clauses": dict(gold_clauses),
        })
 
    return contracts
 
 
def main():
    print("Preparing CUAD dataset...")
 
    # Load the user-selected categories from data/categories.json.
    with open(config.CATEGORIES_PATH) as f:
        selected_categories = json.load(f)["categories"]
    print(f"  {len(selected_categories)} categories selected: "
          f"{', '.join(c['name'] for c in selected_categories)}")
 
    # Download + parse.
    raw = download_cuad_json()
 
    # Reorganise into our format.
    all_contracts = reorganise_cuad(raw, selected_categories)
    print(f"  {len(all_contracts)} contracts re-formatted")
 
    # Keep only contracts that have at least 3 of our categories annotated
    # (otherwise the contract is not useful for our evaluation).
    useful = [
        c for c in all_contracts
        if sum(1 for v in c["gold_clauses"].values() if len(v) > 0) >= 3
    ]
    print(f"  {len(useful)} contracts have >= 3 of our categories annotated")
 
    # Take the first N for our experiment.
    sample = useful[: config.N_CONTRACTS]
    print(f"  keeping {len(sample)} contracts for the experiment")
 
    # Save.
    config.DATA_DIR.mkdir(exist_ok=True, parents=True)
    with open(config.CUAD_SAMPLE_PATH, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    print(f"  saved to {config.CUAD_SAMPLE_PATH}")
 
 
if __name__ == "__main__":
    main()