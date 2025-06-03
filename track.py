import io
import requests
import pandas as pd
import os
import gzip
import shutil

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

# 1) Choose the case ID (patient) you want to pull data for
CASE_ID = 1  # replace with the desired case ID (e.g., 42, 1001, etc.)

# 2) Base API URL for VitalDB open dataset
#    (No authentication required for public cases)
BASE_API_URL = "https://api.vitaldb.net"

# 3) Directory where you want to save downloaded CSVs
OUTPUT_DIR = f"vitaldb_case_{CASE_ID}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# STEP 1: Fetch Track Metadata for the Given Case
# ------------------------------------------------------------------------------

# The "tracks" endpoint returns a GZip‐compressed CSV listing all tracks for CASE_ID.
trk_list_url = f"{BASE_API_URL}/trks?caseid={CASE_ID}"

print(f"➜ Fetching track metadata for case {CASE_ID} ...")
response = requests.get(trk_list_url)

# Raise an error if the request failed.
response.raise_for_status()  # will throw an HTTPError if status >= 400

# The response payload is GZip‐compressed CSV. We can load it directly into pandas.
trk_metadata = pd.read_csv(
    io.StringIO(response.content.decode('utf-8'))
)

print(f"✅ Retrieved {len(trk_metadata)} tracks for case {CASE_ID}.")
print("Sample of track metadata:")
print(trk_metadata.head(5))  # show first 5 rows

# ------------------------------------------------------------------------------
# STEP 2: Loop Through Each Track and Download Its Data
# ------------------------------------------------------------------------------

# The CSV has (at least) these columns: ['id', 'track_name', 'unit', 'gain', 'offset', ...]
# We will use the 'id' column (the unique 'tid') to fetch data for each track.
print("Columns in track metadata:", trk_metadata.columns)
for idx, row in trk_metadata.iterrows():
    tid = row["tid"]               # Unique track ID (tid)
    tname = row["tname"]           # e.g., "Solar 8000/ART"
    print(f"\nDownloading track [{tid}] → '{tname}' ...")

    # Check if 'unit', 'gain', and 'offset' exist before accessing them
    unit = row.get("unit", "unknown")  # Default to "unknown" if not present
    gain = row.get("gain", "unknown")  # Default to "unknown" if not present
    offset = row.get("offset", "unknown")  # Default to "unknown" if not present

    print(f"\nDownloading track [{tid}] → '{tname}' (unit={unit}, gain={gain}, offset={offset}) ...")

    # 2.1) Construct the endpoint to download the actual data (CSV, GZipped)
    #      Endpoint format: GET https://api.vitaldb.net/{tid}
    track_url = f"{BASE_API_URL}/{tid}"

    # 2.2) Send GET request
    trk_response = requests.get(track_url, stream=True)
    trk_response.raise_for_status()

    # 2.3) Save the raw GZip‐compressed file
    gz_path = os.path.join(OUTPUT_DIR, f"track_{tid}.csv.gz")
    with open(gz_path, "wb") as f_out:
        for chunk in trk_response.iter_content(chunk_size=8192):
            f_out.write(chunk)

    # 2.4) Unzip (.csv.gz → .csv)
    csv_path = gz_path.replace(".csv.gz", ".csv")
    with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    # 2.5) (Optional) Delete the .gz if you only need the .csv
    os.remove(gz_path)

    print(f"  ↳ Saved and uncompressed → {csv_path}")

print("\n🔔 All tracks downloaded successfully!")
