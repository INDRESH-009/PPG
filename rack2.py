import io
import requests
import pandas as pd

# 1) Specify the case ID you want (e.g., 1)
CASE_ID = 1

# 2) Fetch all track metadata for that case
trk_list_url = f"https://api.vitaldb.net/trks?caseid={CASE_ID}"
r = requests.get(trk_list_url)
r.raise_for_status()

# 3) Load the CSV text into a DataFrame
trk_meta = pd.read_csv(
    io.StringIO(r.content.decode('utf-8'))
)
# 4) Filter for the ABP/ART track(s)
#    Commonly, track_name contains "ART" or "ABP"
# Debug: Print column names
print("Columns in track metadata:", trk_meta.columns)

# Filter for ABP/ART tracks
abp_rows = trk_meta[
    trk_meta["tname"].str.contains("ART|ABP", case=False, na=False)
]

if abp_rows.empty:
    print("No ABP/ART track found for case", CASE_ID)
else:
    # If there are multiple matches, inspect all
    print("ABP‐related tracks for case", CASE_ID)
    print(abp_rows[["caseid", "tname", "tid"]])

    # Pick one (e.g. the first) and note its 'tid'
    abp_tid = abp_rows.iloc[0]["tid"]
    print("\n→ Use this tid to download raw ABP data:", abp_tid)
    print("  GET https://api.vitaldb.net/{}".format(abp_tid))
    
    abp_url = f"https://api.vitaldb.net/{abp_tid}"
abp_response = requests.get(abp_url)
abp_response.raise_for_status()

# Save the raw data to a file
output_file = f"abp_track_{abp_tid}.csv"
with open(output_file, "wb") as f:
    f.write(abp_response.content)

print(f"Raw ABP data saved to: {output_file}")