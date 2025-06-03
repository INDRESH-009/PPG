from urllib import response
import requests
import json 

BASE_URL = "https://api.vitaldb.net"
OUTPUT_FILE = "vitaldb_metadata.json"
def get_case_list():
    url = f"{BASE_URL}/cases"
    response = requests.get(url)
    response.raise_for_status()
    # Use utf-8-sig to properly decode BOM if present
    cases = json.loads(response.content.decode("utf-8-sig"))
    return cases

def get_metadata_for_case(case_id):
    """Fetch metadata for a specific case ID."""
    url = f"{BASE_URL}/cases/{case_id}"
    response = requests.get(url)
    response.raise_for_status()
    return json.loads(response.content.decode("utf-8-sig"))

def main():
    print("📥 Fetching case list...")
    case_list = get_case_list()  # You can increase limit if needed
    print(f"🔍 Retrieved {len(case_list)} cases.")

    metadata_all = {}

    for case in case_list:
        case_id = case['caseid']
        print(f"📄 Downloading metadata for case ID: {case_id}")
        metadata = get_metadata_for_case(case_id)
        metadata_all[case_id] = metadata

    # Save metadata to file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(metadata_all, f, indent=2)

    print(f"\n✅ Metadata saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
