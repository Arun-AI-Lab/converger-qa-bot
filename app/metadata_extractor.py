# app/metadata_extractor.py
import os
import re
from datetime import datetime
from typing import Dict

# Supported Australian regions (with abbreviations)
AU_REGIONS = {
    "New South Wales": ["nsw", "new south wales"],
    "Victoria": ["vic", "victoria"],
    "Queensland": ["qld", "queensland"],
    "Western Australia": ["wa", "western australia"],
    "South Australia": ["sa", "south australia"],
    "Tasmania": ["tas", "tasmania"],
    "Northern Territory": ["nt", "northern territory"],
    "Australian Capital Territory": ["act", "australian capital territory", "canberra"],
}

def detect_region(text: str, filename: str) -> str:
    """
    Detect Australian state or territory from text or filename.
    Defaults to 'Australia' if no state found.
    """
    combined = f"{filename} {text}".lower()
    for region, patterns in AU_REGIONS.items():
        if any(p in combined for p in patterns):
            return region
    return "Australia"  # fallback default


def detect_contract_type(filename: str, text: str) -> str:
    """Detects contract type based on filename or PDF content."""
    combined = f"{filename} {text}".lower()
    if "vendor" in combined:
        return "Vendor Contract"
    elif "ffs" in combined:
        return "Contract - FFS"
    elif "retainer" in combined or "ret" in combined:
        return "Contract - Retainer"
    else:
        return "General"


def detect_party_type(contract_type: str) -> str:
    """Map contract type to party type."""
    if "Vendor" in contract_type:
        return "Vendor"
    elif "Contract" in contract_type:
        return "Client"
    return "General"


def extract_client_name(text: str, filename: str) -> str:
    """
    Attempts to extract client/vendor name from text or filename.
    """
    match = re.search(
        r"(?:between|with)\s+([A-Z][A-Za-z0-9&\s]+?)(?:Pty|Ltd|Limited|Corporation|Contract|Agreement)",
        text[:400], re.IGNORECASE
    )
    if match:
        return match.group(1).strip()

    base = os.path.basename(filename)
    clean_name = re.sub(r"[_\-]", " ", base).split(".pdf")[0]
    return clean_name.strip().title()


def extract_dates(text: str):
    """Extract possible start and end dates from text."""
    date_pattern = r"(\d{1,2}\s?[A-Za-z]{3,9}\s?\d{4})"
    start_match = re.search(r"(?:commencing|start|effective from|dated)\s+" + date_pattern, text, re.IGNORECASE)
    end_match = re.search(r"(?:until|to|through|end(?:ing)? on)\s+" + date_pattern, text, re.IGNORECASE)
    start_date = start_match.group(1) if start_match else None
    end_date = end_match.group(1) if end_match else None
    return start_date, end_date


def generate_file_metadata(file_path: str, text: str = "") -> Dict:
    """Generates structured metadata for each PDF."""
    filename = os.path.basename(file_path)
    upload_time = datetime.utcnow().isoformat()

    region = detect_region(text, filename)
    contract_type = detect_contract_type(filename, text)
    party_type = detect_party_type(contract_type)
    client_name = extract_client_name(text, filename)
    start_date, end_date = extract_dates(text)

    file_type = "contract" if "contract" in contract_type.lower() else "general"

    return {
        "filename": filename,
        "upload_time": upload_time,
        "file_type": file_type,
        "contract_type": contract_type,
        "party_type": party_type,
        "region": region,
        "client_name": client_name,
        "start_date": start_date,
        "end_date": end_date,
    }
