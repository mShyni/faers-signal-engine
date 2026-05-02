# src/download_extract_faers.py

from pathlib import Path
import requests
import zipfile

BASE_URL = "https://fis.fda.gov/content/Exports/faers_ascii_{}.zip"

QUARTERS = [
    "2023q1", "2023q2", "2023q3", "2023q4",
    "2024q1", "2024q2", "2024q3", "2024q4",
    "2025q1", "2025q2", "2025q3", "2025q4",
]

RAW_DIR = Path("data/raw")
EXTRACT_DIR = Path("data/extracted")

RAW_DIR.mkdir(parents=True, exist_ok=True)
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url, output_path):
    if output_path.exists():
        print(f"Already downloaded: {output_path.name}")
        return

    print(f"Downloading: {url}")

    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    with open(output_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)

    print(f"Saved: {output_path}")


def extract_zip(zip_path, extract_to):
    quarter_folder = extract_to / zip_path.stem

    if quarter_folder.exists():
        print(f"Already extracted: {quarter_folder}")
        return

    quarter_folder.mkdir(parents=True, exist_ok=True)

    print(f"Extracting: {zip_path.name}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(quarter_folder)

    print(f"Extracted to: {quarter_folder}")


def main():
    for quarter in QUARTERS:
        zip_url = BASE_URL.format(quarter)
        zip_path = RAW_DIR / f"faers_ascii_{quarter}.zip"

        try:
            download_file(zip_url, zip_path)
            extract_zip(zip_path, EXTRACT_DIR)
        except Exception as e:
            print(f"Failed for {quarter}: {e}")


if __name__ == "__main__":
    main()