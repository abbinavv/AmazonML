import pandas as pd
from utils import download_images
import os

DATASETS = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset/train.csv')),
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset/test.csv'))
]
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../image_cache'))

def get_all_image_links():
    links = set()
    for dataset in DATASETS:
        df = pd.read_csv(dataset)
        if 'image_link' in df.columns:
            links.update(df['image_link'].dropna().unique())
    return list(links)

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    image_links = get_all_image_links()
    print(f"Found {len(image_links)} unique image links. Starting download...")
    download_images(image_links, CACHE_DIR)
    print("Download complete.")

if __name__ == "__main__":
    main()
