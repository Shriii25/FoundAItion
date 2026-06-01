import os
import json
import numpy as np
from sklearn.metrics import pairwise_distances
from skimage.color import rgb2lab
from scipy.spatial import distance
from scipy.spatial.distance import cdist
import cv2
import base64

# Utility functions

def calculate_palette_distance(palette1, palette2):
    """Compute mean Euclidean distance between two color palettes in LAB space."""
    palette1_lab = rgb2lab(np.array(palette1).reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
    palette2_lab = rgb2lab(np.array(palette2).reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
    d = distance.cdist(palette1_lab, palette2_lab, "euclidean").min(axis=1)
    return np.mean(d)


def find_closest_shade(user_palette, processed_palettes):
    """Find the closest brand palette to the user's palette. Returns (palette, brand)."""
    if not isinstance(user_palette, (list, np.ndarray)):
        raise ValueError("User palette must be a list or ndarray")

    user_palette = [c for c in user_palette if isinstance(c, (list, np.ndarray)) and len(c) == 3]
    if not user_palette:
        raise ValueError("User palette is empty or invalid.")

    user_lab = rgb2lab(np.array(user_palette).reshape(-1, 1, 3) / 255.0).reshape(-1, 3)

    best_distance = float('inf')
    best_match = None
    best_brand = None

    for entry in processed_palettes:
        if not isinstance(entry, dict) or 'palette' not in entry or 'brand' not in entry:
            continue

        brand_palette = entry['palette']
        brand_name = entry['brand']

        if not isinstance(brand_palette, (list, np.ndarray)):
            continue
        brand_palette = [c for c in brand_palette if isinstance(c, (list, np.ndarray)) and len(c) == 3]
        if not brand_palette:
            continue

        try:
            brand_lab = rgb2lab(np.array(brand_palette).reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
            dist = np.mean(cdist(user_lab, brand_lab).min(axis=1))
        except Exception:
            continue

        if dist < best_distance:
            best_distance = dist
            best_match = brand_palette
            best_brand = brand_name

    if best_match is None:
        raise ValueError("No valid brand palettes found in processed_brand_palettes.json")

    return best_match, best_brand


def create_palette_image(palette, swatch_height=80, swatch_width=80):
    """Create a base64-encoded PNG image showing color swatches for a palette."""
    palette = np.array(palette, dtype=np.uint8).reshape(-1, 3)
    h, w = swatch_height, swatch_width
    img = np.zeros((h, w * len(palette), 3), dtype=np.uint8)
    for i, color in enumerate(palette):
        img[:, i*w:(i+1)*w] = color[::-1]  # convert RGB to BGR for cv2
    _, buf = cv2.imencode(".png", img)
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/png;base64,{b64}"


if __name__ == "__main__":
    print("Module loaded. Using LAB color matching.")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    processed_path = os.path.join(BASE_DIR, "processed_brand_palettes.json")
    saved_palette_path = os.path.join(BASE_DIR, "saved_palette.json")

    user_palette = None

    if not os.path.exists(saved_palette_path):
        print("Error: 'saved_palette.json' not found. Save a palette before running this script.")
    else:
        with open(saved_palette_path, "r") as f:
            user_palette = json.load(f)
        print("Loaded user palette from 'saved_palette.json':")
        print(user_palette)

    if not os.path.exists(processed_path):
        print("Error: 'processed_brand_palettes.json' not found. Please process brand files first.")
    else:
        with open(processed_path, "r") as f:
            processed_palettes = json.load(f)
        print(f"Loaded {len(processed_palettes)} brand palettes.")

        if user_palette is not None:
            try:
                match, brand = find_closest_shade(user_palette, processed_palettes)
                print(f"Closest match: {brand}")
                print(f"Matched palette: {match}")
            except Exception as e:
                print(f"Error during matching: {e}")
        else:
            print("User palette was not loaded; skipping match test.")
