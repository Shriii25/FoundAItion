import pandas as pd
import json
import numpy as np
import re
from sklearn.cluster import KMeans
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# Extract dominant colors using KMeans
def get_color_palette(pixels, n_colors=5):
    # Use RGB channels only
    try:
        if pixels.shape[1] == 4:
            pixels = pixels[:, :3]

        pixels = pixels.reshape(-1, 3)

        if len(pixels) < n_colors:
            print(f"Warning: Not enough pixels ({len(pixels)}) to find {n_colors} colors. Skipping.")
            return None

        kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=0)
        kmeans.fit(pixels)

        colors = kmeans.cluster_centers_.astype(int)
        # Sort by luminance for consistent ordering
        colors = sorted(colors, key=lambda c: c[0]*0.299 + c[1]*0.587 + c[2]*0.114)

        return [list(map(int, color)) for color in colors]
    except ValueError as e:
        print(f"Error in KMeans clustering: {e}. Skipping.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred in get_color_palette: {e}")
        return None

# Parse pixel string from CSV into RGB numpy array
def parse_pixel_string(pixel_string):
    try:
        numbers = re.findall(r'(\d+)', pixel_string)
        pixel_array = np.array(numbers, dtype=int)
        pixel_array = pixel_array.reshape(-1, 4)  # RGBA expected
        return pixel_array[:, :3]  # Return RGB
    except Exception as e:
        print(f"Error parsing pixel string: {e}")
        return None

all_brand_palettes = []

csv_files_info = [
    {"file": "loreal_india.csv", "column": "loreal_india", "brand": "L'Oreal"},
    {"file": "mac_ind.csv", "column": "mac_ind", "brand": "MAC"},
    {"file": "maybelline_ind.csv", "column": "maybelline_ind", "brand": "Maybelline"}
]

print("Starting processing of brand product palettes...")

for info in csv_files_info:
    file = info["file"]
    col = info["column"]
    brand = info["brand"]

    print(f"\nProcessing {file}...")
    try:
        df = pd.read_csv(file)
        df = df.rename(columns={"Unnamed: 0": "product_id"})

        for index, row in df.iterrows():
            product_id = row["product_id"]
            pixel_string = row[col]

            pixels = parse_pixel_string(pixel_string)

            if pixels is not None and len(pixels) > 0:
                palette = get_color_palette(pixels, n_colors=5)

                if palette:
                    all_brand_palettes.append({
                        "brand": brand,
                        "product_id": product_id,
                        "palette": palette
                    })
                    print(f"  > Generated palette for {brand} product {product_id}")
                else:
                    print(f"  > Could not generate palette for {brand} product {product_id}")
            else:
                print(f"  > Could not parse pixels for {brand} product {product_id}")

    except FileNotFoundError:
        print(f"Error: {file} not found. Skipping.")
    except Exception as e:
        print(f"An error occurred while processing {file}: {e}")

print("\n...Brand processing complete.")

try:
    save_path = os.path.join(BASE_DIR, 'processed_brand_palettes.json')
    with open(save_path, 'w') as f:
        json.dump(all_brand_palettes, f, indent=4)
    print(f"✅ Saved processed palettes to: {save_path}")
    print(f"\nSuccessfully processed {len(all_brand_palettes)} brand palettes.")
except Exception as e:
    print(f"\nError saving processed palettes to JSON: {e}")

print("\n--- Checking for user's saved face palette ---")
try:
    with open("saved_palette.json", 'r') as f:
        face_palette = json.load(f)
    print("Success: 'saved_palette.json' found.")
except FileNotFoundError:
    print("Error: 'saved_palette.json' not found. Run the webcam script and save your palette.")
except Exception as e:
    print(f"An error occurred reading 'saved_palette.json': {e}")
