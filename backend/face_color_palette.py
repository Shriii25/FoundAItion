import os
import cv2
import dlib
import numpy as np
from sklearn.cluster import KMeans
import json  # <-- ADDED: For saving data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_color_palette(pixels, n_colors=5):
    try:
        pixels = pixels.reshape(-1, 3)
        
        if len(pixels) < n_colors:
            return None

        kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=0)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_.astype(int)
        
        # Sort by luminance (brightness)
        colors = sorted(colors, key=lambda c: c[0]*0.299 + c[1]*0.587 + c[2]*0.114)
        
        return colors
    except ValueError:
        return None

def draw_palette(frame, colors):
    if colors is None:
        return frame
        
    swatch_size = 50
    margin = 10
    
    x_start = margin
    y_start = margin
    
    for i, color in enumerate(colors):
        # The palette is in RGB, but cv2 needs BGR
        bgr_color = (int(color[2]), int(color[1]), int(color[0]))
        
        x = x_start + i * (swatch_size + margin)
        cv2.rectangle(frame, (x, y_start), (x + swatch_size, y_start + swatch_size), bgr_color, -1)
        cv2.rectangle(frame, (x, y_start), (x + swatch_size, y_start + swatch_size), (0,0,0), 2)
        
    return frame

# --- NEW FUNCTION TO SAVE PALETTE AS AN IMAGE ---
def save_palette_image(colors, filename="palette.png"):
    if colors is None:
        print("No palette to save.")
        return
        
    swatch_size = 100  # Make swatches larger for the saved image
    margin = 20
    n_colors = len(colors)
    
    # Create a new white background image
    img_width = (swatch_size + margin) * n_colors + margin
    img_height = swatch_size + 2 * margin
    palette_img = np.full((img_height, img_width, 3), 255, dtype=np.uint8) # White background

    x_start = margin
    y_start = margin
    
    for i, color in enumerate(colors):
        # colors are RGB, cv2 needs BGR
        bgr_color = (int(color[2]), int(color[1]), int(color[0]))
        
        x = x_start + i * (swatch_size + margin)
        cv2.rectangle(palette_img, (x, y_start), (x + swatch_size, y_start + swatch_size), bgr_color, -1)
        cv2.rectangle(palette_img, (x, y_start), (x + swatch_size, y_start + swatch_size), (0,0,0), 2)
    
    cv2.imwrite(filename, palette_img)
    print(f"Palette image saved to {filename}")
# --- END OF NEW FUNCTION ---

print("Loading dlib models...")
# Make sure you have this file in the same directory
try:
    detector = dlib.get_frontal_face_detector()
    predictor_path = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")
    predictor = dlib.shape_predictor(predictor_path)
except RuntimeError:
    print("Error: 'shape_predictor_68_face_landmarks.dat' not found.")
    print("Please download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()
print("Models loaded.")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

LANDMARK_INDICES = {
    "left_cheek": 31,
    "right_cheek": 35,
    "forehead_left": 21,
    "forehead_right": 22,
    "nose_bridge": 27,
    "nose_tip": 30,
    "chin": 8
}

SAMPLE_SIZE = 20
latest_palette = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    all_skin_pixels = []

    for face in faces:
        landmarks = predictor(gray, face)

        for region_name, index in LANDMARK_INDICES.items():
            x = landmarks.part(index).x
            y = landmarks.part(index).y
            
            half = SAMPLE_SIZE // 2
            x1 = max(0, x - half)
            y1 = max(0, y - half)
            x2 = min(frame.shape[1], x + half)
            y2 = min(frame.shape[0], y + half)

            skin_sample = rgb_frame[y1:y2, x1:x2]
            
            if skin_sample.size > 0:
                all_skin_pixels.extend(skin_sample.reshape(-1, 3))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    if all_skin_pixels:
        palette = get_color_palette(np.array(all_skin_pixels), n_colors=5)
        if palette is not None:
            latest_palette = palette

    frame_with_palette = draw_palette(frame.copy(), latest_palette)
    
    # Add text instruction
    cv2.putText(frame_with_palette, "Press 's' to save, 'q' to quit", 
                (10, frame_with_palette.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Face Color Palette", frame_with_palette)

    # --- MODIFIED KEY HANDLER ---
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        if latest_palette is not None:
            print("Saving palette...")
            
            # 1. Save as a PNG image
            save_palette_image(latest_palette, r"C:\Users\Shriya\OneDrive\Desktop\Fount_ai_tion_WebApp\backend\saved_palette.png")
            
            # 2. Save as JSON data
            # Convert numpy array elements to standard int lists for JSON
            palette_data = [list(map(int, color)) for color in latest_palette]
            with open(r"C:\Users\Shriya\OneDrive\Desktop\Fount_ai_tion_WebApp\backend\saved_palette.json", 'w') as f:
                json.dump(palette_data, f, indent=4)
                
            print(f"Palette data saved to saved_palette.json")
            print("...Save complete.")
            
        else:
            print("No palette has been captured yet.")
    # --- END OF MODIFIED KEY HANDLER ---

cap.release()
cv2.destroyAllWindows()