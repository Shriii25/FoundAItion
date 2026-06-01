from flask import Flask, render_template, Response, jsonify
import cv2, json, numpy as np, os, dlib
from backend.face_color_palette import get_color_palette
from backend.model import find_closest_shade

app = Flask(__name__)

# Initialize camera and dlib
camera = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor_path = os.path.join("backend", "shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor(predictor_path)

latest_palette = None

# Define file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
PROCESSED_PATH = os.path.join(BACKEND_DIR, "processed_brand_palettes.json")
USER_PALETTE_PATH = os.path.join(BACKEND_DIR, "saved_palette.json")

# Path debug log
print("\nPATH DEBUG LOG")
print(f"BASE_DIR: {BASE_DIR}")
print(f"BACKEND_DIR: {BACKEND_DIR}")
print(f"Processed palettes path: {PROCESSED_PATH} (exists? {os.path.exists(PROCESSED_PATH)})")
print(f"Saved palette path: {USER_PALETTE_PATH} (exists? {os.path.exists(USER_PALETTE_PATH)})")
print("----------------------\n")

# Load processed brand palettes
try:
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(f"{PROCESSED_PATH} not found.")
    with open(PROCESSED_PATH, "r") as f:
        brand_palettes = json.load(f)
    print(f"Loaded {len(brand_palettes)} brand palettes.")
except Exception as e:
    print(f"Error loading processed brand palettes: {e}")
    brand_palettes = []

def generate_frames():
    """Capture video frames, extract skin pixels, compute palette, and yield JPEG frames."""
    global latest_palette

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        # Extract skin region using face landmarks when a face is detected
        if len(faces) > 0:
            face = faces[0]
            landmarks = predictor(gray, face)

            # Use jawline and eyebrows approximate skin area
            skin_points = []
            for i in range(1, 16):
                skin_points.append((landmarks.part(i).x, landmarks.part(i).y))
            for i in range(17, 27):
                skin_points.append((landmarks.part(i).x, landmarks.part(i).y))

            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.array(skin_points), 255)
            skin = cv2.bitwise_and(frame, frame, mask=mask)

            rgb = cv2.cvtColor(skin, cv2.COLOR_BGR2RGB)
            pixels = rgb[mask > 0].reshape(-1, 3)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pixels = rgb.reshape(-1, 3)

        # Compute current palette from pixels
        if len(pixels) > 0:
            latest_palette = get_color_palette(pixels, n_colors=5)

        # Draw palette swatches on the frame
        if latest_palette is not None:
            sw = 40
            for i, color in enumerate(latest_palette):
                bgr = tuple(map(int, color[::-1]))
                cv2.rectangle(frame, (10 + i * (sw + 5), 10),
                              (10 + i * (sw + 5) + sw, 50), bgr, -1)

        # Encode and yield frame as multipart JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Web routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/detect")
def detect_shade():
    import json, os, numpy as np
    from backend.model import find_closest_shade

    print("/detect endpoint triggered")

    # Ensure required files exist
    if not os.path.exists(PROCESSED_PATH):
        msg = "processed_brand_palettes.json not found"
        print("ERROR:", msg)
        return jsonify({"error": msg}), 400

    if not os.path.exists(USER_PALETTE_PATH):
        msg = "saved_palette.json not found"
        print("ERROR:", msg)
        return jsonify({"error": msg}), 400

    # Load palettes
    try:
        with open(PROCESSED_PATH) as f:
            processed = json.load(f)
        with open(USER_PALETTE_PATH) as f:
            user = json.load(f)
        print("Palettes loaded successfully")
    except Exception as e:
        print("ERROR: Failed to read JSON:", e)
        return jsonify({"error": str(e)}), 500

    # Match shades
    try:
        matched, brand = find_closest_shade(user, processed)
        print(f"Brand match: {brand}")
    except Exception as e:
        print("ERROR: Shade matching failed:", e)
        return jsonify({"error": f"Error during shade matching: {str(e)}"}), 500

    from backend.model import create_palette_image
    palette_img_b64 = create_palette_image(matched)

    return jsonify({
        "status": "success",
        "brand": brand,
        "matched_shade": matched,
        "user_palette": user,
        "matched_image": palette_img_b64
    })

# Run the Flask app (disable reloader to avoid double execution)
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
