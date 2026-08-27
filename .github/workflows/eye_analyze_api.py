"""
NUSKHA - Eye Health Scan API
============================
A lightweight Flask microservice that receives an eye image, detects the
dominant color/region characteristics (sclera color, pupil cloudiness,
conjunctiva paleness), and maps them to a predefined condition using rules
that mirror the `eye_condition_rules` table in the main MySQL database.

Run standalone:
    pip install flask flask-cors opencv-python-headless numpy pillow --break-system-packages
    python eye_analysis_api.py
Default port: 5001 (matches EYE_SCAN_API_URL in includes/var.php)

NOTE: This is a heuristic color-based prototype for demo/education purposes,
NOT a validated medical diagnostic tool. Every response includes a disclaimer
field and PHP-side code always routes results through a human review step.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
import io
from PIL import Image
import os
import hashlib
import time

app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get("NUSKHA_EYE_API_KEY", "CHANGE_ME_SECRET_KEY")  # must match includes/var.php

# ---------------------------------------------------------------------------
# Predefined color -> condition rules (mirrors admin-editable `eye_condition_rules`
# table in MySQL; keep in sync manually, or later fetch this via a small
# internal PHP endpoint if you want single-source-of-truth).
# ---------------------------------------------------------------------------
CONDITION_RULES = {
    "yellow_sclera": {
        "condition": "Jaundice (Possible Liver Issue)",
        "severity": "high",
        "recommendation": "Consult a doctor promptly for a liver function test.",
    },
    "red_sclera": {
        "condition": "Conjunctivitis / Eye Strain",
        "severity": "medium",
        "recommendation": "Consider consulting a doctor if redness persists more than 2 days.",
    },
    "cloudy_pupil": {
        "condition": "Possible Cataract",
        "severity": "high",
        "recommendation": "Consult an eye specialist for further examination.",
    },
    "pale_conjunctiva": {
        "condition": "Possible Anemia",
        "severity": "medium",
        "recommendation": "Recommend a blood test (CBC) to confirm.",
    },
    "normal": {
        "condition": "No Abnormality Detected",
        "severity": "low",
        "recommendation": "No action needed. Routine checkups recommended.",
    },
}


def decode_image(file_storage_or_b64):
    """Accepts either a Flask FileStorage (multipart upload) or a base64 string."""
    if isinstance(file_storage_or_b64, str):
        if "," in file_storage_or_b64:
            file_storage_or_b64 = file_storage_or_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(file_storage_or_b64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    else:
        pil_img = Image.open(file_storage_or_b64.stream).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def detect_eye_region(bgr_img):
    """
    Locate the eye using OpenCV's built-in Haar cascade.
    Falls back to the full image (center-cropped) if no eye is detected,
    so the API degrades gracefully instead of failing hard.
    """
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
    eye_cascade = cv2.CascadeClassifier(cascade_path)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40))

    h, w = bgr_img.shape[:2]
    if len(eyes) > 0:
        # take the largest detected eye region
        x, y, ew, eh = max(eyes, key=lambda e: e[2] * e[3])
        pad_x, pad_y = int(ew * 0.1), int(eh * 0.1)
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(w, x + ew + pad_x), min(h, y + eh + pad_y)
        return bgr_img[y0:y1, x0:x1], True
    else:
        # fallback: center 60% crop
        x0, x1 = int(w * 0.2), int(w * 0.8)
        y0, y1 = int(h * 0.2), int(h * 0.8)
        return bgr_img[y0:y1, x0:x1], False


def classify_eye_color(eye_bgr):
    """
    Heuristic classification using HSV color statistics:
    - Sclera (white-of-eye) region approximated as the brightest, low-saturation pixels
    - Pupil/iris region approximated as the darkest cluster in the center
    Returns a color_key matching CONDITION_RULES and a confidence estimate.
    """
    hsv = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # --- Sclera mask: bright + low saturation pixels ---
    sclera_mask = (v_ch > 140) & (s_ch < 90)
    sclera_pixels = hsv[sclera_mask]

    # --- Center region (approx pupil/iris) for cloudiness check ---
    h_img, w_img = eye_bgr.shape[:2]
    cy, cx = h_img // 2, w_img // 2
    r = min(h_img, w_img) // 4
    center_region = hsv[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
    center_brightness = float(np.mean(center_region[:, :, 2])) if center_region.size else 0

    result = {"color_key": "normal", "confidence": 55.0, "avg_hue": 0, "avg_sat": 0, "avg_val": 0}

    if sclera_pixels.size > 0:
        avg_hue = float(np.mean(sclera_pixels[:, 0]))
        avg_sat = float(np.mean(sclera_pixels[:, 1]))
        avg_val = float(np.mean(sclera_pixels[:, 2]))
        result.update({"avg_hue": avg_hue, "avg_sat": avg_sat, "avg_val": avg_val})

        # Yellow sclera: hue in yellow band (~20-35 in OpenCV's 0-179 scale) with decent saturation
        if 18 <= avg_hue <= 35 and avg_sat > 40:
            result["color_key"] = "yellow_sclera"
            result["confidence"] = min(95.0, 60 + avg_sat / 3)
        # Red sclera: hue near 0 or near 179 (red wraps around) with elevated saturation
        elif (avg_hue <= 10 or avg_hue >= 170) and avg_sat > 55:
            result["color_key"] = "red_sclera"
            result["confidence"] = min(95.0, 60 + avg_sat / 3)
        # Pale conjunctiva: very low saturation + very high brightness (washed out look)
        elif avg_sat < 25 and avg_val > 200:
            result["color_key"] = "pale_conjunctiva"
            result["confidence"] = min(90.0, 55 + (200 - avg_sat))
        else:
            result["color_key"] = "normal"
            result["confidence"] = 70.0

    # Cloudy pupil overrides if the center region is unusually bright (cataract-like haze)
    if center_brightness > 165 and result["color_key"] == "normal":
        result["color_key"] = "cloudy_pupil"
        result["confidence"] = min(90.0, 50 + (center_brightness - 165))

    return result


@app.route("/api/analyze-eye", methods=["POST"])
def analyze_eye():
    # --- Auth check ---
    auth_header = request.headers.get("X-API-KEY", "")
    if auth_header != API_KEY:
        return jsonify({"success": False, "error": "Invalid or missing API key"}), 401

    # --- Get image from multipart upload OR JSON base64 ---
    try:
        if "image" in request.files:
            bgr_img = decode_image(request.files["image"])
        elif request.is_json and "image_base64" in request.json:
            bgr_img = decode_image(request.json["image_base64"])
        else:
            return jsonify({"success": False, "error": "No image provided. Send multipart 'image' or JSON 'image_base64'."}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Could not read image: {str(e)}"}), 400

    eye_crop, eye_detected = detect_eye_region(bgr_img)
    classification = classify_eye_color(eye_crop)
    color_key = classification["color_key"]
    rule = CONDITION_RULES.get(color_key, CONDITION_RULES["normal"])

    response = {
        "success": True,
        "eye_region_detected": eye_detected,
        "detected_color": color_key,
        "predicted_condition": rule["condition"],
        "severity": rule["severity"],
        "recommendation": rule["recommendation"],
        "confidence_score": round(classification["confidence"], 2),
        "raw_metrics": {
            "avg_hue": round(classification["avg_hue"], 2),
            "avg_saturation": round(classification["avg_sat"], 2),
            "avg_brightness": round(classification["avg_val"], 2),
        },
        "disclaimer": (
            "This is an automated preliminary color-pattern screening tool, "
            "not a certified medical diagnosis. Please consult a licensed "
            "doctor for confirmation and treatment."
        ),
        "processed_at": int(time.time()),
    }
    return jsonify(response), 200


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "service": "nuskha-eye-scan-api"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)), debug=False)
