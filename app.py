from flask import Flask, request, render_template, send_from_directory
import os
import cv2
import numpy as np
from ultralytics import YOLO
import uuid

# Load YOLOv8 model
model = YOLO("YOLOBD-best.pt")

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def run_inference(image_path, output_name):
    # Run YOLO inference
    results = model.predict(source=image_path, conf=0.5, save=False)

    # Draw masks/boxes on the original image
    annotated = results[0].plot()

    # Save annotated image to RESULTS_FOLDER
    output_path = os.path.join(RESULTS_FOLDER, output_name)
    cv2.imwrite(output_path, annotated)

    # Collect masks for area calculation
    masks = []
    for r in results:
        if r.masks is not None:
            for m in r.masks.xy:
                masks.append(m)

    return masks, output_name

def calculate_area(masks):
    total_area = 0
    for polygon in masks:
        contour = np.array(polygon, dtype=np.int32)
        area = cv2.contourArea(contour)
        total_area += area
    return len(masks), total_area

@app.route("/", methods=["GET", "POST"])
def index():
    report = None
    before_img = None
    after_img = None

    if request.method == "POST":
        before = request.files["before"]
        after = request.files["after"]

        # Save uploaded images
        before_path = os.path.join(UPLOAD_FOLDER, "before.png")
        after_path = os.path.join(UPLOAD_FOLDER, "after.png")
        before.save(before_path)
        after.save(after_path)

        # Generate unique names for outputs
        before_out = f"before_{uuid.uuid4().hex}.jpg"
        after_out = f"after_{uuid.uuid4().hex}.jpg"

        # Run YOLO segmentation
        before_masks, before_img = run_inference(before_path, before_out)
        after_masks, after_img = run_inference(after_path, after_out)

        # Compute stats
        before_count, before_area = calculate_area(before_masks)
        after_count, after_area = calculate_area(after_masks)

        report = {
            "before_buildings": before_count,
            "after_buildings": after_count,
            "before_area": before_area,
            "after_area": after_area,
            "buildings_lost": before_count - after_count,
            "area_lost": before_area - after_area,
        }

    return render_template("index.html", report=report,
                           before_img=before_img, after_img=after_img)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
