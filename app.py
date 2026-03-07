from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import uuid
import shutil

app = Flask(__name__)

# Load YOLO model safely
model = YOLO(os.path.join(os.getcwd(), "best.pt"))

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files.get("file")

    if file is None or file.filename == "":
        return "No file uploaded"

    # create unique filename
    ext = file.filename.split(".")[-1]
    filename = f"{uuid.uuid4()}.{ext}"

    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(upload_path)

    file_ext = ext.lower()
    output_path = os.path.join(OUTPUT_FOLDER, filename)

    # IMAGE DETECTION
    if file_ext in ["jpg", "jpeg", "png"]:

        results = model(upload_path)
        results[0].save(filename=output_path)

    # VIDEO DETECTION
    elif file_ext in ["mp4", "avi", "mov"]:

        results = model.predict(
            source=upload_path,
            save=True
        )

        save_dir = results[0].save_dir
        files = os.listdir(save_dir)

        video_file = None

        for f in files:
            if f.endswith((".mp4", ".avi", ".mov")):
                video_file = f
                break

        if video_file is None:
            return "Video processing failed"

        yolo_output = os.path.join(save_dir, video_file)
        output_path = os.path.join(OUTPUT_FOLDER, video_file)

        shutil.move(yolo_output, output_path)

    return render_template(
        "index.html",
        output=output_path.replace("\\", "/")
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
