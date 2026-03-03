from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import uuid
import shutil

app = Flask(__name__)

# ✅ Force CPU (Render has no GPU)
model = YOLO("best.pt")
model.to("cpu")

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB limit


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html")

        # ✅ Unique filename to avoid overwrite
        file_ext = file.filename.split(".")[-1].lower()
        unique_name = f"{uuid.uuid4()}.{file_ext}"

        upload_path = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(upload_path)

        output_path = os.path.join(OUTPUT_FOLDER, unique_name)

        # ---------------- IMAGE DETECTION ---------------- #
        if file_ext in ["jpg", "jpeg", "png"]:

            results = model(upload_path, device="cpu")
            results[0].save(filename=output_path)

        # ---------------- VIDEO DETECTION ---------------- #
        elif file_ext in ["mp4", "avi", "mov"]:

            results = model.predict(
                source=upload_path,
                save=True,
                device="cpu"
            )

            save_dir = results[0].save_dir

            video_file = None
            for f in os.listdir(save_dir):
                if f.endswith((".mp4", ".avi", ".mov")):
                    video_file = f
                    break

            if video_file:
                yolo_output = os.path.join(save_dir, video_file)
                output_path = os.path.join(OUTPUT_FOLDER, unique_name)
                shutil.move(yolo_output, output_path)
            else:
                return "Video processing failed."

        else:
            return "Unsupported file format."

        return render_template(
            "index.html",
            output=output_path.replace("\\", "/")
        )

    except Exception as e:
        return f"Error: {str(e)}"

