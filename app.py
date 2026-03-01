from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import uuid
import shutil

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

model = YOLO(MODEL_PATH)

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:

        file = request.files["file"]

        filename = file.filename

        upload_path = os.path.join("static/uploads", filename)

        file.save(upload_path)

        file_ext = filename.split(".")[-1].lower()

        output_path = os.path.join("static/outputs", filename)


        if file_ext in ["jpg","jpeg","png"]:

            results = model(upload_path)

            results[0].save(filename=output_path)


        elif file_ext in ["mp4","avi","mov"]:

            results = model.predict(
                source=upload_path,
                save=True,
                project="static",
                name="outputs",
                exist_ok=True
            )

            for r in results:
                video_file = os.path.basename(r.path)
                output_path = "static/outputs/" + video_file


        return render_template(
            "index.html",
            output=output_path
        )

    except Exception as e:
        return str(e)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

