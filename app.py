from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import uuid
import shutil



app = Flask(__name__)

model = YOLO("best.pt")

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]

    filename = file.filename

    upload_path = os.path.join("static/uploads", filename)

    file.save(upload_path)

    file_ext = filename.split(".")[-1].lower()

    output_path = os.path.join("static/outputs", filename)


    # IMAGE DETECTION
    if file_ext in ["jpg","jpeg","png"]:

        results = model(upload_path)

        results[0].save(filename=output_path)


    elif file_ext in ["mp4","avi","mov"]:

        results = model.predict(
        source=upload_path,
        save=True
        )

        save_dir = results[0].save_dir

        # Find the output video automatically
        files = os.listdir(save_dir)

        video_file = None
        for f in files:
            if f.endswith((".mp4",".avi",".mov")):
                video_file = f
            break

        yolo_output = os.path.join(save_dir, video_file)

        output_path = os.path.join("static/outputs", video_file)

        shutil.move(yolo_output, output_path)


    return render_template(
    "index.html",
    output=output_path.replace("\\","/")
)




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)





