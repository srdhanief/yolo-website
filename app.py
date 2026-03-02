from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import uuid
import shutil



app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")



model = YOLO(MODEL_PATH)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "static/outputs")


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("Request received")

        if "file" not in request.files:
            print("No file in request")
            return "No file uploaded"

        file = request.files["file"]

        if file.filename == "":
            print("Empty filename")
            return "Empty filename"

        file_ext = file.filename.split(".")[-1]
        unique_name = f"{uuid.uuid4()}.{file_ext}"

        upload_path = os.path.join(UPLOAD_FOLDER, unique_name)
        output_path = os.path.join(OUTPUT_FOLDER, unique_name)

        print("Saving file to:", upload_path)
        file.save(upload_path)

        print("Running model...")
        results = model(upload_path)

        print("Saving result to:", output_path)
        results[0].save(filename=output_path)

        print("Success")
        return render_template("index.html", output=output_path)

    except Exception as e:
        print("ERROR OCCURRED:", str(e))
        import traceback
        traceback.print_exc()
        return f"<h1>Error:</h1><pre>{str(e)}</pre>"



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)




