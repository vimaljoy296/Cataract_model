from flask import Flask, render_template, request
from urllib.parse import quote
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('C:/Cognitive Learning/project/models/New_cataract_model.h5')

def predict_cataract(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    return "Cataract" if prediction[0][0] > 0.5 else "No Cataract"

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST" and "photo" in request.files:
        photo = request.files["photo"]
        if photo:
            # Manipulate the filename to remove or replace unsafe characters
            filename = quote(photo.filename)
            filename = filename.replace("/", "_").replace("\\", "_")  # Example: Replace / and \ with underscores
            filepath = os.path.join("uploads", filename)
            photo.save(filepath)

            # Make a prediction
            prediction = predict_cataract(filepath)

            return render_template("result.html", filename=filename, prediction=prediction)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)