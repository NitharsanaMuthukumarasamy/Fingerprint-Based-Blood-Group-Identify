from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model("model_blood_group_detection.keras")

# Class labels in correct order
CLASS_LABELS = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

# Folder to store uploaded images
UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form.get('name')
    dob = request.form.get('dob')
    uploaded_file = request.files.get('image')

    if uploaded_file and uploaded_file.filename != '':
        filepath = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
        uploaded_file.save(filepath)

        # Load and preprocess image
        img = image.load_img(filepath, target_size=(256, 256))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # ✅ Matches ResNet50 preprocessing

        # Make prediction
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_label = CLASS_LABELS[predicted_index]
        confidence = round(float(predictions[0][predicted_index]) * 100, 2)

        return render_template('result.html',
                               name=name,
                               dob=dob,
                               blood_group=predicted_label,
                               confidence=confidence,
                               image_path=filepath)
    
    return 'No image uploaded', 400

if __name__ == '__main__':
    app.run(debug=True)
