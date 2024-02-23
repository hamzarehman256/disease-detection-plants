from flask import Flask, request, jsonify, render_template_string,redirect,flash,url_for,render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import json
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from flask_cors import CORS
app = Flask(__name__)
with open('hashmap.json', 'r') as file:
    your_hashmap = json.load(file)

CORS(app, resources={r"/predict-classify": {"origins": "*"}}) 

# Load your trained model
# MODEL_PATH = 'disease_classification.h5'
# model = load_model(MODEL_PATH)
model = tf.keras.models.load_model('disease_detection.h5')

image_shape = (256, 256)

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=image_shape)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to create batch dimension
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Function to make predictions
def predict_disease(image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    return prediction



def prepare_image(img, target_size):
    """Preprocess the image for model prediction."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255  # Normalize the image if your model expects normalization
    return img_array


@app.route('/', methods=['GET', 'POST'])
def upload_form():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # Here, you can add your logic to save the file and/or process it
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # Redirect to a new URL or return a success message
            return redirect(url_for('success'))  # Define a 'success' route or modify as needed
    return render_template('form.html')

@app.route('/predict-classify', methods=['POST'])
def predict_classify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        # Read the image file to PIL Image
        img_path = 'temp_image.jpg'
        file.save(img_path)
        # Make prediction
        predictions = predict_disease(img_path)
        # image = Image.open(io.BytesIO(file.read()))
        # Preprocess the image and prepare it for classification
        # processed_image = preprocess_image(file.read())  # Adjust target_size as per your model's requirement

        # Predict
        # predictions = model.predict(processed_image)
        print(predictions)
        if type(predictions[0]) == list:
            predictions = predictions[0]
        predicted_class = np.argmax(predictions, axis=1)
        print('PredictedClass', predicted_class)

        # get info from hashmap
        index_str = str(predicted_class[0])
        
        info = your_hashmap[index_str]
        print(info , '    disease found')
        # Return the result
        print(index_str)
        if int(index_str) < 1 or int(index_str) >15 :
            return render_template('not-found.html')
        else:
            return render_template('results.html',result_data = info)
def prepare_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.expand_dims(image, axis=0)
    return image

# Load your trained model
classify_model = load_model('disease_classification.h5')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        # Read the image file to PIL Image
        img = Image.open(io.BytesIO(file.read()))

        # Preprocess the image and prepare it for classification
        processed_image = prepare_image(img, target_size=(256, 256))  # Adjust target_size as per your model's requirement

        # Predict
        predictions = classify_model.predict(processed_image)
        predicted_class = np.argmax(predictions, axis=1)
        print('PredictedClass', predicted_class)

        # Return the result
        return jsonify({'predicted_class': str(predicted_class[0])})


if __name__ == '__main__':
    app.run(debug=True, port=8080)