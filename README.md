# Disease Detection and Classification Flask Application

This application uses a trained machine learning model to detect and classify diseases based on uploaded images.

## Prerequisites

- Python 3.6 or higher
- Flask
- TensorFlow
- Pillow
- Flask-CORS
- Werkzeug

You can install these packages using pip:

```pip install flask tensorflow pillow flask-cors Werkzeug```


## Training the Models

The application uses two models: `disease_detection.h5` and `disease_classification.h5`. These models are trained using TensorFlow. Here is a general process for training these models:

1. Collect a dataset of images for each disease you want to detect and classify. The images should be labeled with the correct disease.

2. Split your dataset into a training set and a validation set.

3. Use TensorFlow to define and compile your model. You might use a pre-trained model like ResNet or VGG16, or define your own model architecture.

4. Train your model using your training set, and validate it using your validation set.

5. Once your model is trained and you're satisfied with its performance, save it using `model.save('disease_detection.h5')` or `model.save('disease_classification.h5')`.

Please note that the actual process may vary depending on your specific requirements and dataset.


## Running the Application

1. Clone the repository to your local machine.

2. Navigate to the project directory:

```cd path_to_your_directory```

3. Run the Flask application:

```python app.py```



The application will start running on `http://localhost:8080`.

## Using the Application

1. Open your web browser and navigate to `http://localhost:8080`.

2. You will see a form to upload an image. Click on "Select image to upload" and choose an image from your local machine.

3. Click on "Submit for Analysis". The image will be sent to the server, processed, and classified by the machine learning model.

4. The result of the classification will be displayed on a new page. The result includes the name of the detected disease and possible treatments.

## Note

Make sure the trained model files `disease_detection.h5` and `disease_classification.h5` are in the same directory as `app.py`. Also, ensure that the `hashmap.json` file, which maps class indices to disease information, is present in the same directory.