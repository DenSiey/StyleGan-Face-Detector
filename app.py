import os
import numpy as np
import secrets
import firebase_admin
import requests
import gdown
from flask import Flask, render_template, request, flash, redirect, url_for, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from firebase import firebase
from firebase_admin import credentials, storage
from io import BytesIO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# Initialize Firebase Admin SDK
cred = credentials.Certificate("./service_key.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'stylegan-330fe.appspot.com'})

app = Flask(__name__, static_folder='static')
app.secret_key = secrets.token_hex(16)

def create_model():
    input_shape = (255, 255, 3)
    input_layer = Input(shape=input_shape)
    epsilon=1e-7
    model = Sequential() 
    x = BatchNormalization()(input_layer)

    # Block 1
    x = Conv2D(16, kernel_size=(3, 3),  activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=3)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    # Block 2
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    
    # Block 3
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    # Block 4
    x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    # Block 5
    x = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    # Block 6
    x = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    # Block 7
    x = Conv2D(1024, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    # Final Dense layer
    x = Flatten()(x)
    output_layer = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Create the model
model = create_model()

# Load only the weights into the model
model.load_weights('model_weights.h5')

#model = load_model('model2.keras')

@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    try:
        if 'imagefile' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)

        imagefile = request.files['imagefile']

        if imagefile.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        # Upload image to Firebase Storage
        bucket = storage.bucket()
        blob = bucket.blob(imagefile.filename)
        blob.upload_from_file(imagefile, content_type=imagefile.content_type)

        # Get the public URL of the uploaded image
        image_url = blob.public_url
        
        # Extract filename from the URL
        _, image_filename = os.path.split(image_url) 
        
        # Download image data from URL
        image_data = blob.download_as_bytes()

        # Open the image data with PIL
        img = Image.open(BytesIO(image_data))
        img = img.resize((255, 255))
        img_array = np.array(img) / 255.0

        prediction_prob = model.predict(np.expand_dims(img_array, axis=0))[0][0]
        percentage_stylegan = prediction_prob * 100
        percentage_authentic = 100 - percentage_stylegan

        if percentage_stylegan > percentage_authentic:
            predicted_class = "StyleGan"
            confidence_level = percentage_stylegan
        else:
            predicted_class = "Authentic"
            confidence_level = percentage_authentic

        classification = f"{confidence_level:.2f}% {predicted_class} Face"
        stylegan_result = f"{percentage_stylegan:.2f}% - StyleGan"
        authentic_result = f"{percentage_authentic:.2f}% - Authentic"

        return render_template('index.html', selected_image="https://firebasestorage.googleapis.com/v0/b/stylegan-330fe.appspot.com/o/"+image_filename+"?alt=media&token=1f21cbc1-fa5c-44c2-aedd-9dbfb8b6bf0c", prediction=classification, result1=stylegan_result, result2=authentic_result)

    except requests.exceptions.ConnectionError:
        error_message = "Error: Unable to connect to the server. Please check your internet connection and try again."
        return render_template('index.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)