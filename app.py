'''
# Note
app -> web app itself
__name__ -> parameter
'''
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app) # -> make it accessible from browsers

MODEL_PATH = 'file name comes here'
model = None

CLASS_NAMES = ['Vegetation',
               'Textile Trash',
               'Glass',
               'Plastic_Transparent',
               'Plastic_Opaque',
               'Paper',
               'Cardboard',
               'Metal',
               'Miscellaneous Trash',
               'Food Organics'
               ]

# Function to load model
def load_model():
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Successfully loaded a model!")
        print(f"input: {model.input_shape}")
        print(f"output: {model.output_shape}")
    except Exception as e:
        print(f"Error!: {e}")

# Image Processing → resize, rgb, 
def process_img(image):
    # Resize and image and convert it to numpy array
    img = image.resize((160, 160))
    img_array = np.array(img)

    # RGB - if image doesn't have 3 channel, make it three
    if img_array.shape[-1] == 4:
        img_array = img_array[:,:, :3]
    
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array) # Normalize to -1 to 1
    img_array = np.expand_dims(img_array, axis = 0)

    # Check
    print(f"Shape after preprocessing: {img_array.shape}")
    print(f"Range of value: {img_array.min():.2f} - {img_array.max():.2f}")
    print(f"Data type: {img_array.dtype}")

    return img_array
# # Test image preprocessing
# test = Image.open("test_img.png")
# processed = process_img(test)
# print(processed.shape)

@app.route('/WheatDisease')
def WheatDisease():
    return { 'status' : 'ok', 'model_loaded': model is not None}

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'No model loaded.'}), 500
    else:
        try:
            # Make sure is users upload image successfully
            if 'image' not in request.files:    # request.files -> store files sent from the front-end. Has a structure like dictionary
                return jsonify({'error': 'No image file found.'}), 400
            
            # Obtain image from request.files and read it as PIL image
            file = request.files['image']
            image = Image.open(io.BytesIO(file.read()))

            # Preprocess the image data
            processedImg = process_img(image)

            # Predict (Obtain logits)
            logits = model.predict(processedImg)
            print(f"Logits: {logits[0]}")

            # Convert logits into probability
            pred = tf.nn.softmax(logits).numpy()
            print(f"Probability: {pred[0]}")
            print(f"Sum: {pred[0].sum():.6f}")

            # Get predicted class
            pred_class_index = int(np.argmax(pred[0]))
            print(f"Predicted Index: {pred_class_index}")
            print(f"Number of Classes in CLASS_NAMES: {len(CLASS_NAMES)}")

            # Check index validity
            if pred_class_index >= len(CLASS_NAMES):
                err_msg = f"Index {pred_class_index} out of range for {len(CLASS_NAMES)}"
                print(f"Error: {err_msg}")
                return jsonify({'error' : err_msg}), 500
            
            pred_class = CLASS_NAMES[pred_class_index]
            confidence = float(pred[0][pred_class_index])

            print(f"Predicted class: {pred_class} ({confidence * 100:.2f}%)")

            # Create all predictions dict
            all_preds = {}
            for i in range(len(CLASS_NAMES)):
                if i < len(pred[0]):
                    all_preds[CLASS_NAMES[i]] = float(pred[0][i])
                else:
                    print(f"WARNING: Missing prediction for class {i}")
            
            print(f"All predictions: {all_preds}")

            result = {
                "class_index": pred_class_index,
                "label" : pred_class,
                'confidence': confidence,
                'all_predictions': all_preds
            }

            print(f"Returning result: {result}")
            return jsonify(result)
        
        except Exception as e:
            return jsonify({'error': f'Prediction Error: {str(e)}'}), 500

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host = '0.0.0.0', port = 5000)