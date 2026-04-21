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

MODEL_PATH = 'waste_ENB7.keras'
model = None

CLASS_NAMES = [
    'Cardboard',
    'Food Organics',
    'Glass',
    'Metal',
    'Miscellaneous Trash',
    'Paper',
    'Plastic_Opaque',
    'Plastic_Transparent',
    'Textile Trash',
    'Vegetation',
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

def process_img(image):
    img = image.resize((224, 224))
    img_array = np.array(img)

    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = img_array.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/RealWaste')
def RealWaste():
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
            pred = model.predict(processedImg)
            print(f"Probability: {pred[0]}")

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