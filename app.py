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
from ultralytics import YOLO
import cv2

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

# Load YOLO model
yolo_model = None

def load_yolo():
    global yolo_model
    yolo_model = YOLO('yolov8n.pt') # automatically download
    print("YOLO loaded!")

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
        
@app.route('/detect', methods=['POST'])
def detect():
    if model is None or yolo_model is None:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image'}), 400

        file = request.files['image']
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        img_array = np.array(image)

        # Detect with YOLO
        results = yolo_model(img_array, classes=[0, 24, 25, 26, 28, 39, 40, 41, 45, 56, 57, 58, 59, 60, 62, 63, 64, 67, 73, 74, 76])
        
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get bounding box and classify with EfficientNetB7
            cropped = image.crop((x1, y1, x2, y2))
            processed = process_img(cropped)
            pred = model.predict(processed)
            class_idx = int(np.argmax(pred[0]))
            confidence = float(pred[0][class_idx])
            
            detections.append({
                'box': [x1, y1, x2, y2],
                'label': CLASS_NAMES[class_idx],
                'confidence': confidence,
                'yolo_confidence': float(box.conf[0])
            })

        return jsonify({'detections': detections})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    load_model()
    load_yolo()
    app.run(debug=True, host = '0.0.0.0', port = 5000)