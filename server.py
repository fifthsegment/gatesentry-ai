import numpy as np
import tensorflow as tf
from PIL import Image
import time
from flask import Flask, request, jsonify
import io

interpreter = None
detection_categories = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
app = Flask(__name__)


def load_image(image_path):
    """Load and preprocess the image."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((299, 299))  # Adjusting the size to 299x299 as expected by the model
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def load_image_bytes(file):
    """Load and preprocess the image."""
    # convert numpy array to Image
    img = Image.open(file).convert('RGB')

    # img = Image.open(io.BytesIO(image_data)).convert('RGB')

    # img = Image.open(bytes).convert('RGB')
    img = img.resize((299, 299))  # Adjusting the size to 299x299 as expected by the model
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def run_inference(file):

    # measure time taken to load the model
    start = time.time()
    
    # Load and preprocess image
    input_data = load_image_bytes(file)

    # Set input tensor
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print("Duration to run inference = ", time.time() - start)

    return output_data



@app.route('/hello', methods=['GET', 'POST'])
def hello():
    return jsonify({'hello': 'world'})

@app.route('/infer', methods=['POST'])
def api():
    if request.method == 'POST':
        if len(request.files) > 0 :
            file = request.files['image']

            # log time it takes to process the image
            start = time.time()
            # image = Image.open(file)
            # image_array = np.array(image)


            # image_path = "test2.jpg"
            output = run_inference(file)
            # Find the index of the maximum probability
            predicted_index = np.argmax(output)

            # Map the index to its corresponding category
            predicted_category = detection_categories[predicted_index]

            # Extract the confidence value for the predicted category
            confidence = output[0][predicted_index]

            response = {
                "category": predicted_category,
                "confidence": float(confidence),  # Ensure it's JSON serializable
                "result": output.tolist(),  # Convert numpy array to list for JSON serialization
                "time": time.time() - start
            }

            return jsonify(response)
        else:
            return jsonify({'error': 'no file'})
        

# if __name__ == "__main__":
#     tflite_model_path = "model2019.tflite"
#     """Load TFLite model and run inference on the provided image."""
#     # Load TFLite model and allocate tensors
#     interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
#     interpreter.allocate_tensors()
#     app.run(debug=False)

tflite_model_path = "model2019.tflite"
"""Load TFLite model and run inference on the provided image."""
# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
print("Model loaded successfully")
# app.run(debug=False)


